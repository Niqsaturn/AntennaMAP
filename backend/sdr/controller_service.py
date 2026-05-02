from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Callable

from backend.sdr.base import BaseSdrAdapter, NormalizedSdrReading


@dataclass(frozen=True)
class SweepBand:
    name: str
    start_hz: float
    end_hz: float


class SdrControllerService:
    """Adaptive sweep controller that emits normalized observations per dwell window."""

    def __init__(
        self,
        adapter: BaseSdrAdapter,
        observation_sink: Callable[[dict[str, Any]], None],
        *,
        dwell_seconds: float = 0.25,
        min_dwell_seconds: float = 0.1,
        max_dwell_seconds: float = 2.0,
        quality_weight: float = 0.7,
        recency_weight: float = 0.3,
    ) -> None:
        self._adapter = adapter
        self._sink = observation_sink
        self._dwell_seconds = dwell_seconds
        self._min_dwell_seconds = min_dwell_seconds
        self._max_dwell_seconds = max_dwell_seconds
        self._quality_weight = quality_weight
        self._recency_weight = recency_weight
        self._capabilities = self._read_capabilities()
        self._bands = self._build_sweep_bands()
        self._band_state: dict[str, dict[str, Any]] = {
            band.name: {"quality": 0.5, "last_seen_at": None, "frames": 0} for band in self._bands
        }
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    def _read_capabilities(self) -> dict[str, Any]:
        center = self._adapter.config.get("center_freq_hz")
        sample_rate = self._adapter.config.get("sample_rate_hz")
        band_limits = self._adapter.config.get("band_limits_hz")

        if not band_limits:
            if center and sample_rate:
                half = float(sample_rate) / 2.0
                band_limits = [{"name": "default", "start_hz": float(center) - half, "end_hz": float(center) + half}]
            else:
                band_limits = [{"name": "default", "start_hz": 88e6, "end_hz": 108e6}]

        return {
            "band_limits_hz": band_limits,
            "max_span_hz": float(self._adapter.config.get("max_span_hz", sample_rate or 2.4e6)),
            "waterfall_resolution_bins": int(self._adapter.config.get("waterfall_resolution_bins", 64)),
        }

    def _build_sweep_bands(self) -> list[SweepBand]:
        out: list[SweepBand] = []
        for idx, band in enumerate(self._capabilities["band_limits_hz"]):
            out.append(
                SweepBand(
                    name=str(band.get("name") or f"band-{idx}"),
                    start_hz=float(band["start_hz"]),
                    end_hz=float(band["end_hz"]),
                )
            )
        return out

    def _band_recency_seconds(self, band_name: str) -> float:
        last_seen = self._band_state[band_name]["last_seen_at"]
        if not last_seen:
            return 1e6
        return max((datetime.now(timezone.utc) - last_seen).total_seconds(), 0.0)

    def _next_band(self) -> SweepBand:
        ranked = sorted(
            self._bands,
            key=lambda b: self._quality_weight * (1.0 - self._band_state[b.name]["quality"]) + self._recency_weight * min(self._band_recency_seconds(b.name) / 10.0, 1.0),
            reverse=True,
        )
        return ranked[0]

    def _make_sweep_plan(self, band: SweepBand) -> list[float]:
        max_span = self._capabilities["max_span_hz"]
        band_width = max(band.end_hz - band.start_hz, 1.0)
        steps = max(1, int((band_width + max_span - 1) // max_span))
        step_hz = band_width / steps
        return [band.start_hz + step_hz * (i + 0.5) for i in range(steps)]

    def _frame_quality(self, reading: NormalizedSdrReading) -> float:
        if not reading.psd_bins_db:
            return 0.0
        snr_score = 0.0 if reading.snr_db is None else max(min((reading.snr_db + 20.0) / 80.0, 1.0), 0.0)
        spread = max(reading.psd_bins_db) - min(reading.psd_bins_db)
        spread_score = max(min(spread / 60.0, 1.0), 0.0)
        return round((snr_score + spread_score) / 2.0, 4)

    def _capture_dwell(self, band: SweepBand, center_hz: float) -> dict[str, Any]:
        self._adapter.config["center_freq_hz"] = center_hz
        stop_at = time.monotonic() + self._dwell_seconds
        frames: list[NormalizedSdrReading] = []
        while time.monotonic() < stop_at:
            frames.append(self._adapter.read_normalized())

        frame_quality = mean(self._frame_quality(frame) for frame in frames) if frames else 0.0
        observed_at = datetime.now(timezone.utc)
        state = self._band_state[band.name]
        state["quality"] = 0.7 * state["quality"] + 0.3 * frame_quality
        state["last_seen_at"] = observed_at
        state["frames"] += len(frames)
        self._dwell_seconds = min(max(self._dwell_seconds * (1.2 if frame_quality > 0.7 else 0.9), self._min_dwell_seconds), self._max_dwell_seconds)

        return {
            "timestamp": observed_at.isoformat(),
            "band": band.name,
            "center_freq_hz": center_hz,
            "frame_count": len(frames),
            "quality": round(frame_quality, 4),
            "recency_seconds": 0.0,
            "capabilities": self._capabilities,
            "frames": [
                {
                    "timestamp": f.timestamp,
                    "provider": f.provider,
                    "center_freq_hz": f.center_freq_hz,
                    "sample_rate_hz": f.sample_rate_hz,
                    "rssi_dbm": f.rssi_dbm,
                    "snr_db": f.snr_db,
                    "waterfall_psd_bins_db": f.psd_bins_db,
                    "source_provenance": f.source_provenance,
                    "metadata": f.metadata,
                }
                for f in frames
            ],
        }

    def process_once(self) -> dict[str, Any]:
        band = self._next_band()
        plan = self._make_sweep_plan(band)
        emitted = 0
        for center in plan:
            observation = self._capture_dwell(band, center)
            self._sink(observation)
            emitted += 1
        return {"band": band.name, "planned_centers": len(plan), "emitted": emitted}

    def start(self, poll_interval_s: float = 0.0) -> None:
        if self._running:
            return
        self._running = True
        self._adapter.connect()

        def _loop() -> None:
            while self._running:
                with self._lock:
                    self.process_once()
                if poll_interval_s > 0:
                    time.sleep(poll_interval_s)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._adapter.disconnect()

    def quality_state(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for band, state in self._band_state.items():
            out[band] = {
                "quality": state["quality"],
                "frame_count": state["frames"],
                "last_seen_at": state["last_seen_at"].isoformat() if state["last_seen_at"] else None,
                "recency_seconds": self._band_recency_seconds(band),
            }
        return out
