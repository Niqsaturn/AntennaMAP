"""Multi-band spectrum mosaic: stitches KiwiSDR (HF) and OpenWebRX (VHF/UHF).

Produces a MosaicFrame — a sorted list of BandSegments covering the full
observable spectrum — published as SSE 'spectrum_mosaic' events every 10s.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_MOSAIC_INTERVAL_S = 10.0


@dataclass
class BandSegment:
    center_freq_hz: float
    bandwidth_hz: float
    bins_db: list[float]
    source: str         # "kiwisdr" | "openwebrx"
    node: str = ""


@dataclass
class MosaicFrame:
    ts: float
    segments: list[BandSegment] = field(default_factory=list)

    def sorted_segments(self) -> list[BandSegment]:
        return sorted(self.segments, key=lambda s: s.center_freq_hz)


class SpectrumMosaic:
    """Accumulates SDR frames and serves a combined mosaic on demand."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame: MosaicFrame = MosaicFrame(ts=0.0)
        # Recent raw frames keyed by source+node; updated by scan methods
        self._hf_bins: dict[str, list[float]] = {}
        self._hf_center: dict[str, float] = {}
        self._hf_bw: dict[str, float] = {}
        self._vhf_bins: dict[str, list[float]] = {}
        self._vhf_center: dict[str, float] = {}
        self._vhf_bw: dict[str, float] = {}

    def ingest_kiwisdr_frame(
        self,
        node_id: str,
        center_freq_hz: float,
        bandwidth_hz: float,
        bins_db: list[float],
    ) -> None:
        with self._lock:
            self._hf_bins[node_id] = bins_db
            self._hf_center[node_id] = center_freq_hz
            self._hf_bw[node_id] = bandwidth_hz

    def ingest_openwebrx_frame(
        self,
        node_id: str,
        center_freq_hz: float,
        bandwidth_hz: float,
        bins_db: list[float],
    ) -> None:
        with self._lock:
            self._vhf_bins[node_id] = bins_db
            self._vhf_center[node_id] = center_freq_hz
            self._vhf_bw[node_id] = bandwidth_hz

    def get_latest(self) -> MosaicFrame:
        return self._build_frame()

    def _build_frame(self) -> MosaicFrame:
        with self._lock:
            hf = dict(self._hf_bins)
            hf_c = dict(self._hf_center)
            hf_b = dict(self._hf_bw)
            vhf = dict(self._vhf_bins)
            vhf_c = dict(self._vhf_center)
            vhf_b = dict(self._vhf_bw)

        segments: list[BandSegment] = []

        # HF segments from KiwiSDR nodes
        for nid, bins in hf.items():
            if bins:
                segments.append(BandSegment(
                    center_freq_hz=hf_c.get(nid, 15e6),
                    bandwidth_hz=hf_b.get(nid, 30e6),
                    bins_db=bins[:1024],
                    source="kiwisdr",
                    node=nid,
                ))

        # VHF/UHF segments from OpenWebRX nodes
        for nid, bins in vhf.items():
            if bins:
                segments.append(BandSegment(
                    center_freq_hz=vhf_c.get(nid, 145e6),
                    bandwidth_hz=vhf_b.get(nid, 2.4e6),
                    bins_db=bins[:1024],
                    source="openwebrx",
                    node=nid,
                ))

        # If no real data, produce a synthetic HF placeholder segment
        if not segments:
            segments.append(_synthetic_hf_segment())

        return MosaicFrame(ts=time.time(), segments=sorted(segments, key=lambda s: s.center_freq_hz))

    def to_sse_payload(self) -> dict:
        frame = self.get_latest()
        return {
            "type": "spectrum_mosaic",
            "ts": frame.ts,
            "segments": [
                {
                    "center_freq_hz": s.center_freq_hz,
                    "bandwidth_hz": s.bandwidth_hz,
                    "bins_db": s.bins_db,
                    "source": s.source,
                    "node": s.node,
                }
                for s in frame.segments
            ],
        }


def _synthetic_hf_segment() -> BandSegment:
    """Minimal synthetic segment so the frontend always has something to render."""
    import math
    bins = [
        -90.0 + 5.0 * math.sin(i * 0.05) + (i % 50) * 0.1
        for i in range(1024)
    ]
    return BandSegment(
        center_freq_hz=15e6,
        bandwidth_hz=30e6,
        bins_db=bins,
        source="synthetic",
        node="placeholder",
    )


# Module-level singleton
spectrum_mosaic = SpectrumMosaic()


async def spectrum_mosaic_loop() -> None:
    """Publish spectrum_mosaic SSE events every _MOSAIC_INTERVAL_S seconds."""
    logger.info("spectrum_mosaic_loop: starting")
    while True:
        try:
            _publish_mosaic()
        except Exception as exc:
            logger.debug("spectrum_mosaic_loop: %s", exc)
        await asyncio.sleep(_MOSAIC_INTERVAL_S)


def _publish_mosaic() -> None:
    from backend.foxhunt.auto_loop import event_bus
    payload = spectrum_mosaic.to_sse_payload()
    event_bus.publish(payload)
