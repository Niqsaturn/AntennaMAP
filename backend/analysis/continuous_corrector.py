"""Self-correction feedback loop for confirmed fox hunt features.

Maintains a watch list of confirmed emitters and periodically re-scans
their frequencies across all nodes. Position updates > drift_threshold_m
are applied via map_store upsert + Kalman filter. Absence > absence_timeout_s
demotes the feature to speculative.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_REVERIFY_INTERVAL_S = 120.0
_DRIFT_THRESHOLD_M = 200.0
_ABSENCE_TIMEOUT_S = 300.0


@dataclass
class _WatchEntry:
    feature_id: str
    freq_hz: float
    last_seen: float = field(default_factory=time.monotonic)
    last_verified: float = field(default_factory=time.monotonic)


class ContinuousCorrector:
    """Background thread that periodically re-verifies confirmed features."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._watched: dict[str, _WatchEntry] = {}
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="corrector"
        )
        self._thread.start()

    def watch(self, feature_id: str, freq_hz: float) -> None:
        with self._lock:
            self._watched[feature_id] = _WatchEntry(feature_id, freq_hz)
        if not self._running:
            self.start()

    def unwatch(self, feature_id: str) -> None:
        with self._lock:
            self._watched.pop(feature_id, None)

    def _run(self) -> None:
        while self._running:
            try:
                self._check_all()
            except Exception as exc:
                logger.debug("corrector: check_all error: %s", exc)
            time.sleep(10.0)

    def _check_all(self) -> None:
        now = time.monotonic()
        with self._lock:
            entries = list(self._watched.values())

        for entry in entries:
            age = now - entry.last_verified
            if age < _REVERIFY_INTERVAL_S:
                continue
            try:
                self._verify_one(entry)
            except Exception as exc:
                logger.debug("corrector: verify %s: %s", entry.feature_id, exc)

    def _verify_one(self, entry: _WatchEntry) -> None:
        from backend.sdr.kiwisdr_client import node_pool
        from backend.foxhunt.multilateration import RSSIObs, locate_transmitter

        readings = node_pool.scan_rssi(entry.freq_hz)
        gps_readings = [r for r in readings if r.get("lat") and r.get("lon")]

        with self._lock:
            entry.last_verified = time.monotonic()

        if not gps_readings:
            # No signal detected — check absence timeout
            silence = time.monotonic() - entry.last_seen
            if silence > _ABSENCE_TIMEOUT_S:
                self._demote_feature(entry.feature_id)
            return

        # Signal still present — update last_seen
        with self._lock:
            entry.last_seen = time.monotonic()

        if len(gps_readings) < 2:
            return

        rssi_obs = [
            RSSIObs(
                lat=float(r["lat"]), lon=float(r["lon"]),
                rssi_dbm=float(r["rssi_dbm"]), freq_hz=entry.freq_hz,
            )
            for r in gps_readings
        ]
        try:
            fix = locate_transmitter(rssi_obs=rssi_obs, freq_hz=entry.freq_hz)
        except Exception:
            return

        self._maybe_update_position(entry.feature_id, fix.lat, fix.lon, fix.uncertainty_m)

    def _maybe_update_position(
        self,
        feature_id: str,
        new_lat: float,
        new_lon: float,
        uncertainty_m: float,
    ) -> None:
        from backend.storage.map_store import upsert_feature, get_all_features
        from backend.analysis.kalman_tracker import PositionKalman

        try:
            fc = get_all_features(limit=5000)
            existing = next(
                (f for f in fc.get("features", [])
                 if f.get("properties", {}).get("id") == feature_id),
                None,
            )
        except Exception:
            existing = None

        if existing:
            coords = existing.get("geometry", {}).get("coordinates", [])
            if len(coords) == 2:
                old_lon, old_lat = float(coords[0]), float(coords[1])
                delta_m = _haversine_m(old_lat, old_lon, new_lat, new_lon)
                if delta_m < _DRIFT_THRESHOLD_M:
                    return
        else:
            delta_m = 0.0

        kf = PositionKalman.for_feature(feature_id, new_lat, new_lon)
        kf.predict()
        s_lat, s_lon, s_unc = kf.update(new_lat, new_lon, uncertainty_m)
        kf.save(feature_id)

        ts = datetime.now(timezone.utc).isoformat()
        feat = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [s_lon, s_lat]},
            "properties": {
                "id": feature_id,
                "kalman_uncertainty_m": round(s_unc, 1),
                "last_corrected": ts,
                "correction_delta_m": round(delta_m, 1),
            },
        }
        try:
            upsert_feature(feat)
        except Exception as exc:
            logger.debug("corrector: upsert %s: %s", feature_id, exc)
            return

        # Publish correction event to SSE
        try:
            from backend.foxhunt.auto_loop import event_bus
            event_bus.publish({
                "type": "feature_corrected",
                "ts": ts,
                "feature_id": feature_id,
                "lat": s_lat,
                "lon": s_lon,
                "delta_m": round(delta_m, 1),
                "uncertainty_m": round(s_unc, 1),
            })
        except Exception:
            pass

        logger.info(
            "corrector: updated %s — moved %.0fm, uncertainty=%.0fm",
            feature_id, delta_m, s_unc,
        )

    def _demote_feature(self, feature_id: str) -> None:
        try:
            from backend.storage.map_store import upsert_feature
            ts = datetime.now(timezone.utc).isoformat()
            upsert_feature({
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "id": feature_id,
                    "kind": "speculative",
                    "demoted_at": ts,
                    "demote_reason": "signal_absent",
                },
            })
        except Exception:
            pass
        with self._lock:
            self._watched.pop(feature_id, None)
        try:
            from backend.foxhunt.auto_loop import event_bus
            event_bus.publish({
                "type": "feature_demoted",
                "ts": datetime.now(timezone.utc).isoformat(),
                "feature_id": feature_id,
                "reason": "signal_absent",
            })
        except Exception:
            pass
        logger.info("corrector: demoted %s (signal absent)", feature_id)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Module-level singleton
corrector = ContinuousCorrector()
