from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.storage import map_store

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SDR_SQLITE = ROOT / "backend" / "pipeline" / "data" / "sdr_ingest.db"


@dataclass
class CorrectionDecision:
    feature_id: str
    updated_feature: dict[str, Any]
    rationale: str
    geometry_delta_m: float
    confidence_delta: float


class AICorrectionLoop:
    """Background loop that proposes guarded feature corrections from recent SDR data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self.interval_s = 20.0
        self.track_recency_s = 600.0
        self.max_confidence_delta = 0.2
        self.max_geometry_delta_deg = 0.12
        self.min_apply_confidence = 0.35
        self.max_divergence_m = 50_000.0
        self.rollback_window = 5
        self._mutations: list[dict[str, Any]] = []
        self._last_run_at: str | None = None
        self._last_error: str = ""

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._running:
                return {"running": True, "message": "already_running"}
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True, name="ai-correction-loop")
            self._thread.start()
            return {"running": True, "message": "started"}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._running = False
        return {"running": False, "message": "stopped"}

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "interval_s": self.interval_s,
            "track_recency_s": self.track_recency_s,
            "max_confidence_delta": self.max_confidence_delta,
            "max_geometry_delta_deg": self.max_geometry_delta_deg,
            "min_apply_confidence": self.min_apply_confidence,
            "max_divergence_m": self.max_divergence_m,
            "last_run_at": self._last_run_at,
            "last_error": self._last_error,
        }

    def _run(self) -> None:
        while self._running:
            try:
                self.run_once()
                self._last_error = ""
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception("ai correction loop failed")
            time.sleep(self.interval_s)

    def _load_recent_tracks(self) -> list[dict[str, Any]]:
        if not SDR_SQLITE.exists():
            return []
        cutoff = datetime.now(timezone.utc).timestamp() - self.track_recency_s
        con = sqlite3.connect(SDR_SQLITE)
        try:
            rows = con.execute("SELECT payload_json FROM track_candidates").fetchall()
        finally:
            con.close()
        out = []
        for (payload_json,) in rows:
            try:
                payload = json.loads(payload_json)
                ts = datetime.fromisoformat(payload.get("timestamp", "").replace("Z", "+00:00")).timestamp()
            except Exception:
                continue
            if ts >= cutoff:
                out.append(payload)
        return out

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        from backend.foxhunt.auto_loop import event_bus

        event_bus.publish({"type": event_type, "ts": datetime.now(timezone.utc).isoformat(), **payload})

    def _propose(self, feature: dict[str, Any], tracks: list[dict[str, Any]]) -> CorrectionDecision | None:
        props = feature.get("properties", {})
        fid = props.get("id")
        if not fid:
            return None

        # Reclassification heuristics.
        best_track_conf = max((float(t.get("confidence", 0.0)) for t in tracks), default=0.0)
        current_conf = float(props.get("confidence", 0.5))
        raw_conf_delta = (best_track_conf - current_conf) * 0.5
        conf_delta = max(-self.max_confidence_delta, min(self.max_confidence_delta, raw_conf_delta))
        next_conf = max(0.0, min(0.99, current_conf + conf_delta))

        geom = feature.get("geometry", {})
        coords = list((geom.get("coordinates") or [0.0, 0.0])[:2])
        lon, lat = float(coords[0]), float(coords[1])

        # Geometry correction from averaged recent track positions.
        track_lats = [float(h["lat"]) for t in tracks for h in t.get("temporal_history", []) if h.get("lat") is not None]
        track_lons = [float(h["lon"]) for t in tracks for h in t.get("temporal_history", []) if h.get("lon") is not None]
        if track_lats and track_lons:
            target_lat = sum(track_lats) / len(track_lats)
            target_lon = sum(track_lons) / len(track_lons)
            dlat = max(-self.max_geometry_delta_deg, min(self.max_geometry_delta_deg, target_lat - lat))
            dlon = max(-self.max_geometry_delta_deg, min(self.max_geometry_delta_deg, target_lon - lon))
            lat += dlat
            lon += dlon
        geometry_delta_m = (((dlat if track_lats else 0.0) ** 2 + (dlon if track_lons else 0.0) ** 2) ** 0.5) * 111_000

        next_kind = props.get("kind", "speculative")
        if next_conf >= 0.82:
            next_kind = "estimate"
        elif next_conf < 0.2:
            next_kind = "speculative"

        if next_conf < self.min_apply_confidence:
            return None

        updated = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {**props, "id": fid, "kind": next_kind, "confidence": round(next_conf, 4)},
        }
        rationale = f"track_conf={best_track_conf:.3f}, conf_delta={conf_delta:.3f}, bounded_geo_delta_m={geometry_delta_m:.1f}"
        return CorrectionDecision(fid, updated, rationale, geometry_delta_m, conf_delta)

    def _rollback_if_divergent(self, feature_id: str) -> bool:
        recent = [m for m in self._mutations[-self.rollback_window:] if m.get("feature_id") == feature_id]
        if not recent:
            return False
        drift = sum(abs(float(m.get("geometry_delta_m", 0.0))) for m in recent)
        if drift <= self.max_divergence_m:
            return False
        original = next((m.get("before") for m in reversed(recent) if m.get("before")), None)
        if not original:
            return False
        map_store.upsert_feature(original)
        self._emit("ai_rationale", {"feature_id": feature_id, "action": "rollback", "reason": f"divergence {drift:.1f}m exceeds {self.max_divergence_m:.1f}m"})
        return True

    def run_once(self) -> dict[str, Any]:
        tracks = self._load_recent_tracks()
        features = map_store.get_uncertain_features(limit=25)
        applied = 0
        for feature in features:
            decision = self._propose(feature, tracks)
            if not decision:
                continue
            before = feature
            map_store.upsert_feature(decision.updated_feature)
            self._mutations.append({
                "feature_id": decision.feature_id,
                "geometry_delta_m": decision.geometry_delta_m,
                "confidence_delta": decision.confidence_delta,
                "before": before,
            })
            self._emit("map_mutation", {"feature_id": decision.feature_id, "action": "upsert", "feature": decision.updated_feature})
            self._emit("ai_rationale", {"feature_id": decision.feature_id, "action": "reclassify_or_correct", "rationale": decision.rationale})
            self._rollback_if_divergent(decision.feature_id)
            applied += 1
        self._last_run_at = datetime.now(timezone.utc).isoformat()
        return {"tracks": len(tracks), "features_considered": len(features), "mutations_applied": applied}


ai_correction_loop = AICorrectionLoop()
