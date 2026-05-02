"""Continuous single-operator fox hunt state machine.

State transitions:
  IDLE → SCANNING → ACQUIRING → COLLECTING → SOLVING → CONFIRMED → SCANNING

At each state:
  SCANNING   — poll all configured KiwiSDR nodes; find the strongest unknown signal
  ACQUIRING  — lock onto the target frequency; gather RSSI from all nodes
  COLLECTING — accumulate bearing + RSSI observations as the operator moves
  SOLVING    — run locate_transmitter() with all collected observations
  CONFIRMED  — write GeoJSON feature to map_store; advance to next-strongest signal

Events are published to a queue consumed by the SSE endpoint.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_BASE_PIN_CONFIDENCE_THRESHOLD = 0.30
_SINGLE_DIRECTIONAL_PIN_CONFIDENCE_THRESHOLD = 0.85

from backend.foxhunt.multilateration import (
    BearingObs,
    FusedFix,
    RSSIObs,
    TDOAObs,
    ellipse_polygon,
    locate_transmitter,
)
from backend.sdr.kiwisdr_client import node_pool


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class FoxTarget:
    """One signal being tracked."""
    freq_hz: float
    band_label: str
    modulation_hint: str
    first_seen: str
    rssi_obs: list[RSSIObs] = field(default_factory=list)
    bearing_obs: list[BearingObs] = field(default_factory=list)
    tdoa_obs: list[TDOAObs] = field(default_factory=list)
    fix: FusedFix | None = None
    confirmed: bool = False
    feature_id: str = ""

    @property
    def observation_count(self) -> int:
        return len(self.rssi_obs) + len(self.bearing_obs)

    @property
    def best_rssi_dbm(self) -> float | None:
        if self.rssi_obs:
            return max(o.rssi_dbm for o in self.rssi_obs)
        return None


@dataclass
class HuntState:
    state: str = "IDLE"
    target: FoxTarget | None = None
    confirmed_targets: list[FoxTarget] = field(default_factory=list)
    scan_results: list[dict] = field(default_factory=list)
    last_event: str = ""
    started_at: str = ""
    cycles: int = 0
    error: str = ""


# ── Event bus (consumed by SSE endpoint) ─────────────────────────────────────

class EventBus:
    def __init__(self) -> None:
        self._listeners: list[Any] = []   # asyncio.Queue objects added at runtime
        self._lock = threading.Lock()
        self._loop: Any = None            # asyncio event loop, set at startup
        self._sdr_frames_emitted = 0
        self._sdr_schema_rejects = 0
        self._sdr_last_frame_ts = ""

    def set_loop(self, loop: Any) -> None:
        self._loop = loop

    def subscribe(self, q: Any) -> None:
        with self._lock:
            self._listeners.append(q)

    def unsubscribe(self, q: Any) -> None:
        with self._lock:
            try:
                self._listeners.remove(q)
            except ValueError:
                pass

    def publish(self, event: dict) -> None:
        """Thread-safe publish to all SSE subscribers."""
        if event.get("type") == "sdr_frame":
            try:
                from backend.sdr.events import SdrFrameEvent
                model = SdrFrameEvent.validate_payload(event)
                event = model.model_dump()
                with self._lock:
                    self._sdr_frames_emitted += 1
                    self._sdr_last_frame_ts = model.timestamp
            except Exception:
                with self._lock:
                    self._sdr_schema_rejects += 1
                return
        if not self._loop or not self._listeners:
            return
        import asyncio
        with self._lock:
            listeners = list(self._listeners)
        for q in listeners:
            try:
                asyncio.run_coroutine_threadsafe(q.put(event), self._loop)
            except Exception:
                pass

    def sdr_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "frames_emitted": self._sdr_frames_emitted,
                "schema_rejects": self._sdr_schema_rejects,
                "last_frame_timestamp": self._sdr_last_frame_ts or None,
            }


event_bus = EventBus()


def _emit(etype: str, data: dict) -> None:
    event_bus.publish({"type": etype, "ts": datetime.now(timezone.utc).isoformat(), **data})


# ── Fox hunt auto loop ────────────────────────────────────────────────────────

class AutoFoxHuntLoop:
    """Continuous fox hunting state machine.

    Run flow per cycle:
      1. SCANNING:   scan all KiwiSDR nodes for signal peaks
      2. ACQUIRING:  pick the strongest unknown peak as target;
                     collect multi-node RSSI for that frequency
      3. COLLECTING: wait for the operator to add bearing observations
                     (or auto-progress after enough RSSI readings)
      4. SOLVING:    run the full multilateration solver
      5. CONFIRMED:  write result to map_store, mark target confirmed,
                     move to next unknown signal

    The loop runs in a background thread.  The frontend adds bearing
    observations in real-time via POST /api/foxhunt/auto/observe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = HuntState()
        self._thread: threading.Thread | None = None
        self._running = False
        # Operator GPS position (updated from API)
        self._op_lat: float = 0.0
        self._op_lon: float = 0.0
        # Frequencies already confirmed (don't revisit)
        self._confirmed_freqs: set[float] = set()
        # Scan interval
        self.scan_interval_s: float = 15.0
        # Min observations before attempting solve
        self.min_obs_to_solve: int = 2
        self.min_auto_bearing_snr_db: float = 3.0

    # ── External control ──────────────────────────────────────────────────────

    def _start_idle(self) -> None:
        """Start the background thread in IDLE state (thread alive, no scanning)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_idle, daemon=True, name="foxhunt-idle")
        self._thread.start()

    def _run_idle(self) -> None:
        """Keep thread alive in IDLE until start() is called."""
        while self._running:
            with self._lock:
                state = self._state.state
            if state != "IDLE":
                self._run()
                return
            import time as _t
            _t.sleep(1.0)

    def start(self, op_lat: float = 0.0, op_lon: float = 0.0) -> None:
        with self._lock:
            self._op_lat = op_lat
            self._op_lon = op_lon
            self._state = HuntState(
                state="SCANNING",
                started_at=datetime.now(timezone.utc).isoformat(),
            )
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True, name="foxhunt-auto")
            self._thread.start()
        _emit("fox_state", {"state": "SCANNING", "message": "Fox hunt started"})

    def stop(self) -> None:
        self._running = False
        with self._lock:
            self._state.state = "IDLE"
        _emit("fox_state", {"state": "IDLE", "message": "Fox hunt stopped"})

    def update_operator_position(self, lat: float, lon: float) -> None:
        with self._lock:
            self._op_lat = lat
            self._op_lon = lon

    def add_bearing_observation(
        self,
        bearing_deg: float,
        snr_db: float,
        freq_hz: float | None = None,
        source: str = "manual",
    ) -> dict:
        """Add a manual bearing reading from the operator's current position."""
        with self._lock:
            target = self._state.target
            if target is None:
                return {"ok": False, "reason": "no active target"}
            obs_freq = freq_hz or target.freq_hz
            bearing_deg = bearing_deg % 360.0
            obs = BearingObs(
                lat=self._op_lat, lon=self._op_lon,
                bearing_deg=bearing_deg, snr_db=snr_db,
                freq_hz=obs_freq,
                source=source,
            )
            target.bearing_obs.append(obs)
            n = len(target.bearing_obs)
        _emit("bearing_added", {
            "lat": self._op_lat, "lon": self._op_lon,
            "bearing_deg": bearing_deg, "snr_db": snr_db,
            "freq_hz": obs_freq, "total_bearings": n, "source": source,
        })
        # Trigger solve if we have enough geometry
        if n >= self.min_obs_to_solve:
            threading.Thread(target=self._try_solve, daemon=True).start()
        return {"ok": True, "total_bearings": n}


    def _policy_snapshot(self) -> dict:
        """Best-effort snapshot of autonomous policy runtime state."""
        try:
            from backend.foxhunt.policy import auto_policy
            policy_status = auto_policy.status()
            decision = policy_status.get("last_decision") or {}
            return {
                "phase": policy_status.get("phase", "IDLE"),
                "last_action": decision.get("phase"),
                "action_confidence": decision.get("confidence"),
            }
        except Exception:
            return {"phase": "IDLE", "last_action": None, "action_confidence": None}

    def status(self) -> dict:
        with self._lock:
            st = self._state
            target_info = None
            if st.target:
                t = st.target
                target_info = {
                    "freq_hz": t.freq_hz,
                    "band_label": t.band_label,
                    "modulation_hint": t.modulation_hint,
                    "rssi_obs_count": len(t.rssi_obs),
                    "bearing_obs_count": len(t.bearing_obs),
                    "confirmed": t.confirmed,
                    "fix": _fix_to_dict(t.fix) if t.fix else None,
                }
            policy = self._policy_snapshot()
            return {
                "running": self._running,
                "state": st.state,
                "cycles": st.cycles,
                "op_lat": self._op_lat,
                "op_lon": self._op_lon,
                "target": target_info,
                "confirmed_count": len(st.confirmed_targets),
                "last_event": st.last_event,
                "error": st.error,
                "policy_phase": policy["phase"],
                "policy_last_action": policy["last_action"],
                "policy_action_confidence": policy["action_confidence"],
            }

    def confirmed_features(self) -> list[dict]:
        """Return all confirmed targets as GeoJSON features."""
        with self._lock:
            targets = list(self._state.confirmed_targets)
        return [_target_to_feature(t) for t in targets if t.fix]

    # ── Background loop ───────────────────────────────────────────────────────

    def _run(self) -> None:
        while self._running:
            try:
                with self._lock:
                    state = self._state.state
                    self._state.cycles += 1
                    cycles = self._state.cycles

                self._execute_policy_cycle(cycles=cycles, state=state)

                if state == "SCANNING":
                    self._do_scan()
                elif state == "ACQUIRING":
                    self._do_acquire()
                elif state == "COLLECTING":
                    # Wait for manual bearings; auto-advance after timeout
                    time.sleep(self.scan_interval_s)
                    with self._lock:
                        target = self._state.target
                    if target and target.observation_count >= self.min_obs_to_solve:
                        self._try_solve()
                    elif target is None:
                        with self._lock:
                            self._state.state = "SCANNING"
                elif state == "SOLVING":
                    time.sleep(0.5)   # solve is called in _try_solve
                elif state == "CONFIRMED":
                    self._advance_to_next()
                elif state == "IDLE":
                    time.sleep(1.0)
                    break
                else:
                    time.sleep(1.0)

            except Exception as exc:
                with self._lock:
                    self._state.error = str(exc)
                time.sleep(5.0)


    def _execute_policy_cycle(self, cycles: int, state: str) -> None:
        """Run autonomous policy once per loop cycle and emit telemetry."""
        try:
            from backend.foxhunt.policy import auto_policy
            result = auto_policy.execute_cycle()
            decision = result.get("decision") or {}
            _emit("policy_cycle", {
                "cycle": cycles,
                "loop_state": state,
                "ok": result.get("ok", False),
                "reason": result.get("reason"),
                "policy_phase": result.get("phase", auto_policy.phase),
                "policy_last_action": decision.get("phase"),
                "policy_action_confidence": decision.get("confidence"),
                "promoted_target": result.get("promoted_target", False),
            })
        except Exception as exc:
            logger.debug("auto_loop: policy cycle skipped: %s", exc)

    def _do_scan(self) -> None:
        """Poll all KiwiSDR nodes for signal peaks."""
        _emit("fox_state", {"state": "SCANNING", "message": "Scanning bands…"})
        peaks = node_pool.scan_peaks_all(min_snr_db=6.0)

        # Deduplicate by frequency (within ±5 kHz)
        deduped: list[dict] = []
        for p in sorted(peaks, key=lambda x: x["snr_db"], reverse=True):
            if not any(abs(p["freq_hz"] - d["freq_hz"]) < 5_000 for d in deduped):
                deduped.append(p)

        # Filter out already-confirmed frequencies
        novel = [
            p for p in deduped
            if not any(abs(p["freq_hz"] - cf) < 10_000 for cf in self._confirmed_freqs)
        ]

        with self._lock:
            self._state.scan_results = novel[:20]

        _emit("scan_results", {"peaks": novel[:20], "total": len(peaks)})

        if novel:
            # Pick the strongest novel peak
            best = max(novel, key=lambda x: x["snr_db"])
            self._start_target(best)
        else:
            _emit("fox_state", {"state": "SCANNING", "message": "No novel signals, retrying…"})
            time.sleep(self.scan_interval_s)

    def _start_target(self, peak: dict) -> None:
        """Begin tracking a signal peak as the current fox hunt target."""
        freq = float(peak["freq_hz"])
        band_label = _freq_to_band(freq)
        target = FoxTarget(
            freq_hz=freq,
            band_label=band_label,
            modulation_hint=peak.get("modulation_hint", "unknown"),
            first_seen=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._state.target = target
            self._state.state = "ACQUIRING"
        _emit("fox_state", {
            "state": "ACQUIRING",
            "freq_hz": freq,
            "band_label": band_label,
            "message": f"Acquired {freq/1e6:.3f} MHz ({band_label})",
        })

    def _do_acquire(self) -> None:
        """Query all nodes for RSSI at the target frequency."""
        with self._lock:
            target = self._state.target
        if target is None:
            with self._lock:
                self._state.state = "SCANNING"
            return

        readings = node_pool.scan_rssi(target.freq_hz)
        nodes_with_gps = [r for r in readings if r.get("lat") and r.get("lon")]

        with self._lock:
            for r in nodes_with_gps:
                target.rssi_obs.append(RSSIObs(
                    lat=float(r["lat"]), lon=float(r["lon"]),
                    rssi_dbm=float(r["rssi_dbm"]),
                    freq_hz=target.freq_hz,
                ))
            self._state.state = "COLLECTING"

        _emit("rssi_acquired", {
            "freq_hz": target.freq_hz,
            "node_count": len(readings),
            "gps_nodes": len(nodes_with_gps),
            "readings": readings,
        })
        _emit("fox_state", {
            "state": "COLLECTING",
            "message": f"{len(nodes_with_gps)} node(s) with GPS. Auto-generating bearings (manual override optional).",
        })

    def _auto_add_bearings_from_sdr(self, target: FoxTarget) -> int:
        """Generate synthetic bearings from SDR node geometry + RSSI evidence."""
        if not target.rssi_obs:
            return 0

        # Prefer high-SNR/strong readings to avoid noisy synthetic geometry.
        best_rssi = max(o.rssi_dbm for o in target.rssi_obs)
        threshold = best_rssi - 8.0
        candidates = [o for o in target.rssi_obs if o.rssi_dbm >= threshold]
        if len(candidates) < 2:
            return 0

        # Weighted centroid estimate of likely TX location.
        weights = [max(1e-3, math.pow(10.0, o.rssi_dbm / 10.0)) for o in candidates]
        wsum = sum(weights)
        lat_est = sum(w * o.lat for w, o in zip(weights, candidates)) / wsum
        lon_est = sum(w * o.lon for w, o in zip(weights, candidates)) / wsum

        added = 0
        for o in candidates:
            # Bearing from node -> centroid estimate.
            dlon = math.radians(lon_est - o.lon)
            lat1 = math.radians(o.lat)
            lat2 = math.radians(lat_est)
            y = math.sin(dlon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
            bearing_deg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
            snr_proxy = max(0.0, o.rssi_dbm - threshold)
            if snr_proxy < self.min_auto_bearing_snr_db:
                continue

            target.bearing_obs.append(BearingObs(
                lat=o.lat,
                lon=o.lon,
                bearing_deg=bearing_deg,
                snr_db=snr_proxy,
                freq_hz=target.freq_hz,
                source="sdr_auto",
            ))
            added += 1
        return added

    def _try_solve(self) -> None:
        """Run the multilateration solver on current observations."""
        with self._lock:
            target = self._state.target
            if target is None:
                return
            auto_added = self._auto_add_bearings_from_sdr(target)
            if auto_added:
                _emit("bearing_added", {
                    "freq_hz": target.freq_hz,
                    "total_bearings": len(target.bearing_obs),
                    "source": "sdr_auto",
                    "auto_added": auto_added,
                })

            # Quality gate: need enough total evidence, with at least one meaningful bearing.
            rssi_count = len(target.rssi_obs)
            bearing_count = len(target.bearing_obs)
            strong_bearings = [o for o in target.bearing_obs if o.snr_db >= self.min_auto_bearing_snr_db or o.source == "manual"]
            if (rssi_count + bearing_count) < self.min_obs_to_solve or not strong_bearings:
                self._state.state = "COLLECTING"
                return
            self._state.state = "SOLVING"

        _emit("fox_state", {"state": "SOLVING", "message": "Running triangulation…"})
        try:
            fix = locate_transmitter(
                rssi_obs=list(target.rssi_obs),
                bearing_obs=list(target.bearing_obs),
                tdoa_obs=list(target.tdoa_obs),
                freq_hz=target.freq_hz,
            )
            with self._lock:
                target.fix = fix
                self._state.state = "CONFIRMED"

            _emit("estimate_updated", {
                "freq_hz": target.freq_hz,
                "lat": fix.lat, "lon": fix.lon,
                "uncertainty_m": fix.uncertainty_m,
                "confidence": fix.confidence,
                "methods": fix.methods_used,
                "ellipse_major_m": fix.ellipse_major_m,
                "ellipse_minor_m": fix.ellipse_minor_m,
                "observability_class": _observability_class(target, fix),
                "certainty_level": _certainty_level(fix.confidence),
            })

            confidence_threshold = _pin_confidence_threshold(target)
            if fix.confidence >= confidence_threshold:
                self._confirm_target(target)
            else:
                _emit("target_pending", {
                    "freq_hz": target.freq_hz,
                    "lat": fix.lat,
                    "lon": fix.lon,
                    "confidence": fix.confidence,
                    "required_confidence": confidence_threshold,
                    "reason": "uncertainty_first_overlay_mode",
                    "observability_class": _observability_class(target, fix),
                    "certainty_level": _certainty_level(fix.confidence),
                })
        except Exception as exc:
            logger.warning("auto_loop: solve failed: %s", exc)
            with self._lock:
                self._state.state = "COLLECTING"
                self._state.error = str(exc)

    def _confirm_target(self, target: FoxTarget) -> None:
        """Write confirmed target to map_store and emit event."""
        fix = target.fix
        if fix is None:
            return

        import hashlib
        fid = "fox_" + hashlib.md5(
            f"{fix.lat:.4f}_{fix.lon:.4f}_{target.freq_hz:.0f}".encode()
        ).hexdigest()[:10]
        target.feature_id = fid
        target.confirmed = True

        feature = _target_to_feature(target)
        try:
            from backend.storage.map_store import upsert_feature
            upsert_feature(feature)
        except Exception as exc:
            logger.warning("auto_loop: upsert_feature failed: %s", exc)

        with self._lock:
            self._confirmed_freqs.add(target.freq_hz)
            self._state.confirmed_targets.append(target)

        _emit("target_pinned", {
            "feature_id": fid,
            "freq_hz": target.freq_hz,
            "lat": fix.lat, "lon": fix.lon,
            "confidence": fix.confidence,
            "feature": feature,
        })

    def _advance_to_next(self) -> None:
        """Pick next unconfirmed peak from last scan and start tracking it."""
        with self._lock:
            results = list(self._state.scan_results)
            confirmed = set(self._confirmed_freqs)

        novel = [
            p for p in results
            if not any(abs(p["freq_hz"] - cf) < 10_000 for cf in confirmed)
        ]
        if novel:
            best = max(novel, key=lambda x: x["snr_db"])
            self._start_target(best)
        else:
            with self._lock:
                self._state.state = "SCANNING"
            _emit("fox_state", {"state": "SCANNING", "message": "Moving to next scan cycle"})
            time.sleep(self.scan_interval_s)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _freq_to_band(freq_hz: float) -> str:
    f = freq_hz / 1e6
    if f < 0.03:   return "ELF/SLF"
    if f < 0.3:    return "VLF"
    if f < 3:      return "LF/MF"
    if f < 30:     return "HF"
    if f < 88:     return "VHF Low"
    if f < 108:    return "FM Broadcast"
    if f < 300:    return "VHF"
    if f < 1000:   return "UHF"
    if f < 6000:   return "SHF/Cellular"
    return "EHF"


def _fix_to_dict(fix: FusedFix) -> dict:
    return {
        "lat": fix.lat, "lon": fix.lon,
        "uncertainty_m": fix.uncertainty_m,
        "confidence": fix.confidence,
        "methods": fix.methods_used,
        "ellipse_major_m": fix.ellipse_major_m,
        "ellipse_minor_m": fix.ellipse_minor_m,
        "ellipse_angle_deg": fix.ellipse_angle_deg,
        "per_method": [
            {"method": e.method, "lat": e.lat, "lon": e.lon,
             "uncertainty_m": e.uncertainty_m, "confidence": e.confidence,
             "notes": e.notes}
            for e in fix.per_method
        ],
    }


def _target_to_feature(target: FoxTarget) -> dict:
    fix = target.fix
    if fix is None:
        return {}
    ellipse_coords = ellipse_polygon(
        fix.lat, fix.lon,
        fix.ellipse_major_m or max(fix.uncertainty_m, 50),
        fix.ellipse_minor_m or max(fix.uncertainty_m * 0.6, 30),
        fix.ellipse_angle_deg,
    )
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [fix.lon, fix.lat]},
        "properties": {
            "id": target.feature_id or f"fox_{target.freq_hz:.0f}",
            "kind": "foxhunt",
            "name": f"Fox: {target.freq_hz/1e6:.3f} MHz ({target.band_label})",
            "freq_hz": target.freq_hz,
            "freq_mhz": round(target.freq_hz / 1e6, 4),
            "band_label": target.band_label,
            "modulation_hint": target.modulation_hint,
            "confidence": fix.confidence,
            "certainty_level": _certainty_level(fix.confidence),
            "observability_class": _observability_class(target, fix),
            "uncertainty_m": fix.uncertainty_m,
            "methods": fix.methods_used,
            "rssi_obs_count": len(target.rssi_obs),
            "bearing_obs_count": len(target.bearing_obs),
            "timestamp": target.first_seen,
            "ellipse_polygon": ellipse_coords,
            "ellipse_major_m": fix.ellipse_major_m,
            "ellipse_minor_m": fix.ellipse_minor_m,
            "beamwidth_deg": 360,
            "ray_length_m": max(200, int(fix.uncertainty_m * 0.3)),
            "wedge_radius_m": max(300, int(fix.uncertainty_m * 0.5)),
            "per_method": [
                {"method": e.method, "lat": e.lat, "lon": e.lon,
                 "uncertainty_m": e.uncertainty_m}
                for e in fix.per_method
            ],
        },
    }


def _pin_confidence_threshold(target: FoxTarget) -> float:
    """Pin threshold policy with stricter gate for single-directional-source solves."""
    if len(target.bearing_obs) <= 1:
        return _SINGLE_DIRECTIONAL_PIN_CONFIDENCE_THRESHOLD
    return _BASE_PIN_CONFIDENCE_THRESHOLD


def _observability_class(target: FoxTarget, fix: FusedFix) -> str:
    """Return a simple observability classification label for UI metadata."""
    if len(target.bearing_obs) <= 1:
        return "single_directional_source"
    if len(target.bearing_obs) >= 3 and fix.uncertainty_m <= 900:
        return "well_observed"
    return "multi_source_limited_geometry"


def _certainty_level(confidence: float) -> str:
    """Map confidence score to certainty label."""
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"


# ── Module-level singleton ────────────────────────────────────────────────────
auto_loop = AutoFoxHuntLoop()
