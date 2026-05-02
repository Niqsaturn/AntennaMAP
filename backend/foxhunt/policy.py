from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from backend.foxhunt.auto_loop import auto_loop


PolicyPhase = Literal["IDLE", "OBSERVE", "INFER", "PLACE_BEARING", "UPDATE_CONFIDENCE", "PAUSED"]


@dataclass
class PolicyDecision:
    phase: PolicyPhase
    reason: str
    confidence: float


class AutoHuntPolicy:
    def __init__(self) -> None:
        self.running = False
        self.phase: PolicyPhase = "IDLE"
        self.last_action_at: datetime | None = None
        self.max_actions_per_minute = 8
        self.last_decision: PolicyDecision | None = None
        self.stale_data_timeout_s = 45
        self.target_promote_threshold = 0.82

    def start(self) -> dict:
        self.running = True
        self.phase = "OBSERVE"
        return self.status()

    def stop(self) -> dict:
        self.running = False
        self.phase = "IDLE"
        return self.status()

    def status(self) -> dict:
        return {
            "running": self.running,
            "phase": self.phase,
            "max_actions_per_minute": self.max_actions_per_minute,
            "stale_data_timeout_s": self.stale_data_timeout_s,
            "target_promote_threshold": self.target_promote_threshold,
            "last_action_at": self.last_action_at.isoformat() if self.last_action_at else None,
            "last_decision": self.last_decision.__dict__ if self.last_decision else None,
        }

    def choose_action(self, loop_status: dict) -> PolicyDecision:
        target = loop_status.get("target") or {}
        fix = target.get("fix") or {}
        uncertainty = float(fix.get("uncertainty_m") or 1000.0)
        conf = float(fix.get("confidence") or 0.0)
        bearing_obs = int(target.get("bearing_obs_count") or 0)
        rssi_obs = int(target.get("rssi_obs_count") or 0)
        recency = self._seconds_since(self.last_action_at)
        geometry_quality = self._geometry_quality(bearing_obs, uncertainty)
        score = 0.5 * (1.0 - min(1.0, uncertainty / 3000.0)) + 0.3 * min(1.0, recency / 30.0) + 0.2 * geometry_quality

        if conf >= self.target_promote_threshold:
            return PolicyDecision("UPDATE_CONFIDENCE", "target confidence reached promotion threshold", conf)
        if uncertainty > 900 or geometry_quality < 0.5:
            return PolicyDecision("PLACE_BEARING", "high uncertainty or weak geometry", conf)
        if rssi_obs < 2:
            return PolicyDecision("OBSERVE", "need more rssi observations", conf)
        if score > 0.45:
            return PolicyDecision("INFER", "sufficient data for solve attempt", conf)
        return PolicyDecision("OBSERVE", "waiting for better recency/quality", conf)

    def execute_cycle(self) -> dict:
        if not self.running:
            return {"ok": False, "reason": "policy not running", **self.status()}
        loop_status = auto_loop.status()
        decision = self.choose_action(loop_status)
        self.last_decision = decision
        self.phase = decision.phase
        now = datetime.now(timezone.utc)
        if self._rate_limited(now):
            self.phase = "PAUSED"
            return {"ok": False, "reason": "rate_limited", "decision": decision.__dict__, **self.status()}

        if decision.phase == "PLACE_BEARING":
            self._place_ai_bearing(loop_status)
        self.last_action_at = now
        promoted = decision.phase == "UPDATE_CONFIDENCE" and decision.confidence >= self.target_promote_threshold
        return {"ok": True, "decision": decision.__dict__, "promoted_target": promoted, **self.status()}

    def _place_ai_bearing(self, loop_status: dict) -> None:
        target = loop_status.get("target") or {}
        freq = target.get("freq_hz")
        if not freq:
            return
        base = 15.0 + (len(str(freq)) % 45)
        bearing = math.fmod(base + 360.0, 360.0)
        auto_loop.add_bearing_observation(bearing_deg=bearing, snr_db=12.0, freq_hz=float(freq), source="ai_policy")

    def _rate_limited(self, now: datetime) -> bool:
        if not self.last_action_at:
            return False
        elapsed = (now - self.last_action_at).total_seconds()
        return elapsed < max(1.0, 60.0 / max(1, self.max_actions_per_minute))

    @staticmethod
    def _seconds_since(ts: datetime | None) -> float:
        if not ts:
            return 9999.0
        return (datetime.now(timezone.utc) - ts).total_seconds()

    @staticmethod
    def _geometry_quality(bearing_obs: int, uncertainty_m: float) -> float:
        coverage = min(1.0, bearing_obs / 3.0)
        precision = max(0.0, 1.0 - min(1.0, uncertainty_m / 2500.0))
        return round(0.6 * coverage + 0.4 * precision, 3)


auto_policy = AutoHuntPolicy()
