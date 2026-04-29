from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from .models import FoxHuntObservation, FoxHuntSession
from .solver import solve
from .trainer import FoxHuntTrainer


class FoxHuntService:
    def __init__(self) -> None:
        self.sessions: dict[str, FoxHuntSession] = {}
        self.trainer = FoxHuntTrainer()

    def start_session(self) -> FoxHuntSession:
        session = FoxHuntSession(id=str(uuid4()), started_at=datetime.now(timezone.utc))
        self.sessions[session.id] = session
        return session

    def stop_session(self, session_id: str) -> FoxHuntSession:
        session = self.sessions[session_id]
        session.stopped_at = datetime.now(timezone.utc)
        session.status = "stopped"
        return session

    def append_observation(self, session_id: str, observation: FoxHuntObservation) -> FoxHuntSession:
        session = self.sessions[session_id]
        session.observations.append(observation)
        estimate = solve(session.observations)
        if estimate:
            estimate.confidence_score = self.trainer.calibrate_confidence(estimate.confidence_score)
            session.estimate = estimate
            session.map_features = self._to_feature_collection(session)
        return session

    def get_session(self, session_id: str) -> FoxHuntSession:
        return self.sessions[session_id]

    def _to_feature_collection(self, session: FoxHuntSession) -> dict:
        est = session.estimate
        assert est is not None
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [est.center_lon, est.center_lat]},
                    "properties": {
                        "kind": "foxhunt_estimate",
                        "session_id": session.id,
                        "confidence_score": est.confidence_score,
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Ellipse",
                        "center": [est.center_lon, est.center_lat],
                        "major_m": est.uncertainty_major_m,
                        "minor_m": est.uncertainty_minor_m,
                        "heading_deg": est.uncertainty_heading_deg,
                    },
                    "properties": {"kind": "foxhunt_uncertainty", "session_id": session.id},
                },
            ],
        }
