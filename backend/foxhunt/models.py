from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class FoxHuntObservation(BaseModel):
    timestamp: datetime
    lat: float
    lon: float
    heading_deg: float = Field(ge=0, lt=360)
    rssi_dbm: float
    snr_db: float
    frequency_hz: float
    bandwidth_hz: float
    antenna_bearing_deg: float | None = Field(default=None, ge=0, lt=360)


class SolverEstimate(BaseModel):
    center_lat: float
    center_lon: float
    confidence_score: float = Field(ge=0, le=1)
    uncertainty_major_m: float
    uncertainty_minor_m: float
    uncertainty_heading_deg: float


class FoxHuntSession(BaseModel):
    id: str
    started_at: datetime
    stopped_at: datetime | None = None
    status: str = "active"
    observations: list[FoxHuntObservation] = Field(default_factory=list)
    estimate: SolverEstimate | None = None
    map_features: dict | None = None
