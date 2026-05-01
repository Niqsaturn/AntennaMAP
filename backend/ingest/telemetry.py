from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.ingest.storage import append_jsonl


class TelemetryRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str = Field(min_length=1, default="telemetry")
    timestamp: datetime
    band: str = Field(min_length=1)
    rssi_dbm: float
    snr_db: float
    bearing_deg: float
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    frequency_hz: float | None = None      # exact center frequency (enables FSPL inversion)
    operator_id: str = "default"           # multi-operator triangulation future use


@dataclass
class IngestResult:
    accepted: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def ingest_telemetry(rows: list[dict[str, Any]], output_path) -> IngestResult:
    accepted: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    seen_keys: set[tuple[str, str, float, float]] = set()
    now = datetime.now(tz=timezone.utc)

    for raw in rows:
        try:
            parsed = TelemetryRecord.model_validate(raw)
        except ValidationError as exc:
            errors.append({"type": "schema_validation", "errors": exc.errors(), "row": raw})
            continue

        normalized = parsed.model_dump(mode="json")
        timestamp = parsed.timestamp

        key = (parsed.band, timestamp.isoformat(), round(parsed.lat, 7), round(parsed.lon, 7))
        if key in seen_keys:
            warnings.append({"type": "duplicate_suppressed", "row": normalized})
            continue
        seen_keys.add(key)

        if timestamp < datetime(2000, 1, 1, tzinfo=timezone.utc):
            errors.append({"type": "timestamp_too_old", "row": normalized})
            continue
        if timestamp > now + timedelta(days=1):
            errors.append({"type": "timestamp_in_future", "row": normalized})
            continue

        accepted.append(normalized)

    append_jsonl(output_path, accepted)
    return IngestResult(accepted=accepted, warnings=warnings, errors=errors)
