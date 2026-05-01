from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from statistics import mean
from typing import Any

from backend.pipeline.compliance import COMPLIANCE_POLICY

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class TelemetrySample(BaseModel):
    """Validated append-only sample (no payload decoding or PII fields)."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    band: str = Field(min_length=1)
    rssi_dbm: float
    snr_db: float
    bearing_deg: float
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    region: str = Field(default="US", min_length=2, max_length=3)
    frequency_mhz: float = Field(default=2450.0, gt=0)
    frequency_hz: float | None = Field(default=None, gt=0)
    operator_id: str = Field(default="default", min_length=1)




def _is_frequency_allowed(region: str, frequency_mhz: float) -> bool:
    ranges = COMPLIANCE_POLICY["frequency_allowlist_mhz"].get(region.upper(), [])
    return any(low <= frequency_mhz <= high for low, high in ranges)


def _contains_payload_fields(sample: dict[str, Any]) -> bool:
    blocked = {"payload", "payload_hex", "payload_bytes", "decoded_payload"}
    return any(field in sample for field in blocked)

@dataclass
class IngestionResult:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    quality_metrics: dict[str, Any]


def _haversine_meters(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    radius = 6_371_000
    dlat = radians(b_lat - a_lat)
    dlon = radians(b_lon - a_lon)
    lat1 = radians(a_lat)
    lat2 = radians(b_lat)
    x = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    y = 2 * atan2(sqrt(x), sqrt(1 - x))
    return radius * y


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str))
            f.write("\n")


def ingest_telemetry(samples: list[dict[str, Any]], append_path: Path) -> IngestionResult:
    """Validate and append clean telemetry samples.

    Data-quality rules:
    - monotonic timestamps (non-decreasing)
    - GPS outlier rejection (speed > 120 m/s)
    - bearing in [0, 360]
    - signal ranges: rssi_dbm in [-140, -20], snr_db in [-20, 60]
    """

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    parsed_rows: list[TelemetrySample] = []

    for raw in samples:
        if _contains_payload_fields(raw):
            rejected.append({"reason": "payload_decode_forbidden", "sample": raw})
            continue
        try:
            parsed_rows.append(TelemetrySample.model_validate(raw))
        except ValidationError as exc:
            rejected.append({"reason": "schema_validation", "errors": exc.errors(), "sample": raw})

    parsed_rows.sort(key=lambda row: row.timestamp)

    prev: TelemetrySample | None = None
    for row in parsed_rows:
        reasons: list[str] = []

        if not (0 <= row.bearing_deg <= 360):
            reasons.append("bearing_out_of_range")

        if not (-140 <= row.rssi_dbm <= -20):
            reasons.append("rssi_out_of_range")

        if not (-20 <= row.snr_db <= 60):
            reasons.append("snr_out_of_range")

        if not _is_frequency_allowed(row.region, row.frequency_mhz):
            reasons.append("frequency_not_allowlisted")

        if prev is not None:
            if row.timestamp < prev.timestamp:
                reasons.append("timestamp_non_monotonic")
            dt_s = (row.timestamp - prev.timestamp).total_seconds()
            if dt_s > 0:
                distance_m = _haversine_meters(prev.lat, prev.lon, row.lat, row.lon)
                speed_m_s = distance_m / dt_s
                if speed_m_s > 120:
                    reasons.append("gps_outlier")

        as_dict = row.model_dump(mode="json")
        if reasons:
            rejected.append({"reason": reasons, "sample": as_dict})
        else:
            accepted.append(as_dict)
            prev = row

    _append_jsonl(append_path, accepted)

    quality_metrics = {
        "input_samples": len(samples),
        "accepted_samples": len(accepted),
        "rejected_samples": len(rejected),
        "acceptance_rate": round((len(accepted) / len(samples)) if samples else 0.0, 4),
    }
    return IngestionResult(accepted=accepted, rejected=rejected, quality_metrics=quality_metrics)


def summarize_telemetry(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in samples:
        grouped.setdefault(row["band"], []).append(row)

    band_summary: dict[str, dict[str, Any]] = {}
    for band, rows in grouped.items():
        band_summary[band] = {
            "sample_count": len(rows),
            "avg_snr_db": round(mean(r["snr_db"] for r in rows), 3),
            "avg_rssi_dbm": round(mean(r["rssi_dbm"] for r in rows), 3),
        }

    return {"total_samples": len(samples), "band_summary": band_summary}


def evaluate_retraining_triggers(
    run_metadata: list[dict[str, Any]],
    *,
    min_new_samples: int = 1000,
    drift_threshold: float = 0.2,
    schedule_days: int = 14,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(tz=timezone.utc)
    total_new_samples = sum((entry.get("metrics") or {}).get("accepted_samples", 0) for entry in run_metadata)
    latest_drift = max(((entry.get("metrics") or {}).get("drift_error", 0.0) for entry in run_metadata), default=0.0)

    most_recent_run = None
    if run_metadata:
        most_recent_run = max(datetime.fromisoformat(x["run_id"]) for x in run_metadata)
    due_by_schedule = most_recent_run is None or (now - most_recent_run) >= timedelta(days=schedule_days)

    reasons = []
    if total_new_samples >= min_new_samples:
        reasons.append("sample_volume")
    if latest_drift >= drift_threshold:
        reasons.append("drift_threshold")
    if due_by_schedule:
        reasons.append("scheduled_window")

    return {
        "should_retrain": bool(reasons),
        "reasons": reasons,
        "thresholds": {
            "min_new_samples": min_new_samples,
            "drift_threshold": drift_threshold,
            "schedule_days": schedule_days,
        },
        "observed": {
            "total_new_samples": total_new_samples,
            "latest_drift_error": latest_drift,
            "due_by_schedule": due_by_schedule,
        },
    }
