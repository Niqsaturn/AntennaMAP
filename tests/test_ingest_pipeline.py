from __future__ import annotations

from pathlib import Path

from backend.ingest.infrastructure import ingest_infrastructure
from backend.ingest.telemetry import ingest_telemetry


def test_invalid_rows_and_schema_rejection(tmp_path: Path):
    out = tmp_path / "telemetry.jsonl"
    rows = [
        {"timestamp": "2026-04-01T00:00:00Z", "band": "A", "rssi_dbm": -60, "snr_db": 10, "bearing_deg": 10, "lat": 10, "lon": 20},
        {"timestamp": "bad-ts", "band": "A", "rssi_dbm": -60, "snr_db": 10, "bearing_deg": 10, "lat": 10, "lon": 20},
        {"timestamp": "2026-04-01T00:00:00Z", "band": "A", "rssi_dbm": -60, "snr_db": 10, "bearing_deg": 10, "lat": 100, "lon": 20},
    ]
    result = ingest_telemetry(rows, out)
    assert len(result.accepted) == 1
    assert len(result.errors) == 2
    assert all(err["type"] == "schema_validation" for err in result.errors)


def test_duplicate_suppression(tmp_path: Path):
    out = tmp_path / "telemetry.jsonl"
    row = {"timestamp": "2026-04-01T00:00:00Z", "band": "A", "rssi_dbm": -60, "snr_db": 10, "bearing_deg": 10, "lat": 10, "lon": 20}
    result = ingest_telemetry([row, dict(row)], out)
    assert len(result.accepted) == 1
    assert len(result.warnings) == 1
    assert result.warnings[0]["type"] == "duplicate_suppressed"


def test_infrastructure_schema_and_coordinate_bounds(tmp_path: Path):
    out = tmp_path / "infra.jsonl"
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-80.2, 25.9]},
            "properties": {"id": "s1", "kind": "infrastructure", "name": "site", "timestamp": "2026-04-01T00:00:00Z"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-200.0, 25.9]},
            "properties": {"id": "s2", "kind": "infrastructure", "name": "site", "timestamp": "2026-04-01T00:00:00Z"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-80.2, 25.9]},
            "properties": {"id": "", "kind": "infrastructure", "name": "site", "timestamp": "2026-04-01T00:00:00Z"},
        },
    ]
    result = ingest_infrastructure(features, out)
    assert len(result.accepted) == 1
    assert len(result.errors) == 2
    assert {e["type"] for e in result.errors} == {"coordinate_out_of_bounds", "schema_validation"}
