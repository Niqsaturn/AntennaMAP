from backend.main import app, summarize_telemetry
from fastapi.testclient import TestClient


def test_summarize_telemetry_groups_by_band():
    summary = summarize_telemetry([
        {"band": "A", "snr_db": 10, "rssi_dbm": -60},
        {"band": "A", "snr_db": 20, "rssi_dbm": -70},
    ])
    assert summary["band_summary"]["A"]["sample_count"] == 2
    assert summary["band_summary"]["A"]["avg_snr_db"] == 15.0


def test_api_health_and_features_shape():
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"

    features = client.get("/api/features")
    assert features.status_code == 200
    payload = features.json()
    assert payload["type"] == "FeatureCollection"
    assert isinstance(payload["features"], list)
