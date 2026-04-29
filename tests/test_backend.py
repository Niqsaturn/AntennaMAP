from backend.main import app, summarize_telemetry
from backend.rf_propagation import path_loss_db
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


def test_reference_path_loss_ranges():
    fspl = path_loss_db("fspl", frequency_mhz=1900, distance_km=1.0)
    urban = path_loss_db("hata_urban", frequency_mhz=1900, distance_km=1.0, tx_height_m=30)
    suburban = path_loss_db("hata_suburban", frequency_mhz=1900, distance_km=1.0, tx_height_m=30)
    assert 95 <= fspl <= 100
    assert 120 <= urban <= 150
    assert suburban < urban


def test_propagation_endpoint_contains_transparency_metadata():
    client = TestClient(app)
    features = client.get("/api/features").json()["features"]
    site_id = next(f["properties"]["id"] for f in features if f["properties"]["kind"] == "infrastructure")
    resp = client.get(f"/api/propagation?site_id={site_id}&model=fspl")
    assert resp.status_code == 200
    payload = resp.json()
    assert "uncertainty" in payload["snapshot"]
    assert "assumptions" in payload["snapshot"]
    assert "contours" in payload["snapshot"]
