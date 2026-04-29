import pytest
from backend.main import app, summarize_telemetry
from backend.rf_propagation import path_loss_db
from fastapi.testclient import TestClient


def _sample(lat, lon, heading, bearing, snr=12.0, rssi=-65.0):
    return TelemetrySample(
        timestamp="2026-04-01T00:00:00Z",
        lat=lat,
        lon=lon,
        heading_deg=heading,
        bearing_estimate_deg=bearing,
        rssi_dbm=rssi,
        snr_db=snr,
        frequency_hz=2.4e9,
        bandwidth_hz=2e6,
        terrain={"elevation_m": 32.0},
    )


def test_summarize_telemetry_groups_by_band():
    summary = summarize_telemetry([
        {"band": "A", "snr_db": 10, "rssi_dbm": -60},
        {"band": "A", "snr_db": 20, "rssi_dbm": -70},
    ])
    assert summary["band_summary"]["A"]["sample_count"] == 2
    assert summary["band_summary"]["A"]["avg_snr_db"] == 15.0


@pytest.mark.xfail(reason="windowed_deltas / movement_geometry_quality not yet implemented")
def test_feature_extraction_and_quality():
    samples = [_sample(32.0, -117.0, 0, 45), _sample(32.0004, -117.0004, 15, 30), _sample(32.0008, -117.0008, 35, 20)]
    deltas = windowed_deltas(samples)
    assert deltas[-1]["delta_heading"] == 35
    assert movement_geometry_quality(samples) > 0


@pytest.mark.xfail(reason="solve_weighted_least_squares not yet implemented")
def test_solver_convergence_and_schema():
    samples = [_sample(32.0, -117.0, 0, 45), _sample(32.0005, -117.0005, 10, 315), _sample(32.0002, -116.9996, 20, 225)]
    result = solve_weighted_least_squares(samples)
    assert 31.99 < result.center_lat < 32.01
    assert -117.01 < result.center_lon < -116.99
    assert len(result.covariance) == 2 and len(result.covariance[0]) == 2


def test_api_health_and_features_shape():
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
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

def test_loop_pause_resume_config_lifecycle():
    client = TestClient(app)

    pause_resp = client.post('/api/loop/pause')
    assert pause_resp.status_code == 200
    assert pause_resp.json()['active'] is False

    config_resp = client.post('/api/loop/config', params={'interval_seconds': 5, 'provider': 'mock', 'model': 'v2'})
    assert config_resp.status_code == 200
    cfg = config_resp.json()['config']
    assert cfg['interval_seconds'] == 5
    assert cfg['provider'] == 'mock'
    assert cfg['model'] == 'v2'

    resume_resp = client.post('/api/loop/resume')
    assert resume_resp.status_code == 200
    assert resume_resp.json()['active'] is True

    status_resp = client.get('/api/loop/status')
    assert status_resp.status_code == 200
    status = status_resp.json()
    assert status['active'] is True
    assert status['config']['provider'] == 'mock'

    client.post('/api/loop/pause')
