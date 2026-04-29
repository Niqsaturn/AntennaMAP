from backend.main import app, summarize_telemetry
from backend.ml.triangulation_pipeline import TelemetrySample, movement_geometry_quality, solve_weighted_least_squares, windowed_deltas
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


def test_feature_extraction_and_quality():
    samples = [_sample(32.0, -117.0, 0, 45), _sample(32.0004, -117.0004, 15, 30), _sample(32.0008, -117.0008, 35, 20)]
    deltas = windowed_deltas(samples)
    assert deltas[-1]["delta_heading"] == 35
    assert movement_geometry_quality(samples) > 0


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


def test_model_status_and_infer_schema():
    client = TestClient(app)
    status = client.get("/api/model/status")
    assert status.status_code == 200
    payload = {
        "samples": [
            {
                "timestamp": "2026-04-01T00:00:00Z",
                "lat": 32.0,
                "lon": -117.0,
                "heading_deg": 0,
                "bearing_estimate_deg": 45,
                "rssi_dbm": -60,
                "snr_db": 12,
                "frequency_hz": 2.4e9,
                "bandwidth_hz": 2e6,
            },
            {
                "timestamp": "2026-04-01T00:00:10Z",
                "lat": 32.0005,
                "lon": -117.0005,
                "heading_deg": 15,
                "bearing_estimate_deg": 315,
                "rssi_dbm": -66,
                "snr_db": 11,
                "frequency_hz": 2.4e9,
                "bandwidth_hz": 2e6,
            },
        ]
    }
    infer = client.post("/api/model/infer", json=payload)
    assert infer.status_code == 200
    body = infer.json()
    assert "center_lat" in body and "covariance" in body
