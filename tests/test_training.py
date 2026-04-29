from fastapi.testclient import TestClient

from backend.main import app
from backend.training.triangulation_baseline import estimate_single_operator


def test_triangulation_baseline_output_schema():
    window = [
        {"lat": 32.0, "lon": -117.0, "rssi_dbm": -70, "snr_db": 15},
        {"lat": 32.0005, "lon": -117.0003, "rssi_dbm": -68, "snr_db": 18},
    ]
    out = estimate_single_operator(window)
    assert out["method"] == "single_triangulation_baseline"
    assert set(out.keys()) == {"method", "estimate", "num_samples", "confidence"}
    assert isinstance(out["estimate"]["lat"], float)
    assert isinstance(out["estimate"]["lon"], float)


def test_training_status_transitions():
    client = TestClient(app)
    assert client.get("/api/training/status").status_code == 200
    resp = client.post("/api/training/start", params={"method": "single_triangulation_baseline"})
    assert resp.status_code == 200
    status = client.get("/api/training/status").json()
    assert status["status"] == "completed"
    history = client.get("/api/training/history").json()
    assert history["count"] >= 1
