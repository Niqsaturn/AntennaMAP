from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def _obs(lat: float, lon: float, heading: float, snr: float = 15.0, rssi: float = -65.0, antenna_bearing: float | None = None):
    return {
        "timestamp": "2026-04-29T00:00:00Z",
        "lat": lat,
        "lon": lon,
        "heading_deg": heading,
        "rssi_dbm": rssi,
        "snr_db": snr,
        "frequency_hz": 2.4e9,
        "bandwidth_hz": 2e6,
        "antenna_bearing_deg": antenna_bearing,
    }


def test_session_lifecycle_start_get_stop():
    start = client.post("/api/foxhunt/session/start")
    assert start.status_code == 200
    session_id = start.json()["id"]

    fetch = client.get(f"/api/foxhunt/session/{session_id}")
    assert fetch.status_code == 200
    assert fetch.json()["status"] == "active"

    stop = client.post("/api/foxhunt/session/stop", params={"session_id": session_id})
    assert stop.status_code == 200
    assert stop.json()["status"] == "stopped"


def test_solver_output_schema_and_map_features():
    session_id = client.post("/api/foxhunt/session/start").json()["id"]

    client.post(f"/api/foxhunt/session/{session_id}/observation", json=_obs(32.0, -117.0, 35, antenna_bearing=45))
    client.post(f"/api/foxhunt/session/{session_id}/observation", json=_obs(32.0004, -117.0005, 310, antenna_bearing=320))
    last = client.post(f"/api/foxhunt/session/{session_id}/observation", json=_obs(31.9999, -116.9993, 220, antenna_bearing=225))

    payload = last.json()
    estimate = payload["estimate"]
    assert {"center_lat", "center_lon", "confidence_score", "uncertainty_major_m", "uncertainty_minor_m", "uncertainty_heading_deg"}.issubset(estimate.keys())
    assert payload["map_features"]["type"] == "FeatureCollection"
    assert len(payload["map_features"]["features"]) == 2


def test_confidence_bounds():
    session_id = client.post("/api/foxhunt/session/start").json()["id"]

    client.post(f"/api/foxhunt/session/{session_id}/observation", json=_obs(32.0, -117.0, 0, snr=5.0, rssi=-80.0))
    client.post(f"/api/foxhunt/session/{session_id}/observation", json=_obs(32.0007, -117.0008, 130, snr=20.0, rssi=-60.0))
    response = client.post(f"/api/foxhunt/session/{session_id}/observation", json=_obs(32.0011, -117.0001, 250, snr=12.0, rssi=-67.0))

    score = response.json()["estimate"]["confidence_score"]
    assert 0.0 <= score <= 1.0
