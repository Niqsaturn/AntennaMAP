from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.ml.triangulation_pipeline import (
    TRAINING_DIR,
    load_latest_model,
    parse_samples,
    predict_error,
    solve_weighted_least_squares,
    train_error_model,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "public" / "data" / "antenna_data.geojson"
SEED_FILE = ROOT / "public" / "data" / "telemetry_samples.json"

app = FastAPI(title="AntennaMAP API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load_geojson() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def summarize_telemetry(samples: list[dict]) -> dict:
    band_summary: dict[str, dict] = {}
    for sample in samples:
        band = sample.get("band", "unknown")
        band_summary.setdefault(band, {"sample_count": 0, "snr": 0.0, "rssi": 0.0})
        band_summary[band]["sample_count"] += 1
        band_summary[band]["snr"] += sample.get("snr_db", 0.0)
        band_summary[band]["rssi"] += sample.get("rssi_dbm", 0.0)
    for _, v in band_summary.items():
        n = v["sample_count"]
        v["avg_snr_db"] = round(v.pop("snr") / n, 3)
        v["avg_rssi_dbm"] = round(v.pop("rssi") / n, 3)
    return {"band_summary": band_summary, "sample_count": len(samples)}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "antennamap"}


@app.get("/api/features")
def get_features(kind: str | None = Query(default=None, pattern="^(infrastructure|estimate)?$"), timestamp_lte: str | None = None) -> dict:
    data = load_geojson()
    features = data["features"]
    if kind:
        features = [f for f in features if f["properties"].get("kind") == kind]
    if timestamp_lte:
        cutoff = datetime.fromisoformat(timestamp_lte.replace("Z", "+00:00"))
        features = [f for f in features if datetime.fromisoformat(f["properties"]["timestamp"].replace("Z", "+00:00")) <= cutoff]
    return {"type": "FeatureCollection", "features": features}


@app.post("/api/model/train")
def model_train(payload: dict) -> dict:
    samples = parse_samples(payload.get("samples", []))
    targets = payload.get("target_error_m", [])
    version = payload.get("model_version", datetime.utcnow().strftime("v%Y%m%d%H%M%S"))
    result = train_error_model(samples, targets, version)
    training_record = {"model_version": version, "trained_at": result.metadata.trained_at, "sample_count": len(samples)}
    (TRAINING_DIR / f"training_run_{version}.json").write_text(json.dumps(training_record, indent=2), encoding="utf-8")
    return {"status": "trained", "model_version": version, "rmse": result.rmse, "metadata": result.metadata.__dict__}


@app.get("/api/model/status")
def model_status() -> dict:
    model = load_latest_model()
    if not model:
        return {"status": "not_trained"}
    return {"status": "ready", "model_version": model["metadata"]["model_version"], "trained_at": model["metadata"]["trained_at"], "feature_schema": model["metadata"]["feature_schema"]}


@app.post("/api/model/infer")
def model_infer(payload: dict) -> dict:
    samples = parse_samples(payload.get("samples", []))
    solution = solve_weighted_least_squares(samples)
    model = load_latest_model()
    if model:
        solution.predicted_error_m = predict_error(model, samples)
    return solution.__dict__


@app.get("/api/model/schema")
def model_schema() -> dict:
    return json.loads(SEED_FILE.read_text(encoding="utf-8"))


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
