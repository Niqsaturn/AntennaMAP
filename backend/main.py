from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request
from fastapi import FastAPI, Query
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.model_providers import discover_python_local_models
from backend.pipeline.ingest import evaluate_retraining_triggers, ingest_telemetry, summarize_telemetry
from backend.rf.geometry import build_propagation_features
from backend.rf_propagation import generate_snapshot

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "public" / "data" / "antenna_data.geojson"
TELEMETRY_FILE = ROOT / "public" / "data" / "telemetry_samples.json"
INGEST_LOG_FILE = ROOT / "backend" / "pipeline" / "data" / "telemetry_ingested.jsonl"
RUN_METADATA_FILE = ROOT / "backend" / "pipeline" / "data" / "model_runs.jsonl"
MODEL_STATE = {"provider": "python_local", "model": None}

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


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True))
        f.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_telemetry_samples() -> list[dict]:
    return _load_json(TELEMETRY_FILE)


def enrich_feature(feature: dict, telemetry: list[dict]) -> dict:
    _ = telemetry
    return feature


def discover_ollama_models() -> list[str]:
    req = request.Request("http://127.0.0.1:11434/api/tags", method="GET")
    try:
        with request.urlopen(req, timeout=2.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (TimeoutError, error.URLError, json.JSONDecodeError):
        return []
    models = payload.get("models", [])
    return sorted(
        {
            item.get("name")
            for item in models
            if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name")
        }
    )


def discover_available_models() -> dict[str, list[str]]:
    return {"ollama": discover_ollama_models(), "python_local": discover_python_local_models()}


def run_inference(provider: str, model: str, payload: dict) -> dict:
    if provider == "ollama":
        body = json.dumps({"model": model, **payload}).encode("utf-8")
        req = request.Request("http://127.0.0.1:11434/api/generate", data=body, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=30.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    if provider == "python_local":
        return {"provider": provider, "model": model, "output": "python-local inference adapter not yet implemented", "input": payload}
    raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "antennamap"}


@app.get("/api/features")
def get_features(kind: str | None = Query(default=None, pattern="^(infrastructure|estimate)?$"), timestamp_lte: str | None = None) -> dict:
    data = load_geojson()
    telemetry = load_telemetry_samples()
    features = [enrich_feature(f, telemetry) for f in data["features"]]

    if kind:
        features = [f for f in features if f["properties"].get("kind") == kind]
    if timestamp_lte:
        cutoff = datetime.fromisoformat(timestamp_lte.replace("Z", "+00:00"))
        features = [f for f in features if datetime.fromisoformat(f["properties"]["timestamp"].replace("Z", "+00:00")) <= cutoff]
    return {"type": "FeatureCollection", "features": features}


@app.post("/api/pipeline/ingest")
def ingest_pipeline(model_version: str = "baseline-v1") -> dict:
    raw_samples = _load_json(TELEMETRY_FILE)
    ingestion_result = ingest_telemetry(raw_samples, INGEST_LOG_FILE)
    summary = summarize_telemetry(ingestion_result.accepted)

    inference_outputs = [
        {
            "timestamp": s["timestamp"],
            "predicted_bearing_deg": s["bearing_deg"],
            "observed_bearing_deg": s["bearing_deg"],
            "abs_error_deg": 0.0,
        }
        for s in ingestion_result.accepted
    ]
    drift_error = 0.0

    timestamps = [datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00")) for s in ingestion_result.accepted]
    data_window = {
        "start": min(timestamps).isoformat() if timestamps else None,
        "end": max(timestamps).isoformat() if timestamps else None,
    }

    run_id = datetime.now(tz=timezone.utc).isoformat()
    run_metadata = {
        "run_id": run_id,
        "model_version": model_version,
        "data_window": data_window,
        "metrics": {
            **ingestion_result.quality_metrics,
            "drift_error": drift_error,
            "post_hoc_mae_deg": 0.0,
            "inference_outputs": inference_outputs,
            "summary": summary,
        },
    }
    _append_jsonl(RUN_METADATA_FILE, run_metadata)

    historical = _read_jsonl(RUN_METADATA_FILE)
    retraining = evaluate_retraining_triggers(historical)

    return {
        "ingestion": ingestion_result.quality_metrics,
        "run_metadata": run_metadata,
        "retraining": retraining,
    }


@app.get("/api/model/metrics")
def model_metrics() -> dict:
    runs = _read_jsonl(RUN_METADATA_FILE)
    latest = runs[-1] if runs else None
    retraining = evaluate_retraining_triggers(runs)
    return {"latest": latest, "runs": runs, "retraining": retraining}


@app.get("/api/models")
def get_models() -> dict:
    providers = discover_available_models()
    provider_payload = [{"id": name, "models": models} for name, models in providers.items()]
    active_model = MODEL_STATE["model"]
    if not active_model:
        for entry in provider_payload:
            if entry["models"]:
                MODEL_STATE["provider"] = entry["id"]
                MODEL_STATE["model"] = entry["models"][0]
                break
    return {"providers": provider_payload, "active": MODEL_STATE}


@app.post("/api/model/select")
def select_model(selection: dict) -> dict:
    provider = selection.get("provider")
    model = selection.get("model")
    if not provider or not model:
        raise HTTPException(status_code=400, detail="provider and model are required")
    providers = discover_available_models()
    if provider not in providers:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    if model not in providers[provider]:
        raise HTTPException(status_code=400, detail=f"Model '{model}' is not available for provider '{provider}'")
    MODEL_STATE["provider"] = provider
    MODEL_STATE["model"] = model
    return {"active": MODEL_STATE}


@app.get("/api/propagation")
def get_propagation(timestamp_lte: str | None = None, site_id: str | None = None, model: str = "fspl") -> dict:
    features = get_features(timestamp_lte=timestamp_lte)["features"]
    prop_features = []
    for feature in features:
        if feature.get("properties", {}).get("kind") != "infrastructure":
            continue
        if site_id and feature.get("properties", {}).get("id") != site_id:
            continue
        prop_features.extend(build_propagation_features(feature))
    selected = next((f for f in features if f.get("properties", {}).get("id") == site_id), None) if site_id else None
    snapshot = None
    if selected:
        lon, lat = selected["geometry"]["coordinates"]
        props = selected["properties"]
        snapshot = generate_snapshot(lat=lat, lon=lon, eirp_dbm=46.0, frequency_mhz=float(props.get("rf_max_mhz", 1900.0)), gain_dbi=3.0, height_m=30.0, beamwidth_deg=78.0, tilt_deg=5.0, orientation_deg=float(props.get("azimuth_deg") or 0.0), model=model, grid_radius_km=5.0, grid_resolution=41)
    return {"type": "FeatureCollection", "features": prop_features, "snapshot": snapshot}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
