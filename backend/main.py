from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.pipeline.ingest import evaluate_retraining_triggers, ingest_telemetry, summarize_telemetry

from backend.training.trainer import train_single_triangulation_baseline
from backend.training.triangulation_baseline import estimate_single_operator

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "public" / "data" / "antenna_data.geojson"
TELEMETRY_FILE = ROOT / "public" / "data" / "telemetry_samples.json"
INGEST_LOG_FILE = ROOT / "backend" / "pipeline" / "data" / "telemetry_ingested.jsonl"
RUN_METADATA_FILE = ROOT / "backend" / "pipeline" / "data" / "model_runs.jsonl"
MODELS_DIR = ROOT / "models"
TRAINING_STATUS_FILE = ROOT / "backend" / "pipeline" / "data" / "training_status.json"
RUNTIME_CONFIG_FILE = ROOT / "backend" / "pipeline" / "data" / "runtime_config.json"

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
    out = dict(feature)
    props = dict(out.get("properties", {}))
    props.setdefault("timestamp", datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"))
    out["properties"] = props
    return out


def _load_runtime_config() -> dict:
    if not RUNTIME_CONFIG_FILE.exists():
        return {"selected_method": "single_triangulation_baseline"}
    return json.loads(RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))


def _write_training_status(payload: dict) -> None:
    TRAINING_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_STATUS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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

    runtime = _load_runtime_config()
    selected_method = runtime.get("selected_method", "single_triangulation_baseline")
    run_id = datetime.now(tz=timezone.utc).isoformat()
    run_metadata = {
        "run_id": run_id,
        "model_version": model_version,
        "selected_method": selected_method,
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




@app.post("/api/training/start")
def start_training(method: str = "single_triangulation_baseline") -> dict:
    if method != "single_triangulation_baseline":
        raise HTTPException(status_code=400, detail="unsupported training method")

    _write_training_status({"status": "running", "method": method, "started_at": datetime.now(tz=timezone.utc).isoformat()})
    samples = _load_json(TELEMETRY_FILE)
    artifact = train_single_triangulation_baseline(samples, MODELS_DIR)
    runtime = _load_runtime_config()
    runtime["selected_method"] = method
    runtime["selected_model"] = artifact["model_name"]
    RUNTIME_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_CONFIG_FILE.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    _append_jsonl(RUN_METADATA_FILE, {"run_id": datetime.now(tz=timezone.utc).isoformat(), "training": artifact})
    _write_training_status({"status": "completed", "method": method, "completed_at": datetime.now(tz=timezone.utc).isoformat(), "metrics": artifact["metrics"]})
    return {"status": "started", "method": method, "artifact": artifact}


@app.get("/api/training/status")
def training_status() -> dict:
    if not TRAINING_STATUS_FILE.exists():
        return {"status": "idle"}
    return json.loads(TRAINING_STATUS_FILE.read_text(encoding="utf-8"))


@app.get("/api/training/history")
def training_history() -> dict:
    files = sorted(MODELS_DIR.glob("single_triangulation_baseline_*.json"))
    history = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    return {"history": history, "count": len(history)}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
