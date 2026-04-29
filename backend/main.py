from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.pipeline.ingest import evaluate_retraining_triggers, ingest_telemetry, summarize_telemetry

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "public" / "data" / "antenna_data.geojson"
TELEMETRY_FILE = ROOT / "public" / "data" / "telemetry_samples.json"
INGEST_LOG_FILE = ROOT / "backend" / "pipeline" / "data" / "telemetry_ingested.jsonl"
RUN_METADATA_FILE = ROOT / "backend" / "pipeline" / "data" / "model_runs.jsonl"

app = FastAPI(title="AntennaMAP API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_geojson() -> dict:
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


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


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "antennamap"}


@app.get("/api/features")
def get_features(
    kind: str | None = Query(default=None, pattern="^(infrastructure|estimate)?$"),
    timestamp_lte: str | None = None,
) -> dict:
    data = load_geojson()
    features = data["features"]

    if kind:
        features = [f for f in features if f["properties"].get("kind") == kind]

    if timestamp_lte:
        cutoff = datetime.fromisoformat(timestamp_lte.replace("Z", "+00:00"))
        features = [
            f
            for f in features
            if datetime.fromisoformat(f["properties"]["timestamp"].replace("Z", "+00:00")) <= cutoff
        ]

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


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
