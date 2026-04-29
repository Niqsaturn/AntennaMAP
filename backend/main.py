from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.ingest.infrastructure import ingest_infrastructure
from backend.ingest.storage import append_jsonl, read_jsonl
from backend.ingest.telemetry import ingest_telemetry
from backend.pipeline.ingest import evaluate_retraining_triggers, summarize_telemetry

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "public" / "data" / "antenna_data.geojson"
TELEMETRY_FILE = ROOT / "public" / "data" / "telemetry_samples.json"
INGEST_LOG_FILE = ROOT / "backend" / "pipeline" / "data" / "telemetry_ingested.jsonl"
INFRA_INGEST_FILE = ROOT / "backend" / "pipeline" / "data" / "infrastructure_ingested.jsonl"
RUN_METADATA_FILE = ROOT / "backend" / "pipeline" / "data" / "model_runs.jsonl"
INGEST_ISSUES_FILE = ROOT / "backend" / "pipeline" / "data" / "ingest_issues.jsonl"

app = FastAPI(title="AntennaMAP API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load_geojson() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def load_telemetry_samples() -> list[dict]:
    with TELEMETRY_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "antennamap"}


@app.get("/api/features")
def get_features(kind: str | None = Query(default=None, pattern="^(infrastructure|estimate)?$"), timestamp_lte: str | None = None) -> dict:
    data = load_geojson()
    features = data.get("features", [])

    if kind:
        features = [f for f in features if f.get("properties", {}).get("kind") == kind]
    if timestamp_lte:
        cutoff = datetime.fromisoformat(timestamp_lte.replace("Z", "+00:00"))
        features = [f for f in features if datetime.fromisoformat(f["properties"]["timestamp"].replace("Z", "+00:00")) <= cutoff]
    return {"type": "FeatureCollection", "features": features}


@app.post("/api/pipeline/ingest")
def ingest_pipeline(model_version: str = "baseline-v1") -> dict:
    raw_samples = load_telemetry_samples()
    telemetry = ingest_telemetry(raw_samples, INGEST_LOG_FILE)

    raw_features = load_geojson().get("features", [])
    infrastructure = ingest_infrastructure(raw_features, INFRA_INGEST_FILE)

    summary = summarize_telemetry(telemetry.accepted)
    issues = telemetry.errors + telemetry.warnings + infrastructure.errors
    append_jsonl(INGEST_ISSUES_FILE, [{"run_id": datetime.now(tz=timezone.utc).isoformat(), "issues": issues}])

    run_id = datetime.now(tz=timezone.utc).isoformat()
    run_metadata = {
        "run_id": run_id,
        "model_version": model_version,
        "metrics": {
            "accepted_samples": len(telemetry.accepted),
            "rejected_samples": len(telemetry.errors),
            "warning_samples": len(telemetry.warnings),
            "accepted_features": len(infrastructure.accepted),
            "rejected_features": len(infrastructure.errors),
            "summary": summary,
            "drift_error": 0.0,
        },
    }
    append_jsonl(RUN_METADATA_FILE, [run_metadata])
    historical = read_jsonl(RUN_METADATA_FILE)

    return {
        "run_metadata": run_metadata,
        "retraining": evaluate_retraining_triggers(historical),
        "issues": {"errors": telemetry.errors + infrastructure.errors, "warnings": telemetry.warnings},
    }


@app.get("/api/pipeline/ingest/issues")
def ingest_issues(limit: int = 25) -> dict:
    rows = read_jsonl(INGEST_ISSUES_FILE)
    return {"count": len(rows), "rows": rows[-limit:]}


@app.get("/api/model/metrics")
def model_metrics() -> dict:
    runs = read_jsonl(RUN_METADATA_FILE)
    latest = runs[-1] if runs else None
    retraining = evaluate_retraining_triggers(runs)
    return {"latest": latest, "runs": runs, "retraining": retraining}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
