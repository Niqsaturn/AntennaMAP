from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.pipeline.compliance import COMPLIANCE_POLICY, append_audit_event, apply_retention, policy_status
from backend.pipeline.ingest import evaluate_retraining_triggers, ingest_telemetry, summarize_telemetry

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "public" / "data" / "antenna_data.geojson"
TELEMETRY_FILE = ROOT / "public" / "data" / "telemetry_samples.json"
INGEST_LOG_FILE = ROOT / "backend" / "pipeline" / "data" / "telemetry_ingested.jsonl"
RUN_METADATA_FILE = ROOT / "backend" / "pipeline" / "data" / "model_runs.jsonl"
AUDIT_LOG_FILE = ROOT / "backend" / "pipeline" / "data" / "audit_events.jsonl"

app = FastAPI(title="AntennaMAP API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load_geojson() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


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
    return {"status": "ok", "service": "antennamap", **policy_status()}


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


@app.post("/api/pipeline/ingest")
def ingest_pipeline(model_version: str = "baseline-v1") -> dict:
    append_audit_event(AUDIT_LOG_FILE, "ingestion_start", {"model_version": model_version})
    raw_samples = _load_json(TELEMETRY_FILE)
    ingestion_result = ingest_telemetry(raw_samples, INGEST_LOG_FILE)
    summary = summarize_telemetry(ingestion_result.accepted)

    run_id = datetime.now(tz=timezone.utc).isoformat()
    run_metadata = {
        "run_id": run_id,
        "model_version": model_version,
        "metrics": {**ingestion_result.quality_metrics, "summary": summary, "drift_error": 0.0},
    }
    _append_jsonl(RUN_METADATA_FILE, run_metadata)

    apply_retention(INGEST_LOG_FILE, COMPLIANCE_POLICY["retention_days"]["raw_telemetry"])
    apply_retention(RUN_METADATA_FILE, COMPLIANCE_POLICY["retention_days"]["aggregated_metrics"])

    historical = _read_jsonl(RUN_METADATA_FILE)
    retraining = evaluate_retraining_triggers(historical)
    append_audit_event(
        AUDIT_LOG_FILE,
        "ingestion_stop",
        {"model_version": model_version, "accepted_samples": ingestion_result.quality_metrics["accepted_samples"]},
    )

    return {"ingestion": ingestion_result.quality_metrics, "run_metadata": run_metadata, "retraining": retraining}


@app.get("/api/model/metrics")
def model_metrics() -> dict:
    append_audit_event(AUDIT_LOG_FILE, "data_export", {"endpoint": "/api/model/metrics"})
    runs = _read_jsonl(RUN_METADATA_FILE)
    latest = runs[-1] if runs else None
    retraining = evaluate_retraining_triggers(runs)
    return {"latest": latest, "runs": runs, "retraining": retraining}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
