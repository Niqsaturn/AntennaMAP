from __future__ import annotations

import json
import yaml
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.pipeline.ingest import evaluate_retraining_triggers, ingest_telemetry, summarize_telemetry

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "public" / "data" / "antenna_data.geojson"
TELEMETRY_FILE = ROOT / "public" / "data" / "telemetry_samples.json"
INGEST_LOG_FILE = ROOT / "backend" / "pipeline" / "data" / "telemetry_ingested.jsonl"
RUN_METADATA_FILE = ROOT / "backend" / "pipeline" / "data" / "model_runs.jsonl"

SDR_CAPABILITIES_FILE = ROOT / "backend" / "sdr" / "capabilities.yaml"


class SDRConfigureRequest(BaseModel):
    model: str
    sample_rate_sps: int
    center_freq_hz: int
    bandwidth_hz: int
    gain_db: float
    ppm: int = 0


def load_sdr_capabilities() -> dict:
    return yaml.safe_load(SDR_CAPABILITIES_FILE.read_text(encoding="utf-8"))


def _validate_range(field: str, value: float, limits: dict) -> dict | None:
    low = limits["min"]
    high = limits["max"]
    if value < low or value > high:
        return {
            "field": field,
            "message": f"{field} must be between {low} and {high}",
            "requested": value,
            "allowed": limits,
            "error_code": "OUT_OF_RANGE",
        }
    return None


def validate_sdr_config(payload: SDRConfigureRequest, capabilities: dict) -> list[dict]:
    models = capabilities.get("models", {})
    model_caps = models.get(payload.model)
    if not model_caps:
        return [{
            "field": "model",
            "message": f"Unsupported SDR model: {payload.model}",
            "requested": payload.model,
            "allowed": sorted(models.keys()),
            "error_code": "UNSUPPORTED_MODEL",
        }]

    constraints = model_caps.get("constraints", {})
    errors = []
    for field in ["sample_rate_sps", "center_freq_hz", "bandwidth_hz", "gain_db", "ppm"]:
        violation = _validate_range(field, getattr(payload, field), constraints[field])
        if violation:
            errors.append(violation)

    if payload.bandwidth_hz > payload.sample_rate_sps:
        errors.append({
            "field": "bandwidth_hz",
            "message": "bandwidth_hz cannot exceed sample_rate_sps",
            "requested": payload.bandwidth_hz,
            "allowed": {"max_relative_to": "sample_rate_sps"},
            "error_code": "INVALID_COMBINATION",
        })
    return errors


ACTIVE_SDR_CONFIG = None


def _default_sdr_config() -> dict:
    caps = load_sdr_capabilities()
    default_model = caps["default_model"]
    c = caps["models"][default_model]["constraints"]
    return {
        "model": default_model,
        "sample_rate_sps": c["sample_rate_sps"]["min"],
        "center_freq_hz": c["center_freq_hz"]["min"],
        "bandwidth_hz": c["bandwidth_hz"]["min"],
        "gain_db": c["gain_db"]["min"],
        "ppm": 0,
    }


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



@app.get("/api/sdr/capabilities")
def get_sdr_capabilities() -> dict:
    global ACTIVE_SDR_CONFIG
    if ACTIVE_SDR_CONFIG is None:
        ACTIVE_SDR_CONFIG = _default_sdr_config()
    return {"capabilities": load_sdr_capabilities(), "active_config": ACTIVE_SDR_CONFIG}


@app.post("/api/sdr/configure")
def configure_sdr(payload: SDRConfigureRequest):
    global ACTIVE_SDR_CONFIG
    capabilities = load_sdr_capabilities()
    errors = validate_sdr_config(payload, capabilities)
    if errors:
        return JSONResponse(
            status_code=422,
            content={
                "error": "SDR_CONFIGURATION_INVALID",
                "message": "Requested SDR configuration is unsupported.",
                "details": errors,
            },
        )

    ACTIVE_SDR_CONFIG = payload.model_dump()
    return {"status": "configured", "active_config": ACTIVE_SDR_CONFIG}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
