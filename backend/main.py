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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load_geojson() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_telemetry_samples() -> list[dict]:
    return _load_json(TELEMETRY_FILE)


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


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _telemetry_in_window(timestamp_lte: str | None) -> tuple[list[dict], datetime | None]:
    telemetry = load_telemetry_samples()
    cutoff = _parse_timestamp(timestamp_lte) if timestamp_lte else None
    if cutoff:
        telemetry = [sample for sample in telemetry if _parse_timestamp(sample["timestamp"]) <= cutoff]
    return telemetry, cutoff


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
        cutoff = _parse_timestamp(timestamp_lte)
        features = [f for f in features if _parse_timestamp(f["properties"]["timestamp"]) <= cutoff]
    return {"type": "FeatureCollection", "features": features}


@app.get("/api/spectrum/timeseries")
def spectrum_timeseries(timestamp_lte: str | None = None) -> dict:
    telemetry, _ = _telemetry_in_window(timestamp_lte)
    buckets: dict[str, dict[str, list[float]]] = {}
    for sample in telemetry:
        ts = sample["timestamp"]
        band = sample.get("band", "unknown")
        buckets.setdefault(ts, {}).setdefault(band, []).append(sample.get("snr_db", 0.0))

    series = []
    for ts in sorted(buckets.keys()):
        occupancy = {band: round(sum(values) / max(len(values), 1), 3) for band, values in buckets[ts].items()}
        series.append({"timestamp": ts, "occupancy": occupancy})
    return {"series": series, "sample_count": len(telemetry)}


@app.get("/api/spectrum/waterfall")
def spectrum_waterfall(timestamp_lte: str | None = None) -> dict:
    telemetry, cutoff = _telemetry_in_window(timestamp_lte)
    times = sorted({sample["timestamp"] for sample in telemetry})
    bands = sorted({sample.get("band", "unknown") for sample in telemetry})

    heatmap = []
    for band in bands:
        row = []
        for ts in times:
            values = [s.get("rssi_dbm", 0.0) for s in telemetry if s.get("band", "unknown") == band and s["timestamp"] == ts]
            row.append(round(sum(values) / len(values), 3) if values else None)
        heatmap.append({"band": band, "values": row})

    freshness_seconds = None
    if telemetry and cutoff:
        latest = max(_parse_timestamp(s["timestamp"]) for s in telemetry)
        freshness_seconds = max(0.0, (cutoff - latest).total_seconds())

    return {
        "times": times,
        "rows": heatmap,
        "provenance": {
            "device_id": "sim-rf-probe-1",
            "adapter_type": "rtl-sdr",
            "sample_count": len(telemetry),
            "freshness_seconds": freshness_seconds,
        },
    }


@app.get("/api/spectrum/occupancy")
def spectrum_occupancy(site_id: str | None = None, timestamp_lte: str | None = None) -> dict:
    telemetry, cutoff = _telemetry_in_window(timestamp_lte)
    bands: dict[str, dict] = {}
    for sample in telemetry:
        band = sample.get("band", "unknown")
        band_row = bands.setdefault(band, {"samples": 0, "rssi": 0.0, "snr": 0.0})
        band_row["samples"] += 1
        band_row["rssi"] += sample.get("rssi_dbm", 0.0)
        band_row["snr"] += sample.get("snr_db", 0.0)

    summary = []
    for band, metrics in sorted(bands.items()):
        n = max(metrics["samples"], 1)
        summary.append(
            {
                "band": band,
                "sample_count": metrics["samples"],
                "avg_rssi_dbm": round(metrics["rssi"] / n, 3),
                "avg_snr_db": round(metrics["snr"] / n, 3),
            }
        )

    return {
        "site_id": site_id,
        "window_end": cutoff.isoformat() if cutoff else None,
        "bands": summary,
        "spectral_stats": {
            "total_samples": len(telemetry),
            "strongest_band": max(summary, key=lambda row: row["avg_snr_db"])["band"] if summary else None,
        },
    }


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

    timestamps = [_parse_timestamp(s["timestamp"]) for s in ingestion_result.accepted]
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
