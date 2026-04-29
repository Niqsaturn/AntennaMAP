from __future__ import annotations

import json
import threading
import time
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


class LoopManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._config = {"interval_seconds": 60, "provider": "local", "model": "baseline-v1"}
        self._last_run_started_at: str | None = None
        self._last_run_completed_at: str | None = None
        self._last_successful_run_at: str | None = None
        self._last_run_duration_ms: float | None = None
        self._provider_errors: list[dict] = []

    def _record_error(self, provider: str, operation: str, exc: Exception) -> None:
        self._provider_errors = [
            *self._provider_errors[-24:],
            {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "provider": provider,
                "operation": operation,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        ]

    def _run_once(self) -> None:
        started = datetime.now(tz=timezone.utc)
        cfg = self.config()
        with self._lock:
            self._last_run_started_at = started.isoformat()

        try:
            ingest_pipeline(model_version=cfg["model"])
            with self._lock:
                self._last_successful_run_at = datetime.now(tz=timezone.utc).isoformat()
        except Exception as exc:
            with self._lock:
                self._record_error(cfg["provider"], "ingest_pipeline", exc)
        finally:
            done = datetime.now(tz=timezone.utc)
            with self._lock:
                self._last_run_completed_at = done.isoformat()
                self._last_run_duration_ms = round((done - started).total_seconds() * 1000, 3)

    def _loop(self) -> None:
        while self.active():
            self._run_once()
            time.sleep(max(1, int(self.config()["interval_seconds"])))

    def start(self) -> dict:
        with self._lock:
            if self._running:
                return {"active": True}
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return {"active": True}

    def stop(self) -> dict:
        with self._lock:
            self._running = False
            return {"active": False}

    def active(self) -> bool:
        with self._lock:
            return self._running

    def update_config(self, interval_seconds: int, provider: str, model: str) -> dict:
        with self._lock:
            self._config = {"interval_seconds": interval_seconds, "provider": provider, "model": model}
            return self._config.copy()

    def config(self) -> dict:
        with self._lock:
            return self._config.copy()

    def status(self) -> dict:
        with self._lock:
            return {
                "active": self._running,
                "config": self._config.copy(),
                "last_run": {
                    "started_at": self._last_run_started_at,
                    "completed_at": self._last_run_completed_at,
                    "duration_ms": self._last_run_duration_ms,
                    "last_successful_run_at": self._last_successful_run_at,
                },
                "provider_errors": self._provider_errors,
            }


loop_manager = LoopManager()


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
    return {"status": "ok", "service": "antennamap"}


@app.post("/api/loop/pause")
def pause_loop() -> dict:
    return loop_manager.stop()


@app.post("/api/loop/resume")
def resume_loop() -> dict:
    return loop_manager.start()


@app.post("/api/loop/config")
def config_loop(interval_seconds: int = 60, provider: str = "local", model: str = "baseline-v1") -> dict:
    return {"config": loop_manager.update_config(interval_seconds, provider, model)}


@app.get("/api/loop/status")
def loop_status() -> dict:
    return loop_manager.status()


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


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
