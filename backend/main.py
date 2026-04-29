from __future__ import annotations

import json
import threading
import time
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
