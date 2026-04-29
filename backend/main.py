from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.ingest.infrastructure import ingest_infrastructure
from backend.ingest.storage import append_jsonl, read_jsonl
from backend.ingest.telemetry import ingest_telemetry
from backend.pipeline.ingest import evaluate_retraining_triggers, summarize_telemetry

from backend.geometry.rf_overlays import build_overlay_geometries

def load_telemetry_samples() -> list[dict]:
    if not TELEMETRY_FILE.exists():
        return []
    return _load_json(TELEMETRY_FILE)

def enrich_feature(feature: dict, telemetry: list[dict]) -> dict:
    enriched = {**feature}
    props = {**feature.get("properties", {})}
    if feature.get("geometry", {}).get("type") == "Point":
        props["beamwidth_deg"] = props.get("beamwidth_deg") or (360 if props.get("directionality") == "Omni" else 80)
        props["ray_length_m"] = props.get("ray_length_m") or (750 if props.get("kind") == "estimate" else 1000)
        props["wedge_radius_m"] = props.get("wedge_radius_m") or (900 if props.get("kind") == "estimate" else 1400)
        props["overlay_geometries"] = build_overlay_geometries({"type":"Feature","geometry":feature.get("geometry"),"properties":props})
    enriched["properties"] = props
    return enriched

from backend.training.trainer import train_single_triangulation_baseline
from backend.training.triangulation_baseline import estimate_single_operator

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "public" / "data" / "antenna_data.geojson"
TELEMETRY_FILE = ROOT / "public" / "data" / "telemetry_samples.json"
INGEST_LOG_FILE = ROOT / "backend" / "pipeline" / "data" / "telemetry_ingested.jsonl"
INFRA_INGEST_FILE = ROOT / "backend" / "pipeline" / "data" / "infrastructure_ingested.jsonl"
RUN_METADATA_FILE = ROOT / "backend" / "pipeline" / "data" / "model_runs.jsonl"
INGEST_ISSUES_FILE = ROOT / "backend" / "pipeline" / "data" / "ingest_issues.jsonl"

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


def _mock_adapter_fetcher() -> list[dict]:
    return []


sdr_service = SDRIngestService(
    adapter_fetcher=_mock_adapter_fetcher,
    storage=SDRStoragePaths(
        raw_jsonl=ROOT / "backend" / "ingest" / "data" / "sdr_raw.jsonl",
        aggregates_jsonl=ROOT / "backend" / "ingest" / "data" / "sdr_aggregates.jsonl",
        reject_jsonl=ROOT / "backend" / "ingest" / "data" / "sdr_rejected.jsonl",
        sqlite_file=ROOT / "backend" / "ingest" / "data" / "sdr_ingest.sqlite3",
    ),
    poll_interval_s=1.0,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

foxhunt_service = FoxHuntService()


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


@app.get("/api/features/count")
def get_features_count(kind: str | None = Query(default=None, pattern="^(infrastructure|estimate)?$"), timestamp_lte: str | None = None) -> dict:
    payload = get_features(kind=kind, timestamp_lte=timestamp_lte)
    return {"count": len(payload["features"]), "kind": kind, "timestamp_lte": timestamp_lte}


@app.post("/api/pipeline/ingest")
def ingest_pipeline(model_version: str = "baseline-v1") -> dict:
    raw_samples = load_telemetry_samples()
    telemetry = ingest_telemetry(raw_samples, INGEST_LOG_FILE)

    raw_features = load_geojson().get("features", [])
    infrastructure = ingest_infrastructure(raw_features, INFRA_INGEST_FILE)

    summary = summarize_telemetry(telemetry.accepted)
    issues = telemetry.errors + telemetry.warnings + infrastructure.errors
    append_jsonl(INGEST_ISSUES_FILE, [{"run_id": datetime.now(tz=timezone.utc).isoformat(), "issues": issues}])

    runtime = _load_runtime_config()
    selected_method = runtime.get("selected_method", "single_triangulation_baseline")
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
