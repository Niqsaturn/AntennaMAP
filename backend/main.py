from __future__ import annotations

import json
import threading
import time
import yaml
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.foxhunt.models import FoxHuntObservation
from backend.foxhunt.service import FoxHuntService
from backend.geometry.rf_overlays import build_overlay_geometries
from backend.ingest.infrastructure import ingest_infrastructure
from backend.ingest.sdr_ingest import SDRIngestService, SDRStoragePaths
from backend.ingest.storage import append_jsonl, read_jsonl
from backend.ingest.telemetry import ingest_telemetry
from backend.ml.triangulation_pipeline import (
    TelemetrySample as MLTelemetrySample,
    load_latest_model,
    parse_samples,
    predict_error,
    solve_weighted_least_squares,
    train_error_model,
)
from backend.analysis.signal_detector import detect_signals
from backend.analysis.array_speculator import speculate_from_detections
from backend.analysis.us_coverage import advance_queue as coverage_advance_queue, next_tiles_to_process
from backend.analysis.examples import (
    add_confirmed_example,
    get_recent_examples,
    format_examples_for_prompt,
)
from backend.datasources.fcc import search_licenses_near
from backend.datasources.cell_towers import search_towers_near
from backend.datasources.spectrum_allocations import (
    allocations_for_freq,
    allocations_in_range,
    known_signals_for_band,
    full_spectrum_summary,
)
from backend.datasources.satellites import (
    fetch_tle_group,
    tle_to_position,
    satellite_ground_track,
    satellite_footprint,
    uplink_downlink_arc,
    build_relay_map,
)
from backend.datasources.earth_stations import EARTH_STATIONS, get_stations_near
from backend.pipeline.compliance import COMPLIANCE_POLICY
from backend.pipeline.ingest import evaluate_retraining_triggers, summarize_telemetry
from backend.rf.array_calculator import (
    LinearArrayParams,
    PlanarArrayParams,
    RadarParams,
    linear_array_pattern,
    planar_array_pattern,
    radar_range_equation,
    estimate_rcs,
    link_budget,
    bearing_from_phase_difference,
)
from backend.rf_propagation import (
    ModelName,
    generate_snapshot,
    em_field_grid,
    ionospheric_skip_distance_km,
    atmospheric_absorption_db,
)
from backend.sdr.computed_spectrum import computed_psd, psd_to_metrics
from backend.storage.map_store import (
    upsert_feature,
    features_in_bounds,
    get_all_features,
    get_coverage_grid,
    get_coverage_progress,
    log_analysis,
    get_analysis_log,
    update_feature_kind,
    update_tile_status,
    upsert_tile,
    get_uncertain_features,
    get_calibration_stats,
    record_position_error,
)
from backend.analysis.waterfall_analyzer import analyze_psd
from backend.analysis.range_estimator import estimate_range_km
from backend.analysis.bearing_tracker import estimate_bearing as estimate_bearing_track
from backend.analysis.kalman_tracker import PositionKalman
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
MODELS_DIR = ROOT / "models"
ML_MODELS_DIR = ROOT / "backend" / "ml" / "models"
RUNTIME_CONFIG_FILE = ROOT / "backend" / "pipeline" / "data" / "runtime_config.json"
TRAINING_STATUS_FILE = ROOT / "backend" / "pipeline" / "data" / "training_status.json"


class SDRConfigureRequest(BaseModel):
    model: str
    sample_rate_sps: int
    center_freq_hz: int
    bandwidth_hz: int
    gain_db: float
    ppm: int = 0


class KiwiConnectRequest(BaseModel):
    host: str
    port: int = 8073
    center_freq_hz: float = 7_100_000.0


class MLTrainRequest(BaseModel):
    samples: list[dict]
    target_error_m: list[float]
    model_version: str = ""


class MLSolveRequest(BaseModel):
    samples: list[dict]


class ArrayCalculateRequest(BaseModel):
    array_type: str = "linear"
    n_elements: int = 8
    element_spacing_m: float = 0.0      # 0 → auto λ/2
    frequency_hz: float = 433.92e6
    steering_angle_deg: float = 0.0
    window: str = "uniform"
    # Planar extras
    n_y: int = 8
    dy_m: float = 0.0
    steer_az_deg: float = 0.0
    steer_el_deg: float = 0.0


class RadarEstimateRequest(BaseModel):
    peak_power_w: float = 1000.0
    antenna_gain_dbi: float = 30.0
    frequency_hz: float = 10e9
    rcs_m2: float | None = None
    target_type: str = "car"
    noise_figure_db: float = 5.0
    bandwidth_hz: float = 1e6
    losses_db: float = 3.0


class LinkBudgetRequest(BaseModel):
    tx_power_dbm: float = 30.0
    tx_gain_dbi: float = 15.0
    rx_gain_dbi: float = 5.0
    frequency_hz: float = 900e6
    distance_km: float = 10.0
    rx_noise_figure_db: float = 5.0
    bandwidth_hz: float = 200e3
    losses_db: float = 2.0
    model: str = "fspl"


class ComputedSpectrumRequest(BaseModel):
    center_freq_hz: float = 433.92e6
    sample_rate_hz: float = 2.4e6
    n_bins: int = 64
    provider: str = "rtlsdr"
    use_band_allocations: bool = True
    known_signals: list[dict] = []


class AIAnalyzeRequest(BaseModel):
    model: str
    prompt: str = ""
    spectrum_data: dict = {}
    context: str = ""


class AnalysisRunRequest(BaseModel):
    provider: str = "ollama"
    model: str = ""
    lat: float | None = None
    lon: float | None = None
    limit_samples: int = 50


class ConfirmDetectionRequest(BaseModel):
    feature_id: str
    confirmed: bool
    true_lat: float | None = None
    true_lon: float | None = None


class UplinkPathRequest(BaseModel):
    ground_lat: float
    ground_lon: float
    norad_id: str = ""
    sat_name: str = ""
    sat_group: str = "geo"


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
_active_kiwi_adapter = None


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


def _kiwi_adapter_fetcher() -> list[dict]:
    """Fetch SDR frames from all configured KiwiSDR nodes."""
    from backend.sdr.kiwisdr_client import node_pool
    return node_pool.scan_peaks_all(min_snr_db=3.0)


sdr_service = SDRIngestService(
    adapter_fetcher=_kiwi_adapter_fetcher,
    storage=SDRStoragePaths(
        raw_jsonl=ROOT / "backend" / "ingest" / "data" / "sdr_raw.jsonl",
        aggregates_jsonl=ROOT / "backend" / "ingest" / "data" / "sdr_aggregates.jsonl",
        reject_jsonl=ROOT / "backend" / "ingest" / "data" / "sdr_rejected.jsonl",
        sqlite_file=ROOT / "backend" / "ingest" / "data" / "sdr_ingest.sqlite3",
    ),
    poll_interval_s=15.0,
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
            # Run AI analysis cycle when a model is selected
            if cfg.get("provider") in ("ollama", "python_local"):
                try:
                    _run_analysis_cycle(cfg)
                except Exception as exc:
                    with self._lock:
                        self._record_error(cfg["provider"], "analysis_cycle", exc)
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


def _run_analysis_cycle(cfg: dict) -> None:
    """Full SDR analysis cycle: waterfall peaks → range estimation → triangulation → Kalman → map store."""
    import asyncio
    import math as _math
    import time as _time
    t0 = _time.monotonic()

    samples = read_jsonl(INGEST_LOG_FILE)[-cfg.get("limit_samples", 50):]
    if not samples:
        return

    # Operator position from most recent telemetry with GPS
    op_lat = next((s["lat"] for s in reversed(samples) if s.get("lat")), 0.0)
    op_lon = next((s["lon"] for s in reversed(samples) if s.get("lon")), 0.0)

    # ------------------------------------------------------------------
    # Waterfall analysis: enrich samples that carry PSD bins
    # ------------------------------------------------------------------
    for s in samples:
        psd = s.get("psd_bins_db")
        if not psd and isinstance(s.get("spectral"), dict):
            psd = s["spectral"].get("psd_bins_db")
        if psd and len(psd) >= 4:
            freq_hz = s.get("frequency_hz") or s.get("freq_hz") or 100e6
            sr = s.get("sample_rate_hz", 2_400_000.0) or 2_400_000.0
            try:
                peaks = analyze_psd(psd, float(freq_hz), float(sr))
                if peaks:
                    s["_spectral_peaks"] = [
                        {"freq_hz": p.center_freq_hz, "bw_hz": p.bandwidth_3db_hz,
                         "peak_dbm": p.peak_dbm, "snr_db": p.snr_db,
                         "modulation_hint": p.modulation_hint}
                        for p in peaks
                    ]
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Triangulation: build TelemetrySamples and run WLS if geometry is good
    # ------------------------------------------------------------------
    triangulation_anchor: dict | None = None
    tri_samples_raw = [
        s for s in samples
        if s.get("bearing_deg") is not None and s.get("lat") and s.get("lon")
    ]
    if len(tri_samples_raw) >= 3:
        try:
            ml_samples = [
                MLTelemetrySample(
                    timestamp=s.get("timestamp", ""),
                    lat=float(s["lat"]),
                    lon=float(s["lon"]),
                    heading_deg=float(s.get("heading_deg", s.get("bearing_deg", 0))),
                    bearing_estimate_deg=float(s["bearing_deg"]),
                    rssi_dbm=float(s.get("rssi_dbm", -100)),
                    snr_db=float(s.get("snr_db", 0)),
                    frequency_hz=float(s.get("frequency_hz") or s.get("freq_hz") or 1e8),
                    bandwidth_hz=float(s.get("bandwidth_hz", 200_000)),
                )
                for s in tri_samples_raw
            ]
            wls = solve_weighted_least_squares(ml_samples)
            if wls.quality_score > 0.3:
                triangulation_anchor = {
                    "lat": wls.center_lat,
                    "lon": wls.center_lon,
                    "uncertainty_m": wls.ellipse_major_m / 2.0,
                    "quality_score": wls.quality_score,
                }
        except Exception:
            pass

    # ------------------------------------------------------------------
    # FCC context
    # ------------------------------------------------------------------
    try:
        fcc_context = asyncio.run(search_licenses_near(op_lat, op_lon, radius_km=25.0, limit=20))
    except Exception:
        fcc_context = []

    # Few-shot examples for Ollama
    examples = get_recent_examples(3)
    examples_text = format_examples_for_prompt(examples)

    # ------------------------------------------------------------------
    # Signal detection with selected model
    # ------------------------------------------------------------------
    model_config = {"provider": cfg.get("provider", "local"), "model": cfg.get("model", "")}
    result = detect_signals(
        samples, model_config, fcc_context, examples_text, op_lat, op_lon
    )

    # ------------------------------------------------------------------
    # Generate and persist speculative features
    # ------------------------------------------------------------------
    speculative = []
    if result.detections:
        speculative = speculate_from_detections(result.detections, op_lat, op_lon)

        # Override position with triangulation anchor when WLS quality is high
        if triangulation_anchor and triangulation_anchor["quality_score"] > 0.6:
            for feat in speculative:
                props = feat.get("properties", {})
                if props.get("kind") == "speculative":
                    feat["geometry"]["coordinates"] = [
                        triangulation_anchor["lon"],
                        triangulation_anchor["lat"],
                    ]
                    props["position_source"] = "triangulation_wls"
                    props["triangulation_quality"] = round(triangulation_anchor["quality_score"], 3)

        for feat in speculative:
            try:
                upsert_feature(feat)
                # Apply Kalman smoothing to persisted position
                fid = feat.get("properties", {}).get("id")
                coords = feat.get("geometry", {}).get("coordinates", [])
                if fid and len(coords) == 2:
                    unc_m = triangulation_anchor["uncertainty_m"] if triangulation_anchor else 2000.0
                    kf = PositionKalman.for_feature(fid, float(coords[1]), float(coords[0]))
                    kf.predict()
                    s_lat, s_lon, s_unc = kf.update(float(coords[1]), float(coords[0]), unc_m)
                    # Update stored position with Kalman-smoothed estimate
                    feat["geometry"]["coordinates"] = [s_lon, s_lat]
                    feat.get("properties", {})["kalman_uncertainty_m"] = round(s_unc, 1)
                    upsert_feature(feat)
                    kf.save(fid)
                # Auto-generate EM field overlay for high-confidence features
                try:
                    from backend.analysis.em_field_auto import maybe_generate_em_field
                    maybe_generate_em_field(feat)
                except Exception:
                    pass
            except Exception as e:
                logger.warning("failed to save kalman state for %s: %s", fid, e)

    # ------------------------------------------------------------------
    # Mark operator's current tile as analyzed
    # ------------------------------------------------------------------
    if op_lat != 0.0 and op_lon != 0.0:
        try:
            import math as _m
            tile_deg = 0.5
            lat_f = round(_m.floor(op_lat / tile_deg) * tile_deg, 1)
            lon_f = round(_m.floor(op_lon / tile_deg) * tile_deg, 1)
            tile_id = f"{lat_f:.1f}_{lon_f:.1f}"
            update_tile_status(tile_id, "analyzed")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Advance US coverage queue (seed next tiles)
    # ------------------------------------------------------------------
    try:
        asyncio.run(coverage_advance_queue(op_lat, op_lon, n=2))
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Auto-retraining check (python_local only)
    # ------------------------------------------------------------------
    if cfg.get("provider") == "python_local":
        try:
            triggers = evaluate_retraining_triggers()
            if triggers.get("should_retrain"):
                # Kick off retraining in background — avoid blocking the loop
                threading.Thread(
                    target=lambda: train_single_triangulation_baseline(INGEST_LOG_FILE),
                    daemon=True,
                ).start()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Log analysis result
    # ------------------------------------------------------------------
    ms = (_time.monotonic() - t0) * 1000
    try:
        log_analysis(
            model=cfg.get("model", ""),
            provider=cfg.get("provider", "local"),
            input_summary=f"{len(samples)} samples, lat={op_lat:.3f}, lon={op_lon:.3f}, tri_anchor={'yes' if triangulation_anchor else 'no'}",
            detections_count=len(result.detections),
            detections_json=json.dumps([d.__dict__ if hasattr(d, '__dict__') else str(d)
                                        for d in result.detections], default=str),
            raw_response=result.raw_response[:2000] if result.raw_response else "",
            processing_ms=round(ms, 1),
        )
    except Exception:
        pass


def load_geojson() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_telemetry_samples() -> list[dict]:
    if not TELEMETRY_FILE.exists():
        return []
    return _load_json(TELEMETRY_FILE)


def enrich_feature(feature: dict, telemetry: list[dict]) -> dict:
    _ = telemetry
    enriched = {**feature}
    props = {**feature.get("properties", {})}
    if feature.get("geometry", {}).get("type") == "Point":
        props["beamwidth_deg"] = props.get("beamwidth_deg") or (360 if props.get("directionality") == "Omni" else 80)
        props["ray_length_m"] = props.get("ray_length_m") or (750 if props.get("kind") == "estimate" else 1000)
        props["wedge_radius_m"] = props.get("wedge_radius_m") or (900 if props.get("kind") == "estimate" else 1400)
        props["overlay_geometries"] = build_overlay_geometries({"type": "Feature", "geometry": feature.get("geometry"), "properties": props})
    enriched["properties"] = props
    return enriched


def policy_status() -> dict:
    return {
        "policy_id": COMPLIANCE_POLICY["policy_id"],
        "metadata_only": COMPLIANCE_POLICY["metadata_only"],
        "retention_days": COMPLIANCE_POLICY["retention_days"],
    }


def _load_runtime_config() -> dict:
    if RUNTIME_CONFIG_FILE.exists():
        return json.loads(RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
    return {"selected_method": "single_triangulation_baseline", "selected_model": "baseline-v1"}


def _write_training_status(status: dict) -> None:
    TRAINING_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")


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


def discover_python_local_models() -> list[str]:
    return sorted(f.stem for f in MODELS_DIR.glob("*.json")) if MODELS_DIR.exists() else []


def discover_available_models() -> dict[str, list[str]]:
    return {"ollama": discover_ollama_models(), "python_local": discover_python_local_models()}


def run_inference(provider: str, model: str, payload: dict) -> dict:
    if provider == "ollama":
        body = json.dumps({"model": model, **payload}).encode("utf-8")
        req = request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=30.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    if provider == "python_local":
        samples_raw = payload.get("samples", [])
        if samples_raw:
            try:
                ml_samples = parse_samples(samples_raw)
                result = solve_weighted_least_squares(ml_samples)
                trained_model = load_latest_model()
                predicted_error = predict_error(trained_model, ml_samples) if trained_model else None
                return {
                    "provider": provider,
                    "model": model,
                    "inference": asdict(result),
                    "predicted_error_m": predicted_error,
                }
            except Exception as exc:
                return {"provider": provider, "model": model, "error": str(exc)}
        return {"provider": provider, "model": model, "output": "no samples provided"}
    raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")


def _parse_timestamp(value: str) -> datetime | None:
    """Parse ISO timestamp string. Returns None for invalid/sentinel values."""
    if not value or value.lower() in ("undefined", "null", "none", ""):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _telemetry_in_window(timestamp_lte: str | None) -> tuple[list[dict], datetime | None]:
    telemetry = load_telemetry_samples()
    cutoff = _parse_timestamp(timestamp_lte) if timestamp_lte else None
    if cutoff:
        telemetry = [sample for sample in telemetry if _parse_timestamp(sample["timestamp"]) <= cutoff]
    return telemetry, cutoff


# ── Health & Loop ──────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "antennamap", **policy_status()}


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


# ── Features ──────────────────────────────────────────────────────────────────

@app.get("/api/features")
def get_features(kind: str | None = Query(default=None, pattern="^(infrastructure|estimate)?$"), timestamp_lte: str | None = None) -> dict:
    data = load_geojson()
    features = data.get("features", [])

    if kind:
        features = [f for f in features if f.get("properties", {}).get("kind") == kind]
    if timestamp_lte:
        cutoff = _parse_timestamp(timestamp_lte)
        if cutoff is not None:  # skip filter for invalid/undefined values
            features = [
                f for f in features
                if (ts := _parse_timestamp(f["properties"].get("timestamp", ""))) and ts <= cutoff
            ]

    features = [enrich_feature(f, []) for f in features]
    return {"type": "FeatureCollection", "features": features}


@app.get("/api/features/count")
def get_features_count(kind: str | None = Query(default=None, pattern="^(infrastructure|estimate)?$"), timestamp_lte: str | None = None) -> dict:
    payload = get_features(kind=kind, timestamp_lte=timestamp_lte)
    return {"count": len(payload["features"]), "kind": kind, "timestamp_lte": timestamp_lte}


# ── SDR ───────────────────────────────────────────────────────────────────────

@app.get("/api/sdr/capabilities")
def sdr_capabilities() -> dict:
    caps = load_sdr_capabilities()
    active = ACTIVE_SDR_CONFIG or _default_sdr_config()
    return {"active_config": active, "capabilities": caps}


@app.post("/api/sdr/kiwi/connect")
def kiwi_connect(body: KiwiConnectRequest) -> dict:
    global _active_kiwi_adapter, ACTIVE_SDR_CONFIG
    from backend.sdr.adapters.kiwisdr_adapter import KiwiSdrAdapter
    _active_kiwi_adapter = KiwiSdrAdapter(config={
        "host": body.host,
        "port": body.port,
        "center_freq_hz": body.center_freq_hz,
    })
    _active_kiwi_adapter.connect()
    ACTIVE_SDR_CONFIG = {
        "model": "kiwisdr",
        "host": body.host,
        "port": body.port,
        "center_freq_hz": int(body.center_freq_hz),
        "sample_rate_sps": 12000,
        "bandwidth_hz": 12000,
        "gain_db": 0,
        "ppm": 0,
    }
    return {
        "status": "connected",
        "host": body.host,
        "port": body.port,
        "websocket_available": _active_kiwi_adapter.read_device_metadata().extras.get("websocket_available", False),
    }


# ── Models ────────────────────────────────────────────────────────────────────

@app.get("/api/models/discover")
def models_discover() -> dict:
    return discover_available_models()


# ── Pipeline & Training ───────────────────────────────────────────────────────

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
    append_jsonl(RUN_METADATA_FILE, [{"run_id": datetime.now(tz=timezone.utc).isoformat(), "training": artifact}])
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


# ── ML Pipeline (error calibration + triangulation) ───────────────────────────

@app.post("/api/ml/train")
def ml_train(body: MLTrainRequest) -> dict:
    if len(body.samples) != len(body.target_error_m):
        raise HTTPException(status_code=400, detail="samples and target_error_m must have equal length")
    if len(body.samples) < 2:
        raise HTTPException(status_code=400, detail="at least 2 samples required for training")

    version = body.model_version or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        ml_samples = parse_samples(body.samples)
        result = train_error_model(ml_samples, body.target_error_m, version)
        return {
            "model_version": version,
            "rmse": result.rmse,
            "sample_count": result.metadata.sample_count,
            "trained_at": result.metadata.trained_at,
            "feature_schema": result.metadata.feature_schema,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/ml/solve")
def ml_solve(body: MLSolveRequest) -> dict:
    if not body.samples:
        raise HTTPException(status_code=400, detail="samples list is empty")
    try:
        ml_samples = parse_samples(body.samples)
        result = solve_weighted_least_squares(ml_samples)
        trained_model = load_latest_model()
        predicted_error = None
        if trained_model and len(ml_samples) >= 2:
            try:
                predicted_error = round(predict_error(trained_model, ml_samples), 2)
            except Exception:
                pass
        return {
            "center_lat": result.center_lat,
            "center_lon": result.center_lon,
            "covariance": result.covariance,
            "ellipse_major_m": result.ellipse_major_m,
            "ellipse_minor_m": result.ellipse_minor_m,
            "ellipse_angle_deg": result.ellipse_angle_deg,
            "quality_score": result.quality_score,
            "predicted_error_m": predicted_error,
            "num_samples": len(ml_samples),
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/ml/models")
def ml_models() -> dict:
    files = sorted(ML_MODELS_DIR.glob("triangulation_*.json"))
    models = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            meta = data.get("metadata", {})
            models.append({
                "filename": f.name,
                "model_version": meta.get("model_version"),
                "trained_at": meta.get("trained_at"),
                "sample_count": meta.get("sample_count"),
                "rmse": data.get("rmse"),
            })
        except Exception:
            pass
    return {"models": models, "count": len(models)}


@app.get("/api/telemetry/samples")
def telemetry_samples(limit: int = 100) -> dict:
    rows = read_jsonl(INGEST_LOG_FILE)
    return {"count": len(rows), "samples": rows[-limit:]}


# ── FoxHunt ───────────────────────────────────────────────────────────────────

@app.post("/api/foxhunt/session/start")
def foxhunt_start_session() -> dict:
    session = foxhunt_service.start_session()
    return session.model_dump(mode="json")


@app.get("/api/foxhunt/sessions")
def foxhunt_list_sessions() -> dict:
    sessions = []
    for s in foxhunt_service.sessions.values():
        sessions.append({
            "id": s.id,
            "status": s.status,
            "started_at": s.started_at.isoformat(),
            "stopped_at": s.stopped_at.isoformat() if s.stopped_at else None,
            "observation_count": len(s.observations),
            "has_estimate": s.estimate is not None,
        })
    return {"sessions": sessions, "count": len(sessions)}


@app.get("/api/foxhunt/session/{session_id}")
def foxhunt_get_session(session_id: str) -> dict:
    if session_id not in foxhunt_service.sessions:
        raise HTTPException(status_code=404, detail="session not found")
    return foxhunt_service.get_session(session_id).model_dump(mode="json")


@app.post("/api/foxhunt/session/stop")
def foxhunt_stop_session(session_id: str) -> dict:
    if session_id not in foxhunt_service.sessions:
        raise HTTPException(status_code=404, detail="session not found")
    return foxhunt_service.stop_session(session_id).model_dump(mode="json")


@app.post("/api/foxhunt/session/{session_id}/observation")
def foxhunt_add_observation(session_id: str, observation: FoxHuntObservation) -> dict:
    if session_id not in foxhunt_service.sessions:
        raise HTTPException(status_code=404, detail="session not found")
    session = foxhunt_service.append_observation(session_id, observation)
    return session.model_dump(mode="json")


# ── Propagation ───────────────────────────────────────────────────────────────

@app.get("/api/propagation")
def propagation_snapshot(site_id: str, model: ModelName = "fspl") -> dict:
    data = load_geojson()
    feature = next(
        (f for f in data.get("features", []) if f.get("properties", {}).get("id") == site_id),
        None,
    )
    if feature is None:
        raise HTTPException(status_code=404, detail=f"site_id '{site_id}' not found")

    coords = feature["geometry"]["coordinates"]
    lon, lat = coords[0], coords[1]
    props = feature.get("properties", {})
    freq = ((props.get("rf_min_mhz") or 900) + (props.get("rf_max_mhz") or 2600)) / 2.0
    beamwidth = float(props.get("beamwidth_deg") or 360)
    orientation = float(props.get("azimuth_deg") or 0)

    snapshot = generate_snapshot(
        lat=lat,
        lon=lon,
        eirp_dbm=43.0,
        frequency_mhz=freq,
        gain_dbi=15.0,
        height_m=30.0,
        beamwidth_deg=beamwidth,
        tilt_deg=0.0,
        orientation_deg=orientation,
        model=model,
        grid_radius_km=5.0,
        grid_resolution=41,
    )
    return {"site_id": site_id, "snapshot": snapshot}


# ── Analysis Loop ─────────────────────────────────────────────────────────────

@app.post("/api/analysis/run")
async def analysis_run(body: AnalysisRunRequest) -> dict:
    """Trigger an immediate AI analysis cycle with the specified model."""
    import asyncio
    import time as _time
    t0 = _time.monotonic()

    samples = read_jsonl(INGEST_LOG_FILE)[-(body.limit_samples):]
    op_lat = body.lat or next((s.get("lat", 0.0) for s in reversed(samples) if s.get("lat")), 0.0)
    op_lon = body.lon or next((s.get("lon", 0.0) for s in reversed(samples) if s.get("lon")), 0.0)

    fcc_context = []
    if op_lat or op_lon:
        try:
            fcc_context = await search_licenses_near(op_lat, op_lon, radius_km=25.0, limit=20)
        except Exception:
            pass

    examples = get_recent_examples(3)
    examples_text = format_examples_for_prompt(examples)
    model_config = {"provider": body.provider, "model": body.model}

    result = detect_signals(samples, model_config, fcc_context, examples_text, op_lat, op_lon)

    speculative_features = []
    if result.detections and (op_lat or op_lon):
        speculative_features = speculate_from_detections(result.detections, op_lat, op_lon)
        for feat in speculative_features:
            try:
                upsert_feature(feat)
            except Exception:
                pass

    ms = (_time.monotonic() - t0) * 1000
    try:
        log_analysis(
            model=body.model,
            provider=body.provider,
            input_summary=f"{len(samples)} samples",
            detections_count=len(result.detections),
            detections_json=json.dumps(
                [d.__dict__ if hasattr(d, '__dict__') else str(d) for d in result.detections],
                default=str,
            ),
            raw_response=(result.raw_response or "")[:2000],
            processing_ms=round(ms, 1),
        )
    except Exception:
        pass

    return {
        "provider": body.provider,
        "model": body.model,
        "samples_analyzed": len(samples),
        "detections": len(result.detections),
        "speculative_features_added": len(speculative_features),
        "summary": result.summary,
        "error": result.error,
        "processing_ms": round(ms, 1),
    }


@app.get("/api/analysis/log")
def analysis_log(limit: int = 20) -> dict:
    """Return recent AI analysis log entries."""
    rows = get_analysis_log(limit)
    return {"count": len(rows), "entries": rows}


@app.get("/api/analysis/uncertain")
def analysis_uncertain(limit: int = 10) -> dict:
    """Return speculative features most in need of user confirmation.

    Ranked by lowest (confidence / analysis_count) ratio — highest-observed,
    lowest-confidence features are most informative to confirm or dismiss.
    """
    features = get_uncertain_features(limit)
    return {"count": len(features), "features": features}


@app.get("/api/analysis/calibration")
def analysis_calibration() -> dict:
    """Return mean / median position error across confirmed features."""
    return get_calibration_stats()


# ── Persistent Map Store ───────────────────────────────────────────────────────

@app.get("/api/map/features")
def map_features(
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    kind: str | None = None,
    limit: int = 500,
) -> dict:
    """Geo-bounded feature query from the persistent map store."""
    return features_in_bounds(lat_min, lat_max, lon_min, lon_max, kind=kind, limit=limit)


@app.post("/api/map/seed")
async def map_seed(lat: float, lon: float, radius_km: float = 50.0) -> dict:
    """Seed the region around (lat, lon) from FCC + cell tower public data."""
    from backend.analysis.us_coverage import seed_tile, _operator_tile, _tile_bounds, _tile_id
    lat_f, lon_f = _operator_tile(lat, lon)
    lat_min, lon_min, lat_max, lon_max = _tile_bounds(lat_f, lon_f)
    tile = {
        "tile_id": _tile_id(lat_f, lon_f),
        "lat_min": lat_min, "lon_min": lon_min,
        "lat_max": lat_max, "lon_max": lon_max,
        "center_lat": lat, "center_lon": lon,
    }
    count = await seed_tile(tile)
    return {"lat": lat, "lon": lon, "radius_km": radius_km, "features_added": count}


@app.get("/api/map/coverage")
def map_coverage() -> dict:
    """Return US CONUS coverage grid as GeoJSON."""
    return get_coverage_grid()


@app.get("/api/usmap/progress")
def usmap_progress() -> dict:
    """Return US mapping progress statistics."""
    return get_coverage_progress()


@app.post("/api/map/confirm")
def map_confirm(body: ConfirmDetectionRequest) -> dict:
    """Confirm or dismiss a speculative detection.

    Confirmed features: promote to 'estimate', store as few-shot example,
    record position error for calibration stats.
    Dismissed features: remove from map store.
    """
    import math as _m
    if body.confirmed:
        current = features_in_bounds(-90, 90, -180, 180, limit=10000)
        feat = next(
            (f for f in current["features"] if f["properties"].get("id") == body.feature_id),
            None,
        )
        position_error_m = None
        if feat and body.true_lat is not None and body.true_lon is not None:
            add_confirmed_example(feat["properties"], body.true_lat, body.true_lon)
            # Compute position error and update calibration stats
            stored_lat = feat["geometry"]["coordinates"][1]
            stored_lon = feat["geometry"]["coordinates"][0]
            R = 6_371_000.0
            dlat = _m.radians(body.true_lat - stored_lat)
            dlon = _m.radians(body.true_lon - stored_lon)
            a = (_m.sin(dlat / 2) ** 2 +
                 _m.cos(_m.radians(stored_lat)) * _m.cos(_m.radians(body.true_lat)) * _m.sin(dlon / 2) ** 2)
            position_error_m = round(2 * R * _m.asin(_m.sqrt(a)), 1)
            try:
                record_position_error(body.feature_id, position_error_m)
            except Exception:
                pass
        update_feature_kind(body.feature_id, "estimate")
        return {
            "feature_id": body.feature_id,
            "action": "confirmed",
            "new_kind": "estimate",
            "position_error_m": position_error_m,
        }
    else:
        update_feature_kind(body.feature_id, "dismissed")
        return {"feature_id": body.feature_id, "action": "dismissed"}


# ── Spectrum Full Summary ──────────────────────────────────────────────────────

@app.get("/api/datasources/spectrum/full")
def spectrum_full_summary() -> dict:
    """Return the complete 0 Hz–THz band plan grouped by ITU designation."""
    summary = full_spectrum_summary()
    total_bands = sum(g["entry_count"] for g in summary)
    return {
        "total_bands": total_bands,
        "itu_groups": len(summary),
        "coverage": "3 Hz (ELF) through 300 GHz (EHF) + THz reference",
        "groups": summary,
    }


# ── EM Field Propagation ───────────────────────────────────────────────────────

@app.get("/api/rf/em_field")
def rf_em_field(lat: float, lon: float, freq_hz: float, radius_km: float = 5.0,
                eirp_dbm: float = 43.0, n_points: int = 9) -> dict:
    """Return a GeoJSON grid of E/H field strengths from an emitter at (lat, lon)."""
    grid = em_field_grid(lat, lon, eirp_dbm, freq_hz, radius_km, n_points)
    return {"lat": lat, "lon": lon, "freq_hz": freq_hz, "radius_km": radius_km,
            "eirp_dbm": eirp_dbm, "geojson": grid}


@app.get("/api/rf/ionospheric_skip")
def rf_ionospheric_skip(freq_mhz: float, solar_flux_index: float = 100.0) -> dict:
    """Calculate HF sky-wave skip distance for a given frequency and solar conditions."""
    return ionospheric_skip_distance_km(freq_mhz, solar_flux_index)


@app.get("/api/rf/atmospheric_absorption")
def rf_atmospheric_absorption(freq_ghz: float, distance_km: float) -> dict:
    """Calculate ITU-R P.676 atmospheric absorption for SHF/EHF signals."""
    return atmospheric_absorption_db(freq_ghz, distance_km)


# ── Satellite Relay Map ────────────────────────────────────────────────────────

@app.get("/api/satellites/positions")
async def satellites_positions(group: str = "geo", limit: int = 50) -> dict:
    """Return current positions of satellites in the named group."""
    from datetime import datetime, timezone as tz
    tle_list = await fetch_tle_group(group, limit=limit)
    now = datetime.now(tz.utc)
    positions = []
    for tle in tle_list:
        pos = tle_to_position(tle["name"], tle["line1"], tle["line2"], now)
        if pos:
            positions.append(pos)
    geojson_features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
            "properties": {"name": p["name"], "alt_km": p["alt_km"], "group": group,
                           "epoch": p["epoch"]},
        }
        for p in positions
    ]
    return {
        "group": group,
        "count": len(positions),
        "geojson": {"type": "FeatureCollection", "features": geojson_features},
    }


@app.get("/api/satellites/ground_track")
async def satellites_ground_track(
    group: str = "goes", sat_name: str = "", hours: float = 6.0, step_min: float = 5.0
) -> dict:
    """Return a ground track for a named satellite."""
    tle_list = await fetch_tle_group(group, limit=100)
    tle = next((t for t in tle_list if sat_name.lower() in t["name"].lower()), None)
    if not tle:
        tle = tle_list[0] if tle_list else None
    if not tle:
        raise HTTPException(status_code=404, detail="Satellite not found")
    track = satellite_ground_track(tle["name"], tle["line1"], tle["line2"], hours, step_min)
    coords = [[p["lon"], p["lat"]] for p in track]
    return {
        "name": tle["name"],
        "hours": hours,
        "points": len(track),
        "geojson": {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"name": tle["name"], "group": group},
        },
    }


@app.get("/api/satellites/footprint")
async def satellites_footprint(group: str = "geo", sat_name: str = "") -> dict:
    """Return the coverage footprint polygon for a satellite."""
    from datetime import datetime, timezone as tz
    tle_list = await fetch_tle_group(group, limit=100)
    tle = next((t for t in tle_list if sat_name.lower() in t["name"].lower()), None)
    if not tle:
        tle = tle_list[0] if tle_list else None
    if not tle:
        raise HTTPException(status_code=404, detail="Satellite not found")
    pos = tle_to_position(tle["name"], tle["line1"], tle["line2"])
    if not pos:
        raise HTTPException(status_code=500, detail="Position calculation failed")
    footprint = satellite_footprint(pos["lat"], pos["lon"], pos["alt_km"])
    return {
        "name": tle["name"],
        "position": pos,
        "geojson": {"type": "Feature", "geometry": footprint,
                    "properties": {"name": tle["name"], "alt_km": pos["alt_km"]}},
    }


@app.post("/api/satellites/uplink_path")
async def satellites_uplink_path(body: UplinkPathRequest) -> dict:
    """Return a great-circle arc from a ground station to a satellite's nadir."""
    from datetime import datetime, timezone as tz
    tle_list = await fetch_tle_group(body.sat_group, limit=100)
    tle = next((t for t in tle_list if body.sat_name.lower() in t["name"].lower()), None)
    if not tle:
        tle = tle_list[0] if tle_list else None
    if not tle:
        raise HTTPException(status_code=404, detail="Satellite not found")
    pos = tle_to_position(tle["name"], tle["line1"], tle["line2"])
    if not pos:
        raise HTTPException(status_code=500, detail="Position calculation failed")
    arc = uplink_downlink_arc(body.ground_lat, body.ground_lon, pos["lat"], pos["lon"])
    return {
        "ground_lat": body.ground_lat,
        "ground_lon": body.ground_lon,
        "satellite": tle["name"],
        "sat_position": pos,
        "arc_type": "uplink_downlink",
        "geojson": {"type": "Feature", "geometry": arc,
                    "properties": {"satellite": tle["name"], "type": "relay_arc"}},
    }


@app.get("/api/satellites/relay_map")
async def satellites_relay_map(groups: str = "gps,goes,geo") -> dict:
    """Return full relay map: satellite positions + footprints + earth station arcs."""
    group_list = [g.strip() for g in groups.split(",") if g.strip()]
    nearby_stations = EARTH_STATIONS[:8]  # limit arcs to avoid enormous response
    relay = await build_relay_map(groups=group_list, earth_stations=nearby_stations)
    return {"groups": group_list, "feature_count": len(relay["features"]), "geojson": relay}


@app.get("/api/satellites/earth_stations")
def earth_stations_list(lat: float | None = None, lon: float | None = None,
                        radius_km: float = 5000.0) -> dict:
    """Return known earth stations, optionally filtered by proximity."""
    if lat is not None and lon is not None:
        stations = get_stations_near(lat, lon, radius_km)
    else:
        stations = EARTH_STATIONS
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
            "properties": {k: v for k, v in s.items() if k not in ("lat", "lon")},
        }
        for s in stations
    ]
    return {"count": len(features),
            "geojson": {"type": "FeatureCollection", "features": features}}


# ── Computed Spectrum ─────────────────────────────────────────────────────────

@app.post("/api/sdr/spectrum/computed")
def sdr_computed_spectrum(body: ComputedSpectrumRequest) -> dict:
    """Generate a mathematically computed PSD for a virtual receiver window."""
    signals = list(body.known_signals)
    if body.use_band_allocations:
        signals = known_signals_for_band(body.center_freq_hz, body.sample_rate_hz) + signals
    psd = computed_psd(
        center_freq_hz=body.center_freq_hz,
        sample_rate_hz=body.sample_rate_hz,
        n_bins=body.n_bins,
        known_signals=signals,
    )
    rssi, snr = psd_to_metrics(psd)
    freq_low = (body.center_freq_hz - body.sample_rate_hz / 2) / 1e6
    freq_high = (body.center_freq_hz + body.sample_rate_hz / 2) / 1e6
    allocations = allocations_in_range(freq_low, freq_high)
    return {
        "provider": body.provider,
        "center_freq_hz": body.center_freq_hz,
        "sample_rate_hz": body.sample_rate_hz,
        "n_bins": body.n_bins,
        "psd_bins_db": psd,
        "rssi_dbm": rssi,
        "snr_db": snr,
        "band_allocations": allocations,
        "computed": True,
    }


# ── Data Sources ──────────────────────────────────────────────────────────────

@app.get("/api/datasources/spectrum/allocations")
def spectrum_allocations(freq_mhz: float, bandwidth_mhz: float = 0.0) -> dict:
    """Return ITU/FCC band plan allocations for a given frequency."""
    if bandwidth_mhz > 0:
        result = allocations_in_range(freq_mhz - bandwidth_mhz / 2, freq_mhz + bandwidth_mhz / 2)
    else:
        result = allocations_for_freq(freq_mhz)
    return {"freq_mhz": freq_mhz, "bandwidth_mhz": bandwidth_mhz, "allocations": result, "count": len(result)}


@app.get("/api/datasources/fcc")
async def fcc_licenses(lat: float, lon: float, radius_km: float = 50.0, limit: int = 100) -> dict:
    """Query FCC ULS for licensed transmitters near (lat, lon)."""
    licenses = await search_licenses_near(lat=lat, lon=lon, radius_km=radius_km, limit=limit)
    return {"lat": lat, "lon": lon, "radius_km": radius_km, "count": len(licenses), "licenses": licenses}


@app.get("/api/datasources/towers")
async def cell_towers(lat: float, lon: float, radius_km: float = 25.0, api_key: str = "") -> dict:
    """Return cell tower data near (lat, lon) from OpenCelliD / Mozilla."""
    towers = await search_towers_near(lat=lat, lon=lon, radius_km=radius_km, api_key=api_key or None)
    return {"lat": lat, "lon": lon, "radius_km": radius_km, "count": len(towers), "towers": towers}


@app.post("/api/datasources/synthesize")
async def synthesize_rf_map(lat: float, lon: float, radius_km: float = 50.0) -> dict:
    """Synthesize a combined RF emitter map from all available data sources.

    Aggregates FCC licenses, cell towers, and spectrum allocation data to
    build a geo-referenced list of known transmitters for the given area.
    """
    fcc_task = search_licenses_near(lat=lat, lon=lon, radius_km=radius_km, limit=200)
    tower_task = search_towers_near(lat=lat, lon=lon, radius_km=radius_km)
    import asyncio
    licenses, towers = await asyncio.gather(fcc_task, tower_task)

    emitters = []
    for lic in licenses:
        if lic.get("lat") and lic.get("lon"):
            emitters.append({
                "type": "fcc_licensed",
                "lat": lic["lat"],
                "lon": lic["lon"],
                "callsign": lic.get("callsign"),
                "service": lic.get("service"),
                "freq_center_mhz": lic.get("freq_center_mhz"),
                "eirp_dbm": lic.get("eirp_dbm"),
                "azimuth_deg": lic.get("azimuth_deg"),
                "source": "fcc_uls",
            })
    for tower in towers:
        if tower.get("lat") and tower.get("lon"):
            emitters.append({
                "type": "cell_tower",
                "lat": tower["lat"],
                "lon": tower["lon"],
                "radio": tower.get("radio"),
                "freq_mhz": tower.get("freq_mhz"),
                "source": tower.get("source"),
            })

    geojson_features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [e["lon"], e["lat"]]},
            "properties": {k: v for k, v in e.items() if k not in ("lat", "lon")},
        }
        for e in emitters
    ]
    return {
        "lat": lat, "lon": lon, "radius_km": radius_km,
        "emitter_count": len(emitters),
        "geojson": {"type": "FeatureCollection", "features": geojson_features},
        "sources": {"fcc_licenses": len(licenses), "cell_towers": len(towers)},
    }


# ── RF Array & Radar Calculations ─────────────────────────────────────────────

@app.post("/api/rf/array/calculate")
def rf_array_calculate(body: ArrayCalculateRequest) -> dict:
    """Compute antenna array beam pattern and key metrics."""
    lam = 3e8 / max(body.frequency_hz, 1.0)
    spacing = body.element_spacing_m if body.element_spacing_m > 0 else lam / 2.0

    if body.array_type == "planar":
        dy = body.dy_m if body.dy_m > 0 else lam / 2.0
        params = PlanarArrayParams(
            n_x=body.n_elements,
            n_y=body.n_y,
            dx_m=spacing,
            dy_m=dy,
            frequency_hz=body.frequency_hz,
            steer_az_deg=body.steer_az_deg,
            steer_el_deg=body.steer_el_deg,
            window=body.window,
        )
        result = planar_array_pattern(params)
    else:
        params_lin = LinearArrayParams(
            n_elements=body.n_elements,
            element_spacing_m=spacing,
            frequency_hz=body.frequency_hz,
            steering_angle_deg=body.steering_angle_deg,
            window=body.window,
        )
        result = linear_array_pattern(params_lin)

    return {
        "array_type": body.array_type,
        "n_elements": body.n_elements,
        "frequency_hz": body.frequency_hz,
        "element_spacing_m": round(spacing, 6),
        "wavelength_m": round(lam, 6),
        "main_beam_deg": result.main_beam_deg,
        "hpbw_deg": result.hpbw_deg,
        "first_sidelobe_db": result.first_sidelobe_db,
        "array_gain_db": result.array_gain_db,
        "pattern": {"angles_deg": result.angles_deg, "pattern_db": result.pattern_db},
        "pattern_2d": result.pattern_2d,
    }


@app.post("/api/rf/radar/estimate")
def rf_radar_estimate(body: RadarEstimateRequest) -> dict:
    """Solve the radar range equation and return detection metrics."""
    rcs = body.rcs_m2
    if rcs is None:
        rcs = estimate_rcs(body.target_type, body.frequency_hz)

    params = RadarParams(
        peak_power_w=body.peak_power_w,
        antenna_gain_dbi=body.antenna_gain_dbi,
        frequency_hz=body.frequency_hz,
        rcs_m2=rcs,
        noise_figure_db=body.noise_figure_db,
        bandwidth_hz=body.bandwidth_hz,
        losses_db=body.losses_db,
    )
    result = radar_range_equation(params)
    return {
        "target_type": body.target_type,
        "rcs_m2": rcs,
        "max_range_km": result.max_range_km,
        "snr_at_range_db": result.snr_at_range_db,
        "min_detectable_rcs_m2": result.min_detectable_rcs_m2,
        "noise_power_dbm": result.noise_power_dbm,
        "frequency_hz": body.frequency_hz,
        "wavelength_m": round(3e8 / body.frequency_hz, 6),
    }


@app.post("/api/rf/link_budget")
def rf_link_budget(body: LinkBudgetRequest) -> dict:
    """Compute a complete RF link budget."""
    result = link_budget(
        tx_power_dbm=body.tx_power_dbm,
        tx_gain_dbi=body.tx_gain_dbi,
        rx_gain_dbi=body.rx_gain_dbi,
        frequency_hz=body.frequency_hz,
        distance_km=body.distance_km,
        rx_noise_figure_db=body.rx_noise_figure_db,
        bandwidth_hz=body.bandwidth_hz,
        losses_db=body.losses_db,
        model=body.model,
    )
    return {
        "received_power_dbm": result.received_power_dbm,
        "path_loss_db": result.path_loss_db,
        "snr_db": result.snr_db,
        "link_margin_db": result.link_margin_db,
        **result.details,
    }


@app.post("/api/rf/bearing")
def rf_bearing_from_phase(
    phase_diff_rad: float,
    element_spacing_m: float,
    frequency_hz: float,
    array_orientation_deg: float = 0.0,
) -> dict:
    """Estimate signal bearing from inter-element phase difference."""
    bearing = bearing_from_phase_difference(
        phase_diff_rad=phase_diff_rad,
        element_spacing_m=element_spacing_m,
        frequency_hz=frequency_hz,
        array_orientation_deg=array_orientation_deg,
    )
    return {
        "bearing_deg": bearing,
        "phase_diff_rad": phase_diff_rad,
        "element_spacing_m": element_spacing_m,
        "frequency_hz": frequency_hz,
    }


# ── AI Analysis ───────────────────────────────────────────────────────────────

@app.post("/api/ai/analyze")
def ai_analyze(body: AIAnalyzeRequest) -> dict:
    """Send spectrum / RF data to an Ollama model for AI-driven analysis.

    The model receives a structured prompt combining the user's context with
    a JSON representation of the spectrum_data payload.
    """
    system_prompt = (
        "You are an expert RF signal analyst. Analyze the provided spectrum data "
        "and identify: signal types, likely emitter sources, bearing estimates, "
        "anomalies, and any potential interference sources. "
        "Provide concise, technically accurate assessments."
    )
    user_content = body.prompt or (
        f"Analyze this spectrum observation:\n```json\n{json.dumps(body.spectrum_data, indent=2)}\n```"
    )
    if body.context:
        user_content = f"Context: {body.context}\n\n{user_content}"

    ollama_body = json.dumps({
        "model": body.model,
        "system": system_prompt,
        "prompt": user_content,
        "stream": False,
    }).encode("utf-8")
    req = request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=ollama_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60.0) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return {"model": body.model, "analysis": result.get("response", ""), "done": result.get("done", False)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {exc}") from exc


# ── SSE event stream ──────────────────────────────────────────────────────────

import asyncio as _asyncio

from fastapi import Request as _Request
from fastapi.responses import StreamingResponse as _StreamingResponse

_app_loop: _asyncio.AbstractEventLoop | None = None


@app.on_event("startup")
async def _capture_loop() -> None:
    global _app_loop
    _app_loop = _asyncio.get_event_loop()
    from backend.foxhunt.auto_loop import event_bus
    event_bus.set_loop(_app_loop)
    # Start SDR ingest service with real KiwiSDR fetcher
    try:
        sdr_service.start()
    except Exception:
        pass
    # Start auto_loop background thread in IDLE state so SSE/event bus is live
    from backend.foxhunt.auto_loop import auto_loop as _auto_loop
    try:
        _auto_loop._start_idle()
    except Exception:
        pass
    # Auto-populate KiwiSDR nodes from public directory (fire-and-forget)
    try:
        _asyncio.create_task(node_pool_auto_populate_loop())
    except Exception:
        pass
    # Multi-band spectrum mosaic loop
    try:
        from backend.sdr.spectrum_mosaic import spectrum_mosaic_loop
        _asyncio.create_task(spectrum_mosaic_loop())
    except Exception:
        pass
    # Start continuous corrector background thread
    try:
        from backend.analysis.continuous_corrector import corrector
        corrector.start()
    except Exception:
        pass


async def node_pool_auto_populate_loop() -> None:
    """Auto-populate KiwiSDR nodes at startup and refresh every 6 hours."""
    import asyncio as _aio
    from backend.sdr.kiwisdr_client import node_pool as _node_pool
    while True:
        try:
            await _node_pool.auto_populate()
        except Exception:
            pass
        await _aio.sleep(6 * 3600)


@app.get("/api/events")
async def sse_events(request: _Request):
    """Server-Sent Events stream for real-time fox hunt and SDR updates."""
    import asyncio as _aio
    q: _aio.Queue = _aio.Queue()
    from backend.foxhunt.auto_loop import event_bus
    event_bus.subscribe(q)

    async def _generate():
        yield "data: {\"type\":\"connected\"}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await _aio.wait_for(q.get(), timeout=25.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except _aio.TimeoutError:
                    yield "data: {\"type\":\"keepalive\"}\n\n"
        finally:
            event_bus.unsubscribe(q)

    return _StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Spectrum mosaic ───────────────────────────────────────────────────────────

@app.get("/api/spectrum/mosaic")
def spectrum_mosaic_latest():
    """Return the latest multi-band spectrum mosaic frame."""
    try:
        from backend.sdr.spectrum_mosaic import spectrum_mosaic
        frame = spectrum_mosaic.get_latest()
        return {
            "ts": frame.ts,
            "segments": [
                {
                    "center_freq_hz": s.center_freq_hz,
                    "bandwidth_hz": s.bandwidth_hz,
                    "bins_db": s.bins_db,
                    "source": s.source,
                    "node": s.node,
                }
                for s in frame.sorted_segments()
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── EM fields ─────────────────────────────────────────────────────────────────

@app.get("/api/em_fields")
def get_em_fields():
    """Return all EM field overlay features as a GeoJSON FeatureCollection."""
    try:
        from backend.storage.map_store import get_all_features
        fc = get_all_features(limit=500)
        em_features = [
            f for f in fc.get("features", [])
            if f.get("properties", {}).get("kind") == "em_field"
        ]
        return {"type": "FeatureCollection", "features": em_features}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── KiwiSDR node management ───────────────────────────────────────────────────

class KiwiNodeRequest(BaseModel):
    host: str
    port: int = 8073
    lat: float | None = None
    lon: float | None = None
    description: str = ""


@app.get("/api/sdr/nodes")
def sdr_nodes_list():
    from backend.sdr.kiwisdr_client import node_pool
    return {"nodes": node_pool.list_nodes()}


@app.post("/api/sdr/nodes")
def sdr_nodes_add(req: KiwiNodeRequest):
    from backend.sdr.kiwisdr_client import node_pool
    node_pool.add_node(req.host, req.port, req.lat, req.lon, req.description)
    return {"ok": True, "nodes": node_pool.list_nodes()}


@app.delete("/api/sdr/nodes/{host}")
def sdr_nodes_remove(host: str, port: int = Query(default=8073)):
    from backend.sdr.kiwisdr_client import node_pool
    node_pool.remove_node(host, port)
    return {"ok": True}


@app.get("/api/sdr/nodes/check")
def sdr_nodes_check():
    from backend.sdr.kiwisdr_client import node_pool
    from dataclasses import asdict as _asdict
    statuses = node_pool.check_all()
    return {"statuses": [_asdict(s) for s in statuses]}


@app.post("/api/sdr/nodes/discover")
async def sdr_nodes_discover():
    """Trigger an immediate refresh from the public KiwiSDR directory."""
    from backend.sdr.kiwisdr_client import node_pool
    before = len(node_pool.list_nodes())
    added = await node_pool.auto_populate()
    total = len(node_pool.list_nodes())
    return {"added": added, "total": total}


@app.get("/api/sdr/scan")
def sdr_scan_peaks(min_snr: float = Query(default=6.0)):
    """Scan all configured KiwiSDR nodes and return signal peaks."""
    from backend.sdr.kiwisdr_client import node_pool
    peaks = node_pool.scan_peaks_all(min_snr_db=min_snr)
    return {"peaks": peaks, "node_count": len(node_pool.list_nodes())}


@app.get("/api/sdr/scan/rssi")
def sdr_scan_rssi(freq_hz: float = Query(...)):
    """Query all nodes for RSSI at a specific frequency."""
    from backend.sdr.kiwisdr_client import node_pool
    readings = node_pool.scan_rssi(freq_hz)
    return {"freq_hz": freq_hz, "readings": readings}


# ── Fox hunt auto loop API ────────────────────────────────────────────────────

class FoxHuntStartRequest(BaseModel):
    lat: float = 0.0
    lon: float = 0.0
    scan_interval_s: float = 15.0
    min_obs_to_solve: int = 2


class FoxBearingRequest(BaseModel):
    bearing_deg: float
    snr_db: float = 10.0
    freq_hz: float | None = None
    lat: float | None = None   # override operator position
    lon: float | None = None


class FoxMultilatRequest(BaseModel):
    rssi_obs: list[dict] = []
    tdoa_obs: list[dict] = []
    bearing_obs: list[dict] = []
    freq_hz: float = 100e6


@app.post("/api/foxhunt/auto/start")
def foxhunt_auto_start(req: FoxHuntStartRequest):
    from backend.foxhunt.auto_loop import auto_loop
    auto_loop.scan_interval_s = req.scan_interval_s
    auto_loop.min_obs_to_solve = req.min_obs_to_solve
    auto_loop.start(req.lat, req.lon)
    return {"ok": True, "state": auto_loop.status()}


@app.post("/api/foxhunt/auto/stop")
def foxhunt_auto_stop():
    from backend.foxhunt.auto_loop import auto_loop
    auto_loop.stop()
    return {"ok": True}


@app.get("/api/foxhunt/auto/status")
def foxhunt_auto_status():
    from backend.foxhunt.auto_loop import auto_loop
    return auto_loop.status()


@app.post("/api/foxhunt/auto/observe")
def foxhunt_add_bearing(req: FoxBearingRequest):
    from backend.foxhunt.auto_loop import auto_loop
    if req.lat is not None and req.lon is not None:
        auto_loop.update_operator_position(req.lat, req.lon)
    result = auto_loop.add_bearing_observation(req.bearing_deg, req.snr_db, req.freq_hz)
    return result


@app.post("/api/foxhunt/auto/position")
def foxhunt_update_position(lat: float = Query(...), lon: float = Query(...)):
    from backend.foxhunt.auto_loop import auto_loop
    auto_loop.update_operator_position(lat, lon)
    return {"ok": True, "lat": lat, "lon": lon}


@app.get("/api/foxhunt/auto/confirmed")
def foxhunt_confirmed_features():
    from backend.foxhunt.auto_loop import auto_loop
    features = auto_loop.confirmed_features()
    return {"type": "FeatureCollection", "features": features}


@app.post("/api/foxhunt/triangulate")
def foxhunt_triangulate(req: FoxMultilatRequest):
    """One-shot triangulation endpoint — supply observations, get a fix.

    Each dict in rssi_obs:    {lat, lon, rssi_dbm, freq_hz, eirp_dbm?, weight?}
    Each dict in bearing_obs: {lat, lon, bearing_deg, snr_db, freq_hz, sigma_deg?}
    Each dict in tdoa_obs:    {lat_ref, lon_ref, lat_remote, lon_remote, tdoa_s, freq_hz, weight?}
    """
    from backend.foxhunt.multilateration import (
        RSSIObs, BearingObs, TDOAObs, locate_transmitter, ellipse_polygon
    )
    rssi = [RSSIObs(**{k: v for k, v in o.items() if k in
                       ("lat", "lon", "rssi_dbm", "freq_hz", "eirp_dbm", "weight")})
            for o in req.rssi_obs if o.get("lat") and o.get("lon")]
    bearings = [BearingObs(**{k: v for k, v in o.items() if k in
                               ("lat", "lon", "bearing_deg", "snr_db", "freq_hz", "sigma_deg")})
                for o in req.bearing_obs if o.get("lat") and o.get("lon")]
    tdoa = [TDOAObs(**{k: v for k, v in o.items() if k in
                        ("lat_ref", "lon_ref", "lat_remote", "lon_remote", "tdoa_s", "freq_hz", "weight")})
            for o in req.tdoa_obs]

    fix = locate_transmitter(rssi_obs=rssi, bearing_obs=bearings, tdoa_obs=tdoa, freq_hz=req.freq_hz)
    ellipse = ellipse_polygon(fix.lat, fix.lon, max(fix.ellipse_major_m, 50), max(fix.ellipse_minor_m, 30))
    return {
        "lat": fix.lat, "lon": fix.lon,
        "uncertainty_m": fix.uncertainty_m,
        "confidence": fix.confidence,
        "methods": fix.methods_used,
        "ellipse_polygon": ellipse,
        "per_method": [
            {"method": e.method, "lat": e.lat, "lon": e.lon,
             "uncertainty_m": e.uncertainty_m, "confidence": e.confidence, "notes": e.notes}
            for e in fix.per_method
        ],
    }


# ── Bayesian field model ──────────────────────────────────────────────────────

class BayesFieldUpdateRequest(BaseModel):
    center_lat: float
    center_lon: float
    radius_km: float = 50.0
    resolution_km: float = 2.0
    rssi_obs: list[dict] = []    # [{obs_lat,obs_lon,rssi_dbm,freq_hz,eirp_dbm?}]
    bearing_obs: list[dict] = [] # [{obs_lat,obs_lon,bearing_deg,sigma_deg?}]
    reset: bool = False


@app.post("/api/bayes/update")
def bayes_field_update(body: BayesFieldUpdateRequest) -> dict:
    """Feed observations into the Bayesian posterior grid and return the heatmap."""
    from backend.analysis.bayes_field import (
        BayesGrid, RSSIUpdate, BearingUpdate, get_or_create_grid, reset_grid,
    )
    if body.reset:
        grid = reset_grid(body.center_lat, body.center_lon, body.radius_km, body.resolution_km)
    else:
        grid = get_or_create_grid(body.center_lat, body.center_lon, body.radius_km, body.resolution_km)

    for o in body.rssi_obs:
        try:
            grid.update_rssi(RSSIUpdate(
                obs_lat=float(o["obs_lat"]), obs_lon=float(o["obs_lon"]),
                rssi_dbm=float(o["rssi_dbm"]), freq_hz=float(o["freq_hz"]),
                eirp_dbm=float(o.get("eirp_dbm", 47.0)),
            ))
        except (KeyError, ValueError):
            pass

    for o in body.bearing_obs:
        try:
            grid.update_bearing(BearingUpdate(
                obs_lat=float(o["obs_lat"]), obs_lon=float(o["obs_lon"]),
                bearing_deg=float(o["bearing_deg"]),
                sigma_deg=float(o.get("sigma_deg", 10.0)),
            ))
        except (KeyError, ValueError):
            pass

    geojson = grid.posterior_geojson(top_k=400)
    return {"geojson": geojson, "observation_count": grid._observation_count}


@app.get("/api/bayes/field")
def bayes_field_query(
    center_lat: float,
    center_lon: float,
    radius_km: float = 50.0,
    resolution_km: float = 2.0,
) -> dict:
    """Return the current posterior heatmap for an existing grid."""
    from backend.analysis.bayes_field import get_or_create_grid
    grid = get_or_create_grid(center_lat, center_lon, radius_km, resolution_km)
    geojson = grid.posterior_geojson(top_k=400)
    return {"geojson": geojson, "observation_count": grid._observation_count}


@app.delete("/api/bayes/field")
def bayes_field_reset(center_lat: float, center_lon: float,
                      radius_km: float = 50.0, resolution_km: float = 2.0) -> dict:
    """Reset the posterior to a flat prior."""
    from backend.analysis.bayes_field import reset_grid
    reset_grid(center_lat, center_lon, radius_km, resolution_km)
    return {"ok": True}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
