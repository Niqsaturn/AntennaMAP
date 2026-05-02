from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TelemetrySample:
    timestamp: str
    lat: float
    lon: float
    heading_deg: float
    bearing_estimate_deg: float
    rssi_dbm: float
    snr_db: float
    frequency_hz: float
    bandwidth_hz: float
    terrain: dict[str, float] | None = None


@dataclass
class ModelMetadata:
    model_version: str
    trained_at: str
    feature_schema: list[str]
    sample_count: int
    notes: str = "linear-regression error calibrator"


@dataclass
class TrainResult:
    metadata: ModelMetadata
    weights: list[float]
    bias: float
    rmse: float


@dataclass
class InferenceResult:
    center_lat: float
    center_lon: float
    covariance: list[list[float]]
    ellipse_major_m: float
    ellipse_minor_m: float
    ellipse_angle_deg: float
    quality_score: float
    predicted_error_m: float | None = None


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "backend" / "ml" / "models"
TRAINING_DIR = ROOT / "backend" / "ml" / "training_data"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

def wrap_angle_deg(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0

def meters_per_deg_lat() -> float:
    return 111_320.0

def meters_per_deg_lon(lat_deg: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat_deg))

def windowed_deltas(samples: list[TelemetrySample], window: int = 3) -> list[dict[str, float]]:
    out = []
    for i in range(len(samples)):
        start = max(0, i - window)
        sub = samples[start : i + 1]
        if len(sub) < 2:
            out.append({"delta_heading": 0.0, "delta_bearing": 0.0, "delta_snr": 0.0})
            continue
        out.append({
            "delta_heading": wrap_angle_deg(sub[-1].heading_deg - sub[0].heading_deg),
            "delta_bearing": wrap_angle_deg(sub[-1].bearing_estimate_deg - sub[0].bearing_estimate_deg),
            "delta_snr": sub[-1].snr_db - sub[0].snr_db,
        })
    return out

def uncertainty_weight(sample: TelemetrySample) -> float:
    return (max(0.1, min(30.0, sample.snr_db)) / 30.0) * max(0.1, (120.0 + sample.rssi_dbm) / 80.0) * (1.0 / max(1.0, sample.bandwidth_hz / 1e6))

def movement_geometry_quality(samples: list[TelemetrySample]) -> float:
    if len(samples) < 3:
        return 0.0
    spreads = [abs(wrap_angle_deg(samples[i].heading_deg - samples[i - 1].heading_deg)) for i in range(1, len(samples))]
    heading_diversity = min(1.0, (sum(spreads) / len(spreads)) / 45.0)
    lats = [s.lat for s in samples]
    lons = [s.lon for s in samples]
    extent = (max(lats) - min(lats)) * meters_per_deg_lat() + (max(lons) - min(lons)) * meters_per_deg_lon(sum(lats) / len(lats))
    return round(0.5 * heading_diversity + 0.5 * min(1.0, extent / 150.0), 4)

def _line_intersection_from_bearings(a: TelemetrySample, b: TelemetrySample) -> tuple[float, float] | None:
    lat0 = (a.lat + b.lat) / 2.0
    bx = (b.lon - a.lon) * meters_per_deg_lon(lat0)
    by = (b.lat - a.lat) * meters_per_deg_lat()
    va = np.array([math.sin(math.radians(a.bearing_estimate_deg)), math.cos(math.radians(a.bearing_estimate_deg))])
    vb = np.array([math.sin(math.radians(b.bearing_estimate_deg)), math.cos(math.radians(b.bearing_estimate_deg))])
    m = np.array([[va[0], -vb[0]], [va[1], -vb[1]]])
    if abs(np.linalg.det(m)) < 1e-6:
        return None
    t = np.linalg.solve(m, np.array([bx, by]))[0]
    x, y = va * t
    return (a.lat + y / meters_per_deg_lat(), a.lon + x / meters_per_deg_lon(lat0))

def bearing_intersections(samples: list[TelemetrySample]) -> list[tuple[float, float]]:
    points = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            p = _line_intersection_from_bearings(samples[i], samples[j])
            if p:
                points.append(p)
    return points

def solve_weighted_least_squares(samples: list[TelemetrySample]) -> InferenceResult:
    inter = bearing_intersections(samples) or [(samples[0].lat, samples[0].lon)]
    raw_weights = [uncertainty_weight(s) for s in samples]

    # Build per-intersection weights: geometric mean of the two contributing sample weights
    pair_weights: list[float] = []
    n = len(samples)
    for i in range(n):
        for j in range(i + 1, n):
            p = _line_intersection_from_bearings(samples[i], samples[j])
            if p:
                pair_weights.append(math.sqrt(raw_weights[i] * raw_weights[j]))
    if not pair_weights:
        pair_weights = [1.0]
    w = np.array(pair_weights[: len(inter)])

    arr = np.array(inter)
    center = np.average(arr, axis=0, weights=w)
    mean_lat_rad = math.radians(float(center[0]))

    # Covariance requires ≥2 distinct points; fall back to identity when degenerate
    if len(inter) >= 2:
        try:
            cov = np.cov((arr - center).T, aweights=w) + np.eye(2) * 1e-10
            if not np.all(np.isfinite(cov)):
                raise ValueError("non-finite covariance")
        except Exception:
            cov = np.eye(2) * 1e-10
    else:
        cov = np.eye(2) * 1e-10

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # Convert covariance axes to metres with correct cos(lat) scaling on lon
    m_per_deg_lat = meters_per_deg_lat()
    m_per_deg_lon = meters_per_deg_lon(float(center[0]))
    ellipse_major_m = float(math.sqrt(max(eigvals[0], 0.0)) * m_per_deg_lat * 2)
    ellipse_minor_m = float(math.sqrt(max(eigvals[1], 0.0)) * m_per_deg_lon * 2)

    return InferenceResult(
        center_lat=float(center[0]),
        center_lon=float(center[1]),
        covariance=cov.tolist(),
        ellipse_major_m=ellipse_major_m,
        ellipse_minor_m=ellipse_minor_m,
        ellipse_angle_deg=float(math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))),
        quality_score=movement_geometry_quality(samples),
    )

def build_feature_matrix(samples: list[TelemetrySample]) -> tuple[np.ndarray, list[str]]:
    delta = windowed_deltas(samples)
    density = len(bearing_intersections(samples)) / max(1, len(samples))
    schema = ["heading_deg", "bearing_estimate_deg", "rssi_dbm", "snr_db", "frequency_hz", "bandwidth_hz", "delta_heading", "delta_bearing", "delta_snr", "uncertainty_weight", "intersection_density", "terrain_elevation_m"]
    rows = []
    for i, s in enumerate(samples):
        rows.append([s.heading_deg, s.bearing_estimate_deg, s.rssi_dbm, s.snr_db, s.frequency_hz, s.bandwidth_hz, delta[i]["delta_heading"], delta[i]["delta_bearing"], delta[i]["delta_snr"], uncertainty_weight(s), density, (s.terrain or {}).get("elevation_m", 0.0)])
    return np.array(rows, dtype=float), schema

def train_error_model(samples: list[TelemetrySample], target_error_m: list[float], model_version: str) -> TrainResult:
    X, schema = build_feature_matrix(samples)
    y = np.array(target_error_m, dtype=float)
    x_mean, x_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    Xd = np.column_stack([((X - x_mean) / x_std), np.ones(len(X))])
    w = np.linalg.pinv(Xd.T @ Xd) @ Xd.T @ y
    pred = Xd @ w
    meta = ModelMetadata(model_version=model_version, trained_at=datetime.now(timezone.utc).isoformat(), feature_schema=schema, sample_count=len(samples))
    artifact = {"metadata": asdict(meta), "weights": w[:-1].tolist(), "bias": float(w[-1]), "x_mean": x_mean.tolist(), "x_std": x_std.tolist(), "rmse": float(np.sqrt(np.mean((pred - y) ** 2)))}
    (MODEL_DIR / f"triangulation_{model_version}.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return TrainResult(metadata=meta, weights=w[:-1].tolist(), bias=float(w[-1]), rmse=artifact["rmse"])

def load_latest_model() -> dict[str, Any] | None:
    files = sorted(MODEL_DIR.glob("triangulation_*.json"))
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None

def predict_error(model: dict[str, Any], samples: list[TelemetrySample]) -> float:
    X, _ = build_feature_matrix(samples)
    x = (X.mean(axis=0) - np.array(model["x_mean"])) / np.array(model["x_std"])
    return float(np.dot(x, np.array(model["weights"])) + model["bias"])

def parse_samples(payload: list[dict[str, Any]]) -> list[TelemetrySample]:
    """Parse dicts into TelemetrySample objects with defensive defaults."""
    import logging as _logging
    _logger = _logging.getLogger(__name__)
    samples = []
    for row in payload:
        try:
            # Accept both "bearing_estimate_deg" (canonical) and legacy "bearing_deg"
            bearing = float(
                row.get("bearing_estimate_deg") if row.get("bearing_estimate_deg") is not None
                else row.get("bearing_deg", 0.0)
            )
            samples.append(TelemetrySample(
                timestamp=row.get("timestamp", ""),
                lat=float(row.get("lat", 0.0)),
                lon=float(row.get("lon", 0.0)),
                heading_deg=float(row.get("heading_deg", bearing)),
                bearing_estimate_deg=bearing,
                rssi_dbm=float(row.get("rssi_dbm", -100.0)),
                snr_db=float(row.get("snr_db", 10.0)),
                frequency_hz=float(row.get("frequency_hz", 100e6)),
                bandwidth_hz=float(row.get("bandwidth_hz", 3000.0)),
            ))
        except (KeyError, ValueError, TypeError) as e:
            _logger.warning("skipping malformed sample: %s", e)
            continue
    return samples


def estimate_bearing_uncertainty(
    sample: TelemetrySample,
    samples: list[TelemetrySample] | None = None,
    sdr_type: str = "rtlsdr",
) -> float:
    """Return calibrated bearing uncertainty (σ_deg) for a triangulation sample.

    Delegates to bearing_tracker.bearing_uncertainty_deg() with geometry quality
    derived from the full sample set, giving the triangulation WLS weighting
    physically-calibrated σ values instead of a raw SNR proxy.
    """
    from backend.analysis.bearing_tracker import bearing_uncertainty_deg
    geom_quality = movement_geometry_quality(samples) if samples and len(samples) >= 3 else 0.5
    return bearing_uncertainty_deg(
        snr_db=sample.snr_db,
        freq_mhz=sample.frequency_hz / 1e6,
        bandwidth_hz=sample.bandwidth_hz,
        sdr_type=sdr_type,
        movement_geometry=geom_quality,
    )
