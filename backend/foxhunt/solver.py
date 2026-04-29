from __future__ import annotations

import math
import numpy as np
from .models import FoxHuntObservation, SolverEstimate

METERS_PER_DEG = 111_320.0


def _to_local_xy(observations: list[FoxHuntObservation]) -> tuple[np.ndarray, float, float]:
    lat0 = float(np.mean([o.lat for o in observations]))
    lon0 = float(np.mean([o.lon for o in observations]))
    cos_lat = math.cos(math.radians(lat0))
    rows = []
    for o in observations:
        x = (o.lon - lon0) * METERS_PER_DEG * cos_lat
        y = (o.lat - lat0) * METERS_PER_DEG
        rows.append([x, y])
    return np.array(rows), lat0, lon0


def _bearing_of(obs: FoxHuntObservation) -> float:
    return float(obs.antenna_bearing_deg if obs.antenna_bearing_deg is not None else obs.heading_deg)


def solve(observations: list[FoxHuntObservation]) -> SolverEstimate | None:
    if len(observations) < 2:
        return None

    xy, lat0, lon0 = _to_local_xy(observations)
    A, b, w = [], [], []
    bearings = []
    snrs = np.array([max(o.snr_db, 0.1) for o in observations])
    rssis = np.array([o.rssi_dbm for o in observations])
    sig_consistency = 1.0 / (1.0 + np.std(snrs) / 10.0 + np.std(rssis) / 20.0)

    for i, o in enumerate(observations):
        br = _bearing_of(o)
        bearings.append(br)
        theta = math.radians(br)
        d = np.array([math.sin(theta), math.cos(theta)])
        n = np.array([d[1], -d[0]])
        A.append(n)
        b.append(float(np.dot(n, xy[i])))
        w.append(1.0 + o.snr_db / 20.0)

    A = np.array(A)
    b = np.array(b)
    W = np.diag(np.clip(np.array(w), 0.2, 5.0))

    lhs = A.T @ W @ A + np.eye(2) * 1e-6
    rhs = A.T @ W @ b
    center_xy = np.linalg.solve(lhs, rhs)
    cov = np.linalg.inv(lhs)

    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    major = float(max(5.0, math.sqrt(max(evals[0], 1e-6)) * 2.5))
    minor = float(max(3.0, math.sqrt(max(evals[1], 1e-6)) * 2.5))
    orient = (math.degrees(math.atan2(evecs[1, 0], evecs[0, 0])) + 360.0) % 360.0

    spread = np.std(np.unwrap(np.deg2rad(bearings)))
    diversity = min(1.0, float(spread / (math.pi / 2)))
    shape_quality = minor / major
    confidence = float(np.clip(0.55 * diversity + 0.3 * sig_consistency + 0.15 * shape_quality, 0.0, 1.0))

    lat = lat0 + center_xy[1] / METERS_PER_DEG
    lon = lon0 + center_xy[0] / (METERS_PER_DEG * math.cos(math.radians(lat0)))

    return SolverEstimate(
        center_lat=lat,
        center_lon=lon,
        confidence_score=confidence,
        uncertainty_major_m=major,
        uncertainty_minor_m=minor,
        uncertainty_heading_deg=orient,
    )
