"""Automated bearing estimation from multi-node RSSI differential.

Eliminates the manual bearing-input requirement in the fox hunt state machine
by deriving bearing from RSSI gradients across geographically distributed
SDR nodes.

Methods (in priority order):
  1. Phase-difference bearing  — when ≥2 coherent nodes with known spacing
  2. RSSI gradient bearing     — when ≥3 nodes with GPS and RSSI (WLS gradient)
  3. Single-node bearing       — when only 1 node available (low confidence)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_DEG_PER_M_LAT = 1.0 / 111_320.0


@dataclass
class AutoBearingResult:
    bearing_deg: float
    sigma_deg: float       # 1-sigma uncertainty
    method: str
    node_count: int
    confidence: float      # 0–1


def compute_auto_bearing(
    node_obs: list[dict],
    freq_hz: float,
    phase_diffs: list[dict] | None = None,
) -> AutoBearingResult:
    """Derive best-available bearing estimate from multi-node observations.

    node_obs: list of {"lat", "lon", "rssi_dbm", "node_id"?}
    phase_diffs: optional list of {"phase_diff_rad", "element_spacing_m",
                                   "array_orientation_deg"}
    """
    if phase_diffs and len(phase_diffs) >= 1:
        result = _phase_diff_bearing(phase_diffs, freq_hz)
        if result is not None:
            return result

    valid = [
        o for o in node_obs
        if o.get("lat") and o.get("lon") and o.get("rssi_dbm") is not None
    ]

    if len(valid) >= 3:
        return _rssi_gradient_bearing(valid)
    if len(valid) >= 2:
        return _two_node_bearing(valid)

    return AutoBearingResult(
        bearing_deg=0.0,
        sigma_deg=90.0,
        method="insufficient_nodes",
        node_count=len(valid),
        confidence=0.0,
    )


def _rssi_gradient_bearing(nodes: list[dict]) -> AutoBearingResult:
    """Estimate bearing using weighted RSSI gradient across ≥3 nodes.

    The bearing toward the signal source is approximately in the direction
    of the steepest RSSI increase (gradient of the RSSI field).
    """
    # Convert RSSI (dBm) to linear power for gradient weighting
    lats = [float(n["lat"]) for n in nodes]
    lons = [float(n["lon"]) for n in nodes]
    rssi_lin = [10 ** (float(n["rssi_dbm"]) / 10.0) for n in nodes]
    total = sum(rssi_lin)
    if total == 0:
        return AutoBearingResult(0.0, 90.0, "rssi_gradient_zero", len(nodes), 0.0)

    # Signal-power-weighted centroid = approximate direction of strongest signal
    w_lat = sum(rssi_lin[i] * lats[i] for i in range(len(nodes))) / total
    w_lon = sum(rssi_lin[i] * lons[i] for i in range(len(nodes))) / total

    # Geometric centroid of nodes
    c_lat = sum(lats) / len(lats)
    c_lon = sum(lons) / len(lons)

    # Vector from geometric centroid to power centroid
    dlat_m = (w_lat - c_lat) / _DEG_PER_M_LAT
    cos_lat = math.cos(math.radians(c_lat))
    dlon_m = (w_lon - c_lon) / (_DEG_PER_M_LAT / max(cos_lat, 1e-6))

    bearing_rad = math.atan2(dlon_m, dlat_m)
    bearing_deg = (math.degrees(bearing_rad)) % 360.0

    # Spread of RSSI across nodes → confidence proxy
    rssi_dbm = [float(n["rssi_dbm"]) for n in nodes]
    rssi_range = max(rssi_dbm) - min(rssi_dbm)
    confidence = min(0.7, rssi_range / 20.0)
    sigma_deg = max(15.0, 40.0 - rssi_range * 1.5)

    return AutoBearingResult(
        bearing_deg=round(bearing_deg, 1),
        sigma_deg=round(sigma_deg, 1),
        method="rssi_gradient",
        node_count=len(nodes),
        confidence=round(confidence, 3),
    )


def _two_node_bearing(nodes: list[dict]) -> AutoBearingResult:
    """Coarse bearing from 2 nodes: bearing toward higher-RSSI node."""
    a, b = nodes[0], nodes[1]
    if float(a["rssi_dbm"]) < float(b["rssi_dbm"]):
        a, b = b, a
    dlat = float(a["lat"]) - float(b["lat"])
    dlon = float(a["lon"]) - float(b["lon"])
    bearing_deg = (math.degrees(math.atan2(dlon, dlat))) % 360.0
    return AutoBearingResult(
        bearing_deg=round(bearing_deg, 1),
        sigma_deg=45.0,
        method="two_node_rssi",
        node_count=2,
        confidence=0.2,
    )


def _phase_diff_bearing(
    phase_diffs: list[dict],
    freq_hz: float,
) -> AutoBearingResult | None:
    """Derive bearing from inter-element phase differences (coherent array)."""
    try:
        from backend.rf.array_calculator import bearing_from_phase_difference
        bearings = []
        for pd in phase_diffs:
            b = bearing_from_phase_difference(
                phase_diff_rad=float(pd["phase_diff_rad"]),
                element_spacing_m=float(pd["element_spacing_m"]),
                frequency_hz=freq_hz,
                array_orientation_deg=float(pd.get("array_orientation_deg", 0.0)),
            )
            bearings.append(b)
        if not bearings:
            return None
        avg_bearing = sum(bearings) / len(bearings)
        sigma = 10.0 if len(bearings) > 1 else 20.0
        return AutoBearingResult(
            bearing_deg=round(avg_bearing % 360.0, 1),
            sigma_deg=sigma,
            method="phase_difference",
            node_count=len(bearings),
            confidence=min(0.85, 0.6 + len(bearings) * 0.1),
        )
    except Exception as exc:
        logger.debug("_phase_diff_bearing: %s", exc)
        return None
