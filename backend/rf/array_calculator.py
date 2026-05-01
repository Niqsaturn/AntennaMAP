"""Antenna array and radar calculation engine.

Provides:
  - Linear and planar phased array beam patterns (array factor)
  - Half-power beamwidth, sidelobe level, array gain
  - Radar cross section (RCS) estimation
  - Radar detection range (radar range equation)
  - Link budget calculator
  - Bearing estimation from array phase measurements

All calculations are purely mathematical — no hardware required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LinearArrayParams:
    n_elements: int          # number of array elements
    element_spacing_m: float # spacing between elements (λ/2 typical)
    frequency_hz: float      # operating frequency
    steering_angle_deg: float = 0.0   # main beam steering (degrees from broadside)
    window: str = "uniform"  # "uniform", "hanning", "hamming", "blackman"


@dataclass
class PlanarArrayParams:
    n_x: int                 # elements along X axis
    n_y: int                 # elements along Y axis
    dx_m: float              # X spacing
    dy_m: float              # Y spacing
    frequency_hz: float
    steer_az_deg: float = 0.0
    steer_el_deg: float = 0.0
    window: str = "uniform"


@dataclass
class RadarParams:
    peak_power_w: float
    antenna_gain_dbi: float
    frequency_hz: float
    rcs_m2: float            # radar cross section of target
    noise_figure_db: float = 5.0
    bandwidth_hz: float = 1e6
    losses_db: float = 3.0   # system losses


@dataclass
class ArrayPatternResult:
    angles_deg: list[float]
    pattern_db: list[float]
    main_beam_deg: float
    hpbw_deg: float          # half-power beamwidth
    first_sidelobe_db: float
    array_gain_db: float
    pattern_2d: list[list[float]] | None = None  # for planar arrays


@dataclass
class RadarResult:
    max_range_km: float
    snr_at_range_db: dict[str, float]   # {range_km_str: snr_db}
    min_detectable_rcs_m2: float        # at given range
    noise_power_dbm: float


@dataclass
class LinkBudgetResult:
    received_power_dbm: float
    path_loss_db: float
    snr_db: float
    link_margin_db: float
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def _window_weights(n: int, window: str) -> np.ndarray:
    if window == "hanning":
        return np.hanning(n)
    if window == "hamming":
        return np.hamming(n)
    if window == "blackman":
        return np.blackman(n)
    return np.ones(n)


# ---------------------------------------------------------------------------
# Linear array
# ---------------------------------------------------------------------------

def linear_array_pattern(params: LinearArrayParams, n_points: int = 360) -> ArrayPatternResult:
    """Compute array factor for a linear array over -90° to +90°."""
    lam = 3e8 / params.frequency_hz
    k = 2.0 * math.pi / lam
    d = params.element_spacing_m
    theta_s = math.radians(params.steering_angle_deg)

    weights = _window_weights(params.n_elements, params.window)
    angles = np.linspace(-90.0, 90.0, n_points)
    af_mag = np.zeros(n_points)

    for idx, ang_deg in enumerate(angles):
        theta = math.radians(ang_deg)
        psi = k * d * (math.sin(theta) - math.sin(theta_s))
        af = sum(w * np.exp(1j * n * psi) for n, w in enumerate(weights))
        af_mag[idx] = abs(af)

    af_max = max(af_mag.max(), 1e-20)
    af_norm = af_mag / af_max
    pattern_db = 20.0 * np.log10(np.maximum(af_norm, 1e-10))

    # Array gain with taper efficiency: gain = 10·log10(N · |Σw|²/(N·Σ|w|²))
    w_sum_sq = abs(sum(weights)) ** 2
    w_sq_sum = float(np.sum(np.abs(weights) ** 2))
    taper_efficiency = w_sum_sq / (params.n_elements * w_sq_sum) if w_sq_sum > 0 else 1.0
    array_gain_db = 10.0 * math.log10(params.n_elements * taper_efficiency)

    # Find main beam
    peak_idx = int(np.argmax(pattern_db))
    main_beam_deg = float(angles[peak_idx])

    # HPBW: search only within ±60° of steering angle to stay in main lobe
    steer_idx = int(np.argmin(np.abs(angles - params.steering_angle_deg)))
    half_win = int(60.0 / (180.0 / n_points))
    lo_idx = max(0, steer_idx - half_win)
    hi_idx = min(n_points - 1, steer_idx + half_win)
    above_3db = np.where(pattern_db[lo_idx:hi_idx + 1] >= -3.0)[0]
    if len(above_3db) >= 2:
        hpbw = float(angles[lo_idx + above_3db[-1]] - angles[lo_idx + above_3db[0]])
    else:
        hpbw = float(180.0 / params.n_elements)

    # First sidelobe: highest peak outside main beam region
    main_width = max(1, int(n_points * 0.05))
    sidelobe_region = list(range(0, max(0, peak_idx - main_width))) + \
                      list(range(min(n_points, peak_idx + main_width), n_points))
    first_sll = float(np.max(pattern_db[sidelobe_region])) if sidelobe_region else -13.0

    return ArrayPatternResult(
        angles_deg=angles.tolist(),
        pattern_db=pattern_db.tolist(),
        main_beam_deg=main_beam_deg,
        hpbw_deg=round(hpbw, 2),
        first_sidelobe_db=round(first_sll, 2),
        array_gain_db=round(array_gain_db, 2),
    )


# ---------------------------------------------------------------------------
# Planar array
# ---------------------------------------------------------------------------

def planar_array_pattern(
    params: PlanarArrayParams,
    n_az: int = 180,
    n_el: int = 90,
) -> ArrayPatternResult:
    """Compute 2D array factor for a planar array (azimuth × elevation)."""
    lam = 3e8 / params.frequency_hz
    k = 2.0 * math.pi / lam

    wx = _window_weights(params.n_x, params.window)
    wy = _window_weights(params.n_y, params.window)
    az_angles = np.linspace(-180.0, 180.0, n_az)
    el_angles = np.linspace(-90.0, 90.0, n_el)

    cos_el_s = math.cos(math.radians(params.steer_el_deg))
    sin_az_s = math.sin(math.radians(params.steer_az_deg)) * cos_el_s
    sin_el_s = math.sin(math.radians(params.steer_el_deg))

    pattern_2d = np.zeros((n_el, n_az))

    for ei, el_deg in enumerate(el_angles):
        cos_el = math.cos(math.radians(el_deg))
        sin_el = math.sin(math.radians(el_deg))
        psi_y = k * params.dy_m * (sin_el - sin_el_s)
        afy = sum(wy[m] * np.exp(1j * m * psi_y) for m in range(params.n_y))
        for ai, az_deg in enumerate(az_angles):
            sin_az = math.sin(math.radians(az_deg)) * cos_el
            psi_x = k * params.dx_m * (sin_az - sin_az_s)
            afx = sum(wx[n] * np.exp(1j * n * psi_x) for n in range(params.n_x))
            pattern_2d[ei, ai] = abs(afx * afy)

    p_max = max(pattern_2d.max(), 1e-20)
    pattern_2d_db = 20.0 * np.log10(np.maximum(pattern_2d / p_max, 1e-10))

    # 1D cut at steering elevation for HPBW and sidelobe
    el_idx = int(np.argmin(np.abs(el_angles - params.steer_el_deg)))
    az_cut_db = pattern_2d_db[el_idx, :]
    peak_idx = int(np.argmax(az_cut_db))
    above_3db = np.where(az_cut_db >= -3.0)[0]
    hpbw = float(az_angles[above_3db[-1]] - az_angles[above_3db[0]]) if len(above_3db) >= 2 else 10.0
    main_beam_deg = float(az_angles[peak_idx])

    main_w = max(1, int(n_az * 0.04))
    side = list(range(0, max(0, peak_idx - main_w))) + \
           list(range(min(n_az, peak_idx + main_w), n_az))
    first_sll = float(np.max(az_cut_db[side])) if side else -13.0
    n_total = params.n_x * params.n_y
    wx_sum_sq = abs(float(np.sum(wx))) ** 2
    wy_sum_sq = abs(float(np.sum(wy))) ** 2
    wx_sq_sum = float(np.sum(np.abs(wx) ** 2))
    wy_sq_sum = float(np.sum(np.abs(wy) ** 2))
    taper_eff_x = wx_sum_sq / (params.n_x * wx_sq_sum) if wx_sq_sum > 0 else 1.0
    taper_eff_y = wy_sum_sq / (params.n_y * wy_sq_sum) if wy_sq_sum > 0 else 1.0
    array_gain_db = 10.0 * math.log10(n_total * taper_eff_x * taper_eff_y)

    return ArrayPatternResult(
        angles_deg=az_angles.tolist(),
        pattern_db=az_cut_db.tolist(),
        main_beam_deg=main_beam_deg,
        hpbw_deg=round(hpbw, 2),
        first_sidelobe_db=round(first_sll, 2),
        array_gain_db=round(array_gain_db, 2),
        pattern_2d=pattern_2d_db.tolist(),
    )


# ---------------------------------------------------------------------------
# Radar calculations
# ---------------------------------------------------------------------------

def radar_range_equation(params: RadarParams) -> RadarResult:
    """Solve the radar range equation for SNR vs range."""
    lam = 3e8 / params.frequency_hz
    g_linear = 10.0 ** (params.antenna_gain_dbi / 10.0)
    nf_linear = 10.0 ** (params.noise_figure_db / 10.0)
    loss_linear = 10.0 ** (params.losses_db / 10.0)
    k_boltzmann = 1.380649e-23
    t0 = 290.0

    noise_power_w = k_boltzmann * t0 * params.bandwidth_hz * nf_linear * loss_linear
    noise_power_dbm = 10.0 * math.log10(noise_power_w * 1000.0)

    # Numerator of radar equation
    numerator = params.peak_power_w * (g_linear ** 2) * (lam ** 2) * params.rcs_m2

    # Max range for SNR = 0 dB (SNR_min = 1)
    denominator = (4.0 * math.pi) ** 3 * noise_power_w
    max_range_m = (numerator / denominator) ** 0.25
    max_range_km = max_range_m / 1000.0

    # SNR at sample ranges
    ranges_km = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    snr_at_range = {}
    for r_km in ranges_km:
        r_m = r_km * 1000.0
        received_w = (params.peak_power_w * (g_linear ** 2) * (lam ** 2) * params.rcs_m2) / (
            (4.0 * math.pi) ** 3 * (r_m ** 4) * loss_linear
        )
        snr = 10.0 * math.log10(max(received_w / noise_power_w, 1e-30))
        snr_at_range[f"{r_km}"] = round(snr, 2)

    # Min detectable RCS at 50 km (SNR ≥ 10 dB)
    r_ref = 50e3
    min_rcs = (
        ((4.0 * math.pi) ** 3 * (r_ref ** 4) * noise_power_w * 10.0)
        / (params.peak_power_w * (g_linear ** 2) * (lam ** 2))
    )

    return RadarResult(
        max_range_km=round(max_range_km, 2),
        snr_at_range_db=snr_at_range,
        min_detectable_rcs_m2=round(min_rcs, 4),
        noise_power_dbm=round(noise_power_dbm, 2),
    )


def estimate_rcs(
    target_type: str,
    frequency_hz: float,
) -> float:
    """Return a typical RCS estimate (m²) for common target types.

    Based on published radar engineering references.
    """
    lam = 3e8 / frequency_hz
    table: dict[str, float] = {
        "person": 1.0,
        "bicycle": 2.0,
        "motorcycle": 3.0,
        "car": 10.0,
        "truck": 200.0,
        "small_aircraft": 2.0,
        "large_aircraft": 100.0,
        "drone_small": 0.01,
        "drone_large": 0.1,
        "ship_small": 1000.0,
        "ship_large": 50000.0,
        "missile": 0.1,
        "sphere_1m": math.pi * 0.5 ** 2,
    }
    return table.get(target_type.lower(), 1.0)


# ---------------------------------------------------------------------------
# Link budget
# ---------------------------------------------------------------------------

def link_budget(
    tx_power_dbm: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    frequency_hz: float,
    distance_km: float,
    rx_noise_figure_db: float = 5.0,
    bandwidth_hz: float = 200e3,
    losses_db: float = 2.0,
    model: str = "fspl",
) -> LinkBudgetResult:
    """Compute a complete RF link budget."""
    from backend.rf_propagation import path_loss_db

    freq_mhz = frequency_hz / 1e6
    pl = path_loss_db(model, freq_mhz, distance_km)  # type: ignore[arg-type]
    eirp_dbm = tx_power_dbm + tx_gain_dbi
    rx_power_dbm = eirp_dbm - pl + rx_gain_dbi - losses_db

    # Noise floor at receiver
    k_boltzmann = 1.380649e-23
    t0 = 290.0
    noise_w = k_boltzmann * t0 * bandwidth_hz * (10.0 ** (rx_noise_figure_db / 10.0))
    noise_dbm = 10.0 * math.log10(noise_w * 1000.0)

    snr_db = rx_power_dbm - noise_dbm
    link_margin_db = snr_db - 10.0  # typical 10 dB margin requirement

    return LinkBudgetResult(
        received_power_dbm=round(rx_power_dbm, 2),
        path_loss_db=round(pl, 2),
        snr_db=round(snr_db, 2),
        link_margin_db=round(link_margin_db, 2),
        details={
            "eirp_dbm": round(eirp_dbm, 2),
            "noise_floor_dbm": round(noise_dbm, 2),
            "distance_km": distance_km,
            "frequency_mhz": round(freq_mhz, 3),
            "propagation_model": model,
        },
    )


# ---------------------------------------------------------------------------
# Phase-difference bearing estimation
# ---------------------------------------------------------------------------

def bearing_from_phase_difference(
    phase_diff_rad: float,
    element_spacing_m: float,
    frequency_hz: float,
    array_orientation_deg: float = 0.0,
) -> float:
    """Estimate signal arrival angle from inter-element phase difference.

    Returns bearing in degrees (0–360, true north).
    """
    lam = 3e8 / frequency_hz
    # phase_diff = (2π/λ) * d * sin(θ)
    sin_theta = (phase_diff_rad * lam) / (2.0 * math.pi * element_spacing_m)
    sin_theta = max(-1.0, min(1.0, sin_theta))
    theta_rad = math.asin(sin_theta)
    bearing = (math.degrees(theta_rad) + array_orientation_deg) % 360.0
    return round(bearing, 2)
