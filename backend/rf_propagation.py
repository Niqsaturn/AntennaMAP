from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

ModelName = Literal["fspl", "hata_urban", "hata_suburban"]


@dataclass(frozen=True)
class AntennaConfig:
    lat: float
    lon: float
    eirp_dbm: float
    frequency_mhz: float
    gain_dbi: float = 0.0
    height_m: float = 30.0
    beamwidth_deg: float = 360.0
    tilt_deg: float = 0.0
    orientation_deg: float = 0.0


@dataclass(frozen=True)
class PropagationRequest:
    antenna: AntennaConfig
    model: ModelName = "fspl"
    grid_radius_km: float = 5.0
    grid_resolution: int = 41


def _angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def path_loss_db(model: ModelName, frequency_mhz: float, distance_km: float, tx_height_m: float = 30.0) -> float:
    d = max(distance_km, 0.001)
    if model == "fspl":
        return 32.44 + 20.0 * math.log10(frequency_mhz) + 20.0 * math.log10(d)

    # COST-231 Hata style approximation
    hr = 1.5
    a_hr = (1.1 * math.log10(frequency_mhz) - 0.7) * hr - (1.56 * math.log10(frequency_mhz) - 0.8)
    urban = (
        46.3
        + 33.9 * math.log10(frequency_mhz)
        - 13.82 * math.log10(max(tx_height_m, 1.0))
        - a_hr
        + (44.9 - 6.55 * math.log10(max(tx_height_m, 1.0))) * math.log10(d)
    )
    if model == "hata_urban":
        return urban
    if model == "hata_suburban":
        return urban - 2.0 * (math.log10(frequency_mhz / 28.0) ** 2) - 5.4
    raise ValueError(f"unsupported model: {model}")


def _beam_adjustment_db(cfg: AntennaConfig, target_bearing_deg: float) -> float:
    if cfg.beamwidth_deg >= 359.0:
        return 0.0
    off_axis = _angle_diff_deg(target_bearing_deg, cfg.orientation_deg)
    if off_axis <= cfg.beamwidth_deg / 2.0:
        return 0.0
    return -min(20.0, (off_axis - cfg.beamwidth_deg / 2.0) * 0.5)


def estimate_rsrp_dbm(cfg: AntennaConfig, model: ModelName, distance_km: float, bearing_deg: float) -> float:
    eirp = cfg.eirp_dbm + cfg.gain_dbi + _beam_adjustment_db(cfg, bearing_deg) - abs(cfg.tilt_deg) * 0.15
    return eirp - path_loss_db(model, cfg.frequency_mhz, distance_km, cfg.height_m)


def _to_lon_delta_km(lat: float, lon_delta_km: float) -> float:
    return lon_delta_km / (111.32 * math.cos(math.radians(lat)))


def contour_polygon(cfg: AntennaConfig, model: ModelName, threshold_dbm: float, points: int = 40) -> dict:
    dist = 0.05
    while dist < 40:
        rsrp = estimate_rsrp_dbm(cfg, model, dist, cfg.orientation_deg)
        if rsrp < threshold_dbm:
            break
        dist += 0.05
    ring = []
    for i in range(points + 1):
        theta = 2 * math.pi * i / points
        dy = dist * math.sin(theta)
        dx = dist * math.cos(theta)
        lat = cfg.lat + dy / 111.32
        lon = cfg.lon + _to_lon_delta_km(cfg.lat, dx)
        ring.append([lon, lat])
    return {"type": "Polygon", "coordinates": [ring]}


@lru_cache(maxsize=128)
def generate_snapshot(
    lat: float,
    lon: float,
    eirp_dbm: float,
    frequency_mhz: float,
    gain_dbi: float,
    height_m: float,
    beamwidth_deg: float,
    tilt_deg: float,
    orientation_deg: float,
    model: ModelName,
    grid_radius_km: float,
    grid_resolution: int,
) -> dict:
    cfg = AntennaConfig(lat, lon, eirp_dbm, frequency_mhz, gain_dbi, height_m, beamwidth_deg, tilt_deg, orientation_deg)
    n = max(11, min(grid_resolution, 151))
    step = (2 * grid_radius_km) / (n - 1)
    grid = []
    for r in range(n):
        row = []
        y_km = -grid_radius_km + r * step
        for c in range(n):
            x_km = -grid_radius_km + c * step
            lat2 = cfg.lat + y_km / 111.32
            lon2 = cfg.lon + _to_lon_delta_km(cfg.lat, x_km)
            d = max(0.001, (x_km * x_km + y_km * y_km) ** 0.5)
            bearing = _bearing_deg(cfg.lat, cfg.lon, lat2, lon2)
            row.append(round(estimate_rsrp_dbm(cfg, model, d, bearing), 2))
        grid.append(row)

    zones = {
        "strong": {"threshold_dbm": -75, "polygon": contour_polygon(cfg, model, -75)},
        "moderate": {"threshold_dbm": -90, "polygon": contour_polygon(cfg, model, -90)},
        "weak": {"threshold_dbm": -105, "polygon": contour_polygon(cfg, model, -105)},
    }
    return {
        "site": cfg.__dict__,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grid": {
            "radius_km": grid_radius_km,
            "resolution": n,
            "bounds": {
                "min_lat": cfg.lat - grid_radius_km / 111.32,
                "max_lat": cfg.lat + grid_radius_km / 111.32,
                "min_lon": cfg.lon - _to_lon_delta_km(cfg.lat, grid_radius_km),
                "max_lon": cfg.lon + _to_lon_delta_km(cfg.lat, grid_radius_km),
            },
            "field_strength_rsrp_dbm": grid,
        },
        "contours": zones,
        "uncertainty": {
            "sigma_db": 8.0,
            "confidence": 0.8,
            "factors": ["terrain ignored", "building clutter simplified", "weather ignored"],
        },
        "assumptions": [
            "Antenna gain/tilt patterns are simplified.",
            "LOS/NLOS transitions are not explicitly modeled.",
            "Model is suitable for planning, not compliance measurements.",
        ],
    }


# ── EM Field Functions ────────────────────────────────────────────────────────

def em_field_at_point(
    eirp_dbm: float,
    frequency_hz: float,
    distance_m: float,
) -> dict[str, float]:
    """Compute far-field E and H field strengths at distance_m.

    Uses the Friis far-field relationship:
      E (V/m) = sqrt(30 × EIRP_watts) / distance_m   (far field, LOS)
      H (A/m) = E / 377  (free-space impedance)
    Near-field region boundary: distance < λ / (2π)
    """
    eirp_w = 10.0 ** ((eirp_dbm - 30.0) / 10.0)
    lam = 3e8 / max(frequency_hz, 1.0)
    near_field_boundary_m = lam / (2.0 * math.pi)
    far_field = distance_m >= near_field_boundary_m

    d = max(distance_m, 0.001)
    if far_field:
        e_vm = math.sqrt(30.0 * eirp_w) / d
    else:
        # Near-field approximation (simplified Hertzian dipole)
        e_vm = (math.sqrt(30.0 * eirp_w) / d) * (lam / (2.0 * math.pi * d))

    h_am = e_vm / 376.73
    power_density_wm2 = e_vm ** 2 / 376.73
    power_density_dbm_m2 = 10.0 * math.log10(max(power_density_wm2, 1e-30) * 1000.0)

    return {
        "e_field_v_per_m": round(e_vm, 6),
        "h_field_a_per_m": round(h_am, 9),
        "power_density_w_per_m2": round(power_density_wm2, 9),
        "power_density_dbm_per_m2": round(power_density_dbm_m2, 2),
        "near_field_boundary_m": round(near_field_boundary_m, 3),
        "is_far_field": far_field,
        "wavelength_m": round(lam, 6),
    }


def em_field_grid(
    lat: float,
    lon: float,
    eirp_dbm: float,
    frequency_hz: float,
    radius_km: float = 5.0,
    n_points: int = 9,
) -> dict:
    """Return a GeoJSON FeatureCollection grid of EM field values."""
    step = (2 * radius_km) / max(n_points - 1, 1)
    features = []
    for r in range(n_points):
        y_km = -radius_km + r * step
        for c in range(n_points):
            x_km = -radius_km + c * step
            d_m = max(1.0, math.sqrt(x_km ** 2 + y_km ** 2) * 1000.0)
            glat = lat + y_km / 111.32
            glon = lon + x_km / (111.32 * math.cos(math.radians(lat)))
            field = em_field_at_point(eirp_dbm, frequency_hz, d_m)
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(glon, 6), round(glat, 6)]},
                "properties": {
                    "distance_m": round(d_m, 1),
                    **field,
                    "freq_hz": frequency_hz,
                },
            })
    return {"type": "FeatureCollection", "features": features}


def ionospheric_skip_distance_km(
    freq_mhz: float,
    solar_flux_index: float = 100.0,
    ionosphere_height_km: float = 300.0,
) -> dict[str, float]:
    """Estimate HF sky-wave skip distance and maximum usable frequency.

    Uses a simplified ionospheric model. Valid for HF band (3–30 MHz).
    The skip distance is where the ground wave ends and sky wave returns.

    foF2 (critical frequency) ≈ 0.0089 × sqrt(solar_flux_index) MHz
    """
    fo_f2_mhz = 0.0089 * math.sqrt(max(solar_flux_index, 1.0)) * 10.0
    muf_mhz = fo_f2_mhz * math.sqrt(1.0 + (1500.0 / ionosphere_height_km) ** 2)

    if freq_mhz <= 0 or fo_f2_mhz <= 0:
        skip_km = 0.0
    elif freq_mhz >= muf_mhz:
        skip_km = float("inf")
    else:
        sin_angle = fo_f2_mhz / freq_mhz
        sin_angle = min(1.0, sin_angle)
        elevation_rad = math.asin(sin_angle)
        skip_km = 2.0 * ionosphere_height_km / math.tan(max(elevation_rad, 0.01))

    return {
        "freq_mhz": freq_mhz,
        "fo_f2_mhz": round(fo_f2_mhz, 2),
        "muf_mhz": round(muf_mhz, 2),
        "skip_distance_km": round(min(skip_km, 10000.0), 1),
        "ionosphere_height_km": ionosphere_height_km,
        "solar_flux_index": solar_flux_index,
        "propagation_possible": freq_mhz <= muf_mhz,
    }


def atmospheric_absorption_db(freq_ghz: float, distance_km: float) -> dict[str, float]:
    """Estimate atmospheric absorption loss (ITU-R P.676 simplified).

    Significant at SHF/EHF — O₂ peak at 60 GHz, H₂O at 22 GHz/183 GHz.
    Standard atmosphere: T=15°C, P=1013 hPa, 7.5 g/m³ water vapor.
    """
    # O₂ specific attenuation (dB/km) — simplified double-Gaussian model
    o2 = (
        7.19e-3
        + 6.09 / (freq_ghz ** 2 + 0.227)
        + 4.81 / ((freq_ghz - 57.0) ** 2 + 1.50)
    ) * freq_ghz ** 2 * 1e-3

    # H₂O specific attenuation (dB/km) at 7.5 g/m³
    h2o = (
        0.050 + 0.0021 * 7.5
        + 3.6 / ((freq_ghz - 22.2) ** 2 + 8.5)
        + 10.6 / ((freq_ghz - 183.3) ** 2 + 9.0)
        + 8.9 / ((freq_ghz - 325.4) ** 2 + 26.3)
    ) * freq_ghz ** 2 * 7.5e-4

    total_db_per_km = max(0.0, o2 + h2o)
    total_db = total_db_per_km * distance_km

    return {
        "freq_ghz": freq_ghz,
        "distance_km": distance_km,
        "o2_attenuation_db_per_km": round(o2, 4),
        "h2o_attenuation_db_per_km": round(h2o, 4),
        "total_db_per_km": round(total_db_per_km, 4),
        "total_absorption_db": round(total_db, 3),
    }
