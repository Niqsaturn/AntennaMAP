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
