from __future__ import annotations

import math
from typing import Any

from backend.analysis.range_estimator import estimate_range_km

EARTH_RADIUS_M = 6_371_000

def _destination(lon: float, lat: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    bearing = math.radians(bearing_deg)
    angular = distance_m / EARTH_RADIUS_M
    lat2 = math.asin(math.sin(lat1) * math.cos(angular) + math.cos(lat1) * math.sin(angular) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(angular) * math.cos(lat1), math.cos(angular) - math.sin(lat1) * math.sin(lat2))
    return (math.degrees(lon2) + 540) % 360 - 180, math.degrees(lat2)

def _ellipse(lon: float, lat: float, azimuth_deg: float, major_m: float, minor_m: float, steps: int = 48) -> list[list[float]]:
    ring: list[list[float]] = []
    rot = math.radians(azimuth_deg)
    for i in range(steps):
        theta = (2 * math.pi * i) / steps
        x = major_m * math.cos(theta)
        y = minor_m * math.sin(theta)
        xr = x * math.cos(rot) - y * math.sin(rot)
        yr = x * math.sin(rot) + y * math.cos(rot)
        dist = math.hypot(xr, yr)
        bearing = (math.degrees(math.atan2(xr, yr)) + 360) % 360
        ring.append(list(_destination(lon, lat, bearing, dist)))
    ring.append(ring[0])
    return ring


def _ring_band(lon: float, lat: float, inner_m: float, outer_m: float, steps: int = 48) -> list[list[list[float]]]:
    bearings = [i * (360 / steps) for i in range(steps)]
    inner = [_destination(lon, lat, b, max(1.0, inner_m)) for b in bearings]
    outer = [_destination(lon, lat, b, max(inner_m + 1.0, outer_m)) for b in bearings]
    inner_ring = [list(p) for p in inner] + [list(inner[0])]
    outer_ring = [list(p) for p in reversed(outer)] + [list(outer[-1])]
    return [outer_ring, inner_ring]

def build_overlay_geometries(feature: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    geom = feature.get("geometry", {})
    props = feature.get("properties", {})
    if geom.get("type") != "Point":
        return {"site_point": None, "direction_ray": None, "sector_wedge": None, "confidence_ellipse": None, "range_likely_band": None, "uncertainty_polygon": None}
    lon, lat = geom.get("coordinates", [None, None])
    if lon is None or lat is None:
        return {"site_point": None, "direction_ray": None, "sector_wedge": None, "confidence_ellipse": None, "range_likely_band": None, "uncertainty_polygon": None}
    azimuth = float(props.get("azimuth_deg") or 0)
    beamwidth = float(props.get("beamwidth_deg") or (360 if props.get("directionality") == "Omni" else 80))
    ray_length = float(props.get("ray_length_m") or 900)
    wedge_radius = float(props.get("wedge_radius_m") or 1200)
    conf_major = float(props.get("confidence_major_m") or 700)
    conf_minor = float(props.get("confidence_minor_m") or max(200, conf_major * 0.6))
    snr_db = float(props.get("snr_db") or 10.0)
    rssi_dbm = float(props.get("rssi_dbm") or -85.0)
    freq_mhz = float(props.get("frequency_mhz") or (float(props.get("frequency_hz") or 100_000_000.0) / 1e6))
    bandwidth_hz = float(props.get("bandwidth_hz") or 25_000.0)

    range_km, range_sigma_km = estimate_range_km(rssi_dbm=rssi_dbm, freq_mhz=freq_mhz, snr_db=snr_db, bandwidth_hz=bandwidth_hz)
    likely_inner_m = max(50.0, (range_km - range_sigma_km) * 1000.0)
    likely_outer_m = max(likely_inner_m + 50.0, (range_km + range_sigma_km) * 1000.0)

    ray_end = _destination(lon, lat, azimuth, ray_length)
    start = azimuth - beamwidth / 2
    sector = [[lon, lat]]
    for i in range(21):
        b = start + (beamwidth * i / 20)
        x, y = _destination(lon, lat, b, wedge_radius)
        sector.append([x, y])
    sector.append([lon, lat])

    return {
        "site_point": {"type": "Point", "coordinates": [lon, lat]},
        "direction_ray": {"type": "LineString", "coordinates": [[lon, lat], [ray_end[0], ray_end[1]]]},
        "sector_wedge": {"type": "Polygon", "coordinates": [sector]},
        "confidence_ellipse": {"type": "Polygon", "coordinates": [_ellipse(lon, lat, azimuth, conf_major, conf_minor)]},
        "range_likely_band": {"type": "Polygon", "coordinates": _ring_band(lon, lat, likely_inner_m, likely_outer_m)},
        "uncertainty_polygon": {"type": "Polygon", "coordinates": [_ellipse(lon, lat, azimuth, conf_major * 1.4, conf_minor * 1.4)]},
    }
