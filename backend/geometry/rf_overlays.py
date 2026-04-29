from __future__ import annotations

import math
from typing import Any

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

def build_overlay_geometries(feature: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    geom = feature.get("geometry", {})
    props = feature.get("properties", {})
    if geom.get("type") != "Point":
        return {"site_point": None, "direction_ray": None, "sector_wedge": None, "confidence_ellipse": None}
    lon, lat = geom.get("coordinates", [None, None])
    if lon is None or lat is None:
        return {"site_point": None, "direction_ray": None, "sector_wedge": None, "confidence_ellipse": None}
    azimuth = float(props.get("azimuth_deg") or 0)
    beamwidth = float(props.get("beamwidth_deg") or (360 if props.get("directionality") == "Omni" else 80))
    ray_length = float(props.get("ray_length_m") or 900)
    wedge_radius = float(props.get("wedge_radius_m") or 1200)
    conf_major = float(props.get("confidence_major_m") or 700)
    conf_minor = float(props.get("confidence_minor_m") or max(200, conf_major * 0.6))

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
    }
