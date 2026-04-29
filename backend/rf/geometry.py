from __future__ import annotations

import math
from typing import Any

EARTH_RADIUS_M = 6_371_000


def _destination(lon: float, lat: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """Return lon/lat destination given origin, bearing, and distance."""
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    bearing = math.radians(bearing_deg)
    ang_dist = distance_m / EARTH_RADIUS_M

    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang_dist)
        + math.cos(lat1) * math.sin(ang_dist) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(ang_dist) * math.cos(lat1),
        math.cos(ang_dist) - math.sin(lat1) * math.sin(lat2),
    )

    lon_out = (math.degrees(lon2) + 540) % 360 - 180
    lat_out = math.degrees(lat2)
    return lon_out, lat_out


def _power_class(radius_hint_m: float) -> str:
    if radius_hint_m >= 1800:
        return "high"
    if radius_hint_m >= 1200:
        return "medium"
    return "low"


def _radii_for_feature(props: dict[str, Any]) -> tuple[float, float, float]:
    if props.get("kind") == "estimate":
        base = float(props.get("confidence_major_m") or 900)
    else:
        span = float((props.get("rf_max_mhz") or 1800) - (props.get("rf_min_mhz") or 700))
        base = 900 + max(0.0, min(span, 2200.0)) * 0.6

    power_class = _power_class(base)
    power_scale = {"low": 0.95, "medium": 1.1, "high": 1.25}[power_class]
    wedge_radius = base * power_scale

    tilt_proxy_deg = 8 if props.get("kind") == "estimate" else 5
    ray_length = wedge_radius * (1.0 - tilt_proxy_deg / 70.0)
    footprint_radius = wedge_radius * (1.3 if props.get("kind") == "estimate" else 1.15)
    return ray_length, wedge_radius, footprint_radius


def build_propagation_features(feature: dict[str, Any]) -> list[dict[str, Any]]:
    props = feature.get("properties", {})
    geometry = feature.get("geometry", {})
    if geometry.get("type") != "Point":
        return []

    lon, lat = geometry.get("coordinates", [None, None])
    if lon is None or lat is None:
        return []

    kind = props.get("kind", "unknown")
    azimuth = float(props.get("azimuth_deg") if props.get("azimuth_deg") is not None else 0)
    beamwidth = 360 if props.get("directionality") == "Omni" else (78 if kind == "infrastructure" else 95)
    tilt_proxy_deg = 8 if kind == "estimate" else 5
    ray_len, wedge_radius, footprint_radius = _radii_for_feature(props)
    power_class = _power_class(wedge_radius)

    center_end = _destination(lon, lat, azimuth, ray_len)
    centerline = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[lon, lat], [center_end[0], center_end[1]]]},
        "properties": {
            "source_id": props.get("id"),
            "source_name": props.get("name"),
            "source_kind": kind,
            "beam_type": "centerline",
            "azimuth_deg": azimuth,
            "beamwidth_deg": beamwidth,
            "tilt_proxy_deg": tilt_proxy_deg,
            "power_class": power_class,
            "radius_m": round(ray_len, 1),
            "timestamp": props.get("timestamp"),
            "assumptions": "Beamwidth and tilt are inferred from directionality/kind and RF metadata.",
        },
    }

    steps = 20
    start = azimuth - beamwidth / 2
    arc = [[lon, lat]]
    for i in range(steps + 1):
        b = start + (beamwidth * i / steps)
        x, y = _destination(lon, lat, b, wedge_radius)
        arc.append([x, y])
    arc.append([lon, lat])

    wedge = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [arc]},
        "properties": {
            "source_id": props.get("id"),
            "source_name": props.get("name"),
            "source_kind": kind,
            "beam_type": "wedge",
            "azimuth_deg": azimuth,
            "beamwidth_deg": beamwidth,
            "tilt_proxy_deg": tilt_proxy_deg,
            "power_class": power_class,
            "radius_m": round(wedge_radius, 1),
            "timestamp": props.get("timestamp"),
            "assumptions": "Sector wedge uses inferred beamwidth and a power-class-scaled radius.",
        },
    }

    ring = []
    major = footprint_radius
    minor = footprint_radius * (0.65 if kind == "estimate" else 0.78)
    for i in range(0, 48):
        theta = (i / 48) * 360
        radians = math.radians(theta)
        x_local = major * math.cos(radians)
        y_local = minor * math.sin(radians)
        rot = math.radians(azimuth)
        x_rot = x_local * math.cos(rot) - y_local * math.sin(rot)
        y_rot = x_local * math.sin(rot) + y_local * math.cos(rot)
        dist = math.hypot(x_rot, y_rot)
        bearing = (math.degrees(math.atan2(x_rot, y_rot)) + 360) % 360
        x, y = _destination(lon, lat, bearing, dist)
        ring.append([x, y])
    ring.append(ring[0])

    confidence = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "source_id": props.get("id"),
            "source_name": props.get("name"),
            "source_kind": kind,
            "beam_type": "confidence",
            "azimuth_deg": azimuth,
            "beamwidth_deg": beamwidth,
            "tilt_proxy_deg": tilt_proxy_deg,
            "power_class": power_class,
            "radius_m": round(footprint_radius, 1),
            "confidence_score": props.get("confidence_score"),
            "timestamp": props.get("timestamp"),
            "assumptions": "Confidence footprint is an orientation-aligned ellipse scaled by power class.",
        },
    }

    return [centerline, wedge, confidence]
