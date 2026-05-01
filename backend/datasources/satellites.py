"""Satellite position and relay map data.

Uses publicly available TLE (Two-Line Element) data from Celestrak to compute
satellite positions, ground tracks, footprints, and uplink/downlink arcs.
All data is fetched from public sources — no API key required.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_CELESTRAK_BASE = "https://celestrak.org/SOCRATES/"
_TLE_URLS: dict[str, str] = {
    "gps": "https://celestrak.org/pub/TLE/gps.txt",
    "stations": "https://celestrak.org/pub/TLE/stations.txt",
    "active": "https://celestrak.org/pub/TLE/active.txt",
    "starlink": "https://celestrak.org/pub/TLE/starlink.txt",
    "iridium": "https://celestrak.org/pub/TLE/iridium.txt",
    "goes": "https://celestrak.org/pub/TLE/goes.txt",
    "geo": "https://celestrak.org/pub/TLE/geo.txt",
    "noaa": "https://celestrak.org/pub/TLE/noaa.txt",
}
_TIMEOUT = 15.0


async def fetch_tle_group(group: str, limit: int = 50) -> list[dict[str, Any]]:
    """Fetch TLE data for a named group from Celestrak.

    Returns list of {"name", "line1", "line2"} dicts.
    """
    url = _TLE_URLS.get(group.lower(), _TLE_URLS["active"])
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            text = resp.text
    except Exception as exc:
        logger.warning("Celestrak TLE fetch failed for group=%s: %s", group, exc)
        return []

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    sats = []
    i = 0
    while i + 2 < len(lines):
        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]
        if line1.startswith("1 ") and line2.startswith("2 "):
            sats.append({"name": name, "line1": line1, "line2": line2})
            i += 3
        else:
            i += 1
        if len(sats) >= limit:
            break
    return sats


def tle_to_position(
    name: str,
    line1: str,
    line2: str,
    epoch_utc: datetime | None = None,
) -> dict[str, Any] | None:
    """Compute satellite position (lat, lon, alt_km) from TLE at epoch_utc."""
    try:
        from sgp4.api import Satrec, jday
    except ImportError:
        logger.warning("sgp4 not installed; satellite positioning unavailable")
        return None

    epoch = epoch_utc or datetime.now(timezone.utc)
    sat = Satrec.twoline2rv(line1, line2)
    jd, fr = jday(epoch.year, epoch.month, epoch.day,
                  epoch.hour, epoch.minute, epoch.second + epoch.microsecond / 1e6)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return None

    # r is in km (TEME frame) — convert to geodetic lat/lon
    x, y, z = r
    lon_rad = math.atan2(y, x)
    hyp = math.sqrt(x * x + y * y)
    lat_rad = math.atan2(z, hyp)

    # Account for Earth's rotation (TEME → ECEF sidereal transform, simplified)
    gmst = _gmst(jd + fr)
    lon_ecef = lon_rad - gmst
    lon_ecef = (math.degrees(lon_ecef) + 360) % 360
    if lon_ecef > 180:
        lon_ecef -= 360

    alt_km = math.sqrt(x * x + y * y + z * z) - 6371.0

    return {
        "name": name,
        "lat": round(math.degrees(lat_rad), 4),
        "lon": round(lon_ecef, 4),
        "alt_km": round(alt_km, 2),
        "epoch": epoch.isoformat(),
    }


def _gmst(jd_ut1: float) -> float:
    """Compute Greenwich Mean Sidereal Time in radians."""
    t = (jd_ut1 - 2451545.0) / 36525.0
    theta = (67310.54841 + (876600 * 3600 + 8640184.812866) * t +
             0.093104 * t * t - 6.2e-6 * t * t * t)
    return math.radians(theta % 86400 / 240.0)


def satellite_ground_track(
    name: str,
    line1: str,
    line2: str,
    hours: float = 6.0,
    step_min: float = 5.0,
) -> list[dict[str, Any]]:
    """Return ground track positions at step_min intervals over hours."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    track = []
    t = now
    end = now + timedelta(hours=hours)
    while t <= end:
        pos = tle_to_position(name, line1, line2, t)
        if pos:
            track.append({
                "lat": pos["lat"],
                "lon": pos["lon"],
                "alt_km": pos["alt_km"],
                "timestamp": t.isoformat(),
            })
        t += timedelta(minutes=step_min)
    return track


def satellite_footprint(
    lat: float,
    lon: float,
    alt_km: float,
    min_elevation_deg: float = 5.0,
    n_points: int = 60,
) -> dict[str, Any]:
    """Return GeoJSON Polygon for satellite visibility footprint (coverage circle)."""
    Re = 6371.0
    rho = math.acos(Re / (Re + alt_km)) - math.radians(min_elevation_deg)
    rho = max(0.0, rho)
    half_angle_deg = math.degrees(rho)
    ring = []
    for i in range(n_points + 1):
        bearing = 360.0 * i / n_points
        pt = _destination(lon, lat, bearing, half_angle_deg * 111320)
        ring.append(pt)
    return {"type": "Polygon", "coordinates": [ring]}


def uplink_downlink_arc(
    ground_lat: float,
    ground_lon: float,
    sat_lat: float,
    sat_lon: float,
    n_points: int = 20,
) -> dict[str, Any]:
    """Return GeoJSON LineString great-circle arc from ground to satellite nadir."""
    coords = _great_circle_arc(ground_lon, ground_lat, sat_lon, sat_lat, n_points)
    return {"type": "LineString", "coordinates": coords}


def _great_circle_arc(
    lon1: float, lat1: float,
    lon2: float, lat2: float,
    n: int,
) -> list[list[float]]:
    pts = []
    for i in range(n + 1):
        f = i / n
        lat, lon = _interpolate_gc(lat1, lon1, lat2, lon2, f)
        pts.append([round(lon, 5), round(lat, 5)])
    return pts


def _interpolate_gc(lat1: float, lon1: float, lat2: float, lon2: float, f: float) -> tuple[float, float]:
    phi1, lam1 = math.radians(lat1), math.radians(lon1)
    phi2, lam2 = math.radians(lat2), math.radians(lon2)
    d = math.acos(math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(phi2) * math.cos(lam2 - lam1))
    if d < 1e-10:
        return lat1, lon1
    A = math.sin((1 - f) * d) / math.sin(d)
    B = math.sin(f * d) / math.sin(d)
    x = A * math.cos(phi1) * math.cos(lam1) + B * math.cos(phi2) * math.cos(lam2)
    y = A * math.cos(phi1) * math.sin(lam1) + B * math.cos(phi2) * math.sin(lam2)
    z = A * math.sin(phi1) + B * math.sin(phi2)
    lat = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon


def _destination(lon: float, lat: float, bearing_deg: float, distance_m: float) -> list[float]:
    R = 6371000.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    br = math.radians(bearing_deg)
    d = distance_m / R
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(br))
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return [round(math.degrees(lon2), 6), round(math.degrees(lat2), 6)]


async def build_relay_map(
    groups: list[str] | None = None,
    earth_stations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a GeoJSON FeatureCollection of satellites + footprints + arcs."""
    from backend.datasources.earth_stations import EARTH_STATIONS
    groups = groups or ["gps", "goes", "geo"]
    earth_stations = earth_stations or EARTH_STATIONS

    features: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    # Fetch TLE groups concurrently
    import asyncio
    tle_lists = await asyncio.gather(*[fetch_tle_group(g, limit=30) for g in groups])

    for group, tle_list in zip(groups, tle_lists):
        for tle in tle_list:
            pos = tle_to_position(tle["name"], tle["line1"], tle["line2"], now)
            if not pos:
                continue
            lat, lon, alt = pos["lat"], pos["lon"], pos["alt_km"]

            # Satellite position point
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "kind": "satellite",
                    "name": tle["name"],
                    "group": group,
                    "alt_km": alt,
                    "epoch": now.isoformat(),
                },
            })

            # Footprint polygon
            footprint = satellite_footprint(lat, lon, alt)
            features.append({
                "type": "Feature",
                "geometry": footprint,
                "properties": {"kind": "sat_footprint", "name": tle["name"], "alt_km": alt},
            })

            # Uplink/downlink arcs from earth stations to this satellite
            for es in earth_stations:
                arc = uplink_downlink_arc(es["lat"], es["lon"], lat, lon)
                features.append({
                    "type": "Feature",
                    "geometry": arc,
                    "properties": {
                        "kind": "relay_arc",
                        "ground_station": es["name"],
                        "satellite": tle["name"],
                        "arc_type": "uplink_downlink",
                    },
                })

    # Earth station points
    for es in earth_stations:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [es["lon"], es["lat"]]},
            "properties": {
                "kind": "earth_station",
                "name": es["name"],
                "operator": es.get("operator"),
                "freq_band": es.get("freq_band"),
                "type": es.get("type"),
            },
        })

    return {"type": "FeatureCollection", "features": features}
