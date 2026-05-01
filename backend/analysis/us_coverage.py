"""Progressive US CONUS coverage manager.

Maintains a 0.5° lat/lon tile grid covering the contiguous United States
(CONUS). Tiles are seeded from public data sources (FCC ULS, cell towers)
and analyzed progressively starting from the operator's current position.
"""
from __future__ import annotations

import asyncio
import math
from typing import Any

from backend.storage.map_store import upsert_tile, get_coverage_progress

# CONUS bounding box (0.5° tiles → 50 lat × 117 lon = 5,850 tiles)
_LAT_MIN, _LAT_MAX = 24.5, 49.5
_LON_MIN, _LON_MAX = -125.0, -66.5
_TILE_DEG = 0.5


def _tile_id(lat_floor: float, lon_floor: float) -> str:
    return f"{lat_floor:.1f}_{lon_floor:.1f}"


def _tile_bounds(lat_floor: float, lon_floor: float) -> tuple[float, float, float, float]:
    return lat_floor, lon_floor, lat_floor + _TILE_DEG, lon_floor + _TILE_DEG


def _operator_tile(lat: float, lon: float) -> tuple[float, float]:
    lat_floor = math.floor(lat / _TILE_DEG) * _TILE_DEG
    lon_floor = math.floor(lon / _TILE_DEG) * _TILE_DEG
    return lat_floor, lon_floor


def _moore_neighborhood(lat_f: float, lon_f: float) -> list[tuple[float, float]]:
    """Return the 8 adjacent tiles surrounding (lat_f, lon_f)."""
    neighbors = []
    for dlat in (-_TILE_DEG, 0.0, _TILE_DEG):
        for dlon in (-_TILE_DEG, 0.0, _TILE_DEG):
            if dlat == 0.0 and dlon == 0.0:
                continue
            nlat = round(lat_f + dlat, 1)
            nlon = round(lon_f + dlon, 1)
            if _LAT_MIN <= nlat <= _LAT_MAX - _TILE_DEG and _LON_MIN <= nlon <= _LON_MAX - _TILE_DEG:
                neighbors.append((nlat, nlon))
    return neighbors


def next_tiles_to_process(
    operator_lat: float,
    operator_lon: float,
    n: int = 5,
) -> list[dict[str, Any]]:
    """Return up to n tiles to process, prioritised by proximity to operator."""
    from backend.storage.map_store import _conn

    lat_f, lon_f = _operator_tile(operator_lat, operator_lon)
    candidates: list[tuple[float, float]] = [(lat_f, lon_f)]
    candidates.extend(_moore_neighborhood(lat_f, lon_f))

    # Add more tiles in expanding rings until we have enough candidates
    ring = 2
    while len(candidates) < n * 3 and ring <= 10:
        for dlat_steps in range(-ring, ring + 1):
            for dlon_steps in range(-ring, ring + 1):
                if abs(dlat_steps) != ring and abs(dlon_steps) != ring:
                    continue
                nlat = round(lat_f + dlat_steps * _TILE_DEG, 1)
                nlon = round(lon_f + dlon_steps * _TILE_DEG, 1)
                if _LAT_MIN <= nlat < _LAT_MAX and _LON_MIN <= nlon < _LON_MAX:
                    candidates.append((nlat, nlon))
        ring += 1

    # Filter to only unanalyzed/unseeded tiles
    with _conn() as con:
        results = []
        seen = set()
        for tlat, tlon in candidates:
            tid = _tile_id(tlat, tlon)
            if tid in seen:
                continue
            seen.add(tid)
            row = con.execute(
                "SELECT status FROM coverage_tiles WHERE tile_id=?", (tid,)
            ).fetchone()
            if row is None or row["status"] == "unanalyzed":
                lat_min, lon_min, lat_max, lon_max = _tile_bounds(tlat, tlon)
                results.append({
                    "tile_id": tid,
                    "lat_min": lat_min,
                    "lon_min": lon_min,
                    "lat_max": lat_max,
                    "lon_max": lon_max,
                    "center_lat": (lat_min + lat_max) / 2,
                    "center_lon": (lon_min + lon_max) / 2,
                })
            if len(results) >= n:
                break
    return results


async def seed_tile(tile: dict[str, Any]) -> int:
    """Seed a tile from FCC + cell tower public data. Returns feature count."""
    from backend.datasources.fcc import search_licenses_near
    from backend.datasources.cell_towers import search_towers_near
    from backend.storage.map_store import upsert_feature

    lat = tile["center_lat"]
    lon = tile["center_lon"]

    licenses, towers = await asyncio.gather(
        search_licenses_near(lat, lon, radius_km=35.0, limit=100),
        search_towers_near(lat, lon, radius_km=35.0),
    )

    count = 0
    for lic in licenses:
        if lic.get("lat") and lic.get("lon"):
            fid = f"fcc_{lic['callsign']}_{lic.get('service','')}"
            props: dict[str, Any] = {
                "id": fid,
                "kind": "infrastructure",
                "name": f"{lic.get('callsign','')} ({lic.get('service','')})",
                "callsign": lic.get("callsign"),
                "licensee_name": lic.get("licensee_name"),
                "service": lic.get("service"),
                "freq_band": (
                    f"{lic['freq_center_mhz']:.1f} MHz"
                    if lic.get("freq_center_mhz") else None
                ),
                "eirp_dbm": lic.get("eirp_dbm"),
                "azimuth_deg": lic.get("azimuth_deg") or 0.0,
                "beamwidth_deg": 360.0,
                "confidence": 1.0,
                "source": "fcc_uls",
                "timestamp": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
            }
            upsert_feature({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lic["lon"], lic["lat"]]},
                "properties": props,
            })
            count += 1

    for tower in towers:
        if tower.get("lat") and tower.get("lon"):
            fid = f"tower_{tower['lat']:.4f}_{tower['lon']:.4f}"
            props = {
                "id": fid,
                "kind": "infrastructure",
                "name": f"{tower.get('radio','?')} tower",
                "freq_band": (
                    f"{tower['freq_mhz']:.0f} MHz" if tower.get("freq_mhz") else "cellular"
                ),
                "confidence": 0.9,
                "source": tower.get("source", "opencellid"),
                "radio": tower.get("radio"),
                "azimuth_deg": 0.0,
                "beamwidth_deg": 360.0,
                "timestamp": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
            }
            upsert_feature({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [tower["lon"], tower["lat"]]},
                "properties": props,
            })
            count += 1

    upsert_tile(
        tile_id=tile["tile_id"],
        lat_min=tile["lat_min"],
        lon_min=tile["lon_min"],
        lat_max=tile["lat_max"],
        lon_max=tile["lon_max"],
        status="seeded",
        feature_count=count,
    )
    return count


async def advance_queue(
    operator_lat: float,
    operator_lon: float,
    n: int = 2,
) -> dict[str, Any]:
    """Seed the next n highest-priority unprocessed tiles."""
    tiles = next_tiles_to_process(operator_lat, operator_lon, n)
    total_features = 0
    processed = []
    for tile in tiles:
        try:
            count = await seed_tile(tile)
            total_features += count
            processed.append({"tile_id": tile["tile_id"], "features_added": count})
        except Exception as exc:
            processed.append({"tile_id": tile["tile_id"], "error": str(exc)})
    return {
        "tiles_processed": len(processed),
        "total_features_added": total_features,
        "tiles": processed,
        "progress": get_coverage_progress(),
    }
