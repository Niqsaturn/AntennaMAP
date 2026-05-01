"""Cell tower data via OpenCelliD / Mozilla Location Service.

Queries public cell tower APIs to locate cellular base stations near a given
lat/lon.  Falls back gracefully when network is unavailable or an API key is
not configured.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# OpenCelliD public API — set OPENCELLID_API_KEY env var for full access.
# Without a key the request is rejected; we fall back to Mozilla.
_OPENCELLID_URL = "https://opencellid.org/cell/getInArea"
_MOZILLA_URL = "https://location.services.mozilla.com/v1/geolocate"
_TIMEOUT = 10.0

# ITU band → approximate frequency MHz lookup
_BAND_FREQ: dict[str, float] = {
    "700": 746.0,
    "800": 824.0,
    "850": 850.0,
    "900": 935.0,
    "1700": 1710.0,
    "1800": 1805.0,
    "1900": 1930.0,
    "2100": 2110.0,
    "2300": 2300.0,
    "2500": 2500.0,
    "2600": 2620.0,
    "3500": 3500.0,
    "5g_sub6": 3700.0,
}


async def search_towers_near(
    lat: float,
    lon: float,
    radius_km: float = 25.0,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Return cell towers within radius_km of (lat, lon).

    Tries OpenCelliD first (if api_key provided), then Mozilla fallback.
    Returns an empty list on all failures.
    """
    key = api_key or os.environ.get("OPENCELLID_API_KEY", "")
    if key:
        towers = await _opencellid_search(lat, lon, radius_km, key)
        if towers:
            return towers
    return await _mozilla_search(lat, lon)


async def _opencellid_search(
    lat: float, lon: float, radius_km: float, api_key: str
) -> list[dict[str, Any]]:
    params = {
        "key": api_key,
        "BBOX": f"{lat - radius_km/111.32},{lon - radius_km/111.32},{lat + radius_km/111.32},{lon + radius_km/111.32}",
        "format": "json",
        "limit": 200,
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_OPENCELLID_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        return [_parse_opencellid(c) for c in data.get("cells", [])]
    except Exception as exc:
        logger.debug("OpenCelliD query failed: %s", exc)
        return []


async def _mozilla_search(lat: float, lon: float) -> list[dict[str, Any]]:
    """Mozilla fallback — returns a single estimated position, not tower list."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                _MOZILLA_URL,
                json={"considerIp": True, "wifiAccessPoints": []},
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        loc = data.get("location", {})
        return [{
            "lat": loc.get("lat", lat),
            "lon": loc.get("lng", lon),
            "accuracy_m": data.get("accuracy"),
            "source": "mozilla_location",
            "radio": "unknown",
            "freq_mhz": None,
        }]
    except Exception as exc:
        logger.debug("Mozilla location query failed: %s", exc)
        return []


def _parse_opencellid(cell: dict[str, Any]) -> dict[str, Any]:
    radio = str(cell.get("radio", "")).upper()
    band = str(cell.get("averageSignalStrength", ""))
    return {
        "lat": _safe_float(cell.get("lat")),
        "lon": _safe_float(cell.get("lon")),
        "mcc": cell.get("mcc"),
        "mnc": cell.get("mnc"),
        "lac": cell.get("lac"),
        "cell_id": cell.get("cellid"),
        "radio": radio,
        "samples": cell.get("samples"),
        "freq_mhz": _band_to_freq(radio, cell.get("averageSignalStrength")),
        "rssi_dbm": _safe_float(cell.get("averageSignalStrength")),
        "source": "opencellid",
    }


def _band_to_freq(radio: str, _band: Any) -> float | None:
    if radio in ("LTE", "NR"):
        return _BAND_FREQ.get("1800")
    if radio == "UMTS":
        return _BAND_FREQ.get("2100")
    if radio == "GSM":
        return _BAND_FREQ.get("900")
    return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None
