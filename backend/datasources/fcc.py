"""FCC ULS geo-search integration.

Queries the public FCC Universal Licensing System API for licensed transmitters
near a given lat/lon — no API key required.

Note: the /json path-segment form of the endpoint has been retired by the FCC.
We try the dot-notation alternative; if both return 404 we silently return []
since FCC data is supplementary, not critical to operation.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_FCC_GEO_URLS = [
    "https://data.fcc.gov/api/license/searchByGeoSearch.json",
    "https://data.fcc.gov/api/license/searchByGeoSearch/json",
]
_TIMEOUT = 10.0


async def search_licenses_near(
    lat: float,
    lon: float,
    radius_km: float = 50.0,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return FCC licensed transmitters within radius_km of (lat, lon).

    Each result dict includes: callsign, licensee_name, service, status,
    freq_low_mhz, freq_high_mhz, lat, lon, azimuth (where available).
    Returns an empty list on network failure or when FCC API is unavailable.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "radiusInKm": radius_km,
        "limit": limit,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for url in _FCC_GEO_URLS:
            try:
                resp = await client.get(url, params=params)
                if resp.status_code == 404:
                    logger.debug("FCC ULS endpoint 404 (retired): %s", url)
                    continue
                resp.raise_for_status()
                data = resp.json()
                results = []
                for lic in data.get("Licenses", {}).get("License", []) or []:
                    try:
                        results.append(_parse_license(lic))
                    except Exception:
                        continue
                return results
            except httpx.HTTPStatusError as exc:
                logger.warning("FCC ULS HTTP error %s: %s", exc.response.status_code, url)
            except Exception as exc:
                logger.warning("FCC ULS query failed: %s", exc)
                return []
    # All URLs returned 404 — endpoint is gone, no noise needed
    return []


def _parse_license(lic: dict[str, Any]) -> dict[str, Any]:
    loc = lic.get("licenseAttachment", {})
    freq_low = _safe_float(lic.get("frequencyLow"))
    freq_high = _safe_float(lic.get("frequencyHigh"))
    return {
        "callsign": lic.get("callsign", ""),
        "licensee_name": lic.get("licenseeName", ""),
        "service": lic.get("radioServiceCode", ""),
        "status": lic.get("licenseStatusCode", ""),
        "freq_low_mhz": freq_low,
        "freq_high_mhz": freq_high,
        "freq_center_mhz": ((freq_low + freq_high) / 2.0) if freq_low and freq_high else freq_low,
        "lat": _safe_float(lic.get("latitude") or loc.get("latitude")),
        "lon": _safe_float(lic.get("longitude") or loc.get("longitude")),
        "azimuth_deg": _safe_float(lic.get("azimuth")),
        "eirp_dbm": _safe_float(lic.get("eirp")),
        "source": "fcc_uls",
    }


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None
