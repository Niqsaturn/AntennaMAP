"""Curated database of major public earth stations and satellite ground sites.

Sources: FCC earth station ULS filings, NASA public data, NOAA/GOES ground
network documentation, publicly known teleport facilities.
"""
from __future__ import annotations

from typing import Any

# Each entry: name, lat, lon, type, freq_band, operator
EARTH_STATIONS: list[dict[str, Any]] = [
    # NASA Deep Space Network
    {"name": "DSN Goldstone", "lat": 35.4267, "lon": -116.8900,
     "type": "deep_space", "freq_band": "S/X/Ka", "operator": "NASA/JPL"},
    {"name": "DSN Madrid", "lat": 40.4314, "lon": -4.2481,
     "type": "deep_space", "freq_band": "S/X/Ka", "operator": "NASA/JPL"},
    {"name": "DSN Canberra", "lat": -35.4014, "lon": 148.9816,
     "type": "deep_space", "freq_band": "S/X/Ka", "operator": "NASA/JPL"},

    # NOAA GOES uplink / satellite operations
    {"name": "NESDIS Wallops CDA", "lat": 37.9401, "lon": -75.4664,
     "type": "meteorological", "freq_band": "S/L", "operator": "NOAA/NESDIS"},
    {"name": "NESDIS Fairbanks", "lat": 64.9718, "lon": -147.5060,
     "type": "meteorological", "freq_band": "S/L", "operator": "NOAA/NESDIS"},

    # GPS/GNSS master control
    {"name": "GPS Master Control (Schriever)", "lat": 38.8047, "lon": -104.5240,
     "type": "gnss_control", "freq_band": "L", "operator": "USSF/50SW"},
    {"name": "GPS Ground Antenna (Ascension)", "lat": -7.9697, "lon": -14.3940,
     "type": "gnss_uplink", "freq_band": "L", "operator": "USSF"},
    {"name": "GPS Ground Antenna (Diego Garcia)", "lat": -7.3144, "lon": 72.4230,
     "type": "gnss_uplink", "freq_band": "L", "operator": "USSF"},
    {"name": "GPS Ground Antenna (Kwajalein)", "lat": 9.3955, "lon": 167.4779,
     "type": "gnss_uplink", "freq_band": "L", "operator": "USSF"},
    {"name": "GPS Ground Antenna (Cape Canaveral)", "lat": 28.4889, "lon": -80.5778,
     "type": "gnss_uplink", "freq_band": "L", "operator": "USSF"},

    # Iridium gateway
    {"name": "Iridium Gateway Tempe AZ", "lat": 33.4255, "lon": -111.9400,
     "type": "leo_gateway", "freq_band": "Ka/L", "operator": "Iridium"},

    # Known commercial teleports (publicly documented)
    {"name": "Intelsat Clarksburg", "lat": 39.1434, "lon": -77.2219,
     "type": "commercial_teleport", "freq_band": "C/Ku/Ka", "operator": "Intelsat"},
    {"name": "Teleport Woodbine NJ", "lat": 39.2451, "lon": -74.8096,
     "type": "commercial_teleport", "freq_band": "C/Ku", "operator": "SES"},
    {"name": "ViaSat Carlsbad CA", "lat": 33.1338, "lon": -117.2892,
     "type": "gateway", "freq_band": "Ka", "operator": "Viasat"},

    # GOES East/West uplink (Wallops + Boulder)
    {"name": "GOES Uplink Boulder CO", "lat": 40.0150, "lon": -105.2705,
     "type": "meteorological", "freq_band": "S", "operator": "NOAA"},

    # Starlink gateway examples (public FCC filings)
    {"name": "Starlink Gateway (Brewster WA)", "lat": 48.0923, "lon": -119.7779,
     "type": "leo_gateway", "freq_band": "Ku/Ka/V", "operator": "SpaceX"},
    {"name": "Starlink Gateway (Utqiagvik AK)", "lat": 71.2906, "lon": -156.7887,
     "type": "leo_gateway", "freq_band": "Ku/Ka", "operator": "SpaceX"},

    # Naval / government (public record)
    {"name": "Naval Research Lab (DC)", "lat": 38.8199, "lon": -77.0266,
     "type": "research", "freq_band": "Various", "operator": "NRL/USN"},
    {"name": "AFSCN Vandenberg", "lat": 34.7420, "lon": -120.5724,
     "type": "military_satellite", "freq_band": "S/C", "operator": "USSF"},
]


def get_stations_near(lat: float, lon: float, radius_km: float = 500.0) -> list[dict[str, Any]]:
    """Return earth stations within radius_km of (lat, lon)."""
    import math
    R = 6371.0
    result = []
    for es in EARTH_STATIONS:
        dlat = math.radians(es["lat"] - lat)
        dlon = math.radians(es["lon"] - lon)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat)) * math.cos(math.radians(es["lat"])) *
             math.sin(dlon / 2) ** 2)
        dist_km = 2 * R * math.asin(math.sqrt(a))
        if dist_km <= radius_km:
            result.append({**es, "distance_km": round(dist_km, 1)})
    return sorted(result, key=lambda x: x["distance_km"])
