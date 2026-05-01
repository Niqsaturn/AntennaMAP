"""Bayesian 2D posterior grid for emitter location estimation.

Each observation progressively updates a discrete log-posterior over a
geographic grid.  Supports:
  - RSSI ring constraints  (Gaussian around the estimated range circle)
  - Bearing wedge constraints (von Mises distribution around bearing line)

The grid is maintained in log-probability space for numerical stability.
`posterior()` normalises and returns a GeoJSON FeatureCollection of grid
cells suitable for heatmap rendering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RSSIUpdate:
    obs_lat: float
    obs_lon: float
    rssi_dbm: float
    freq_hz: float
    eirp_dbm: float = 47.0      # typical EIRP; caller should supply best estimate


@dataclass
class BearingUpdate:
    obs_lat: float
    obs_lon: float
    bearing_deg: float
    sigma_deg: float = 10.0


_DEG_TO_M_LAT = 111_320.0


def _fspl_range_m(rssi_dbm: float, freq_hz: float, eirp_dbm: float) -> float:
    """Invert FSPL to get estimated distance in metres."""
    pl_db = eirp_dbm - rssi_dbm
    if pl_db <= 0:
        return 10.0
    f_mhz = max(freq_hz, 1.0) / 1e6
    exp = (pl_db - 32.44 - 20 * math.log10(f_mhz)) / 20.0
    return max(10.0, (10 ** exp) * 1000.0)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _bearing_to_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing from (lat1,lon1) to (lat2,lon2) in degrees [0,360)."""
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon))
    return math.degrees(math.atan2(y, x)) % 360.0


class BayesGrid:
    """Discrete 2D log-posterior P(emitter at cell | observations).

    Parameters
    ----------
    center_lat, center_lon : float
        Centre of the grid (WGS-84 degrees).
    radius_km : float
        Half-extent of the grid in km.
    resolution_km : float
        Size of each grid cell in km.
    """

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float = 50.0,
        resolution_km: float = 2.0,
    ) -> None:
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius_km = radius_km
        self.resolution_km = resolution_km

        n_cells = max(3, int(2 * radius_km / resolution_km))
        self._n = n_cells
        # Log-prior: uniform over all cells
        self._log_p = np.zeros((n_cells, n_cells), dtype=np.float64)
        self._observation_count = 0

        # Build lat/lon grid
        step_lat = resolution_km / 111.32
        cos_lat = max(math.cos(math.radians(center_lat)), 1e-6)
        step_lon = resolution_km / (111.32 * cos_lat)
        half = (n_cells - 1) / 2.0
        self._lats = center_lat + (np.arange(n_cells) - half) * step_lat
        self._lons = center_lon + (np.arange(n_cells) - half) * step_lon

        # Meshgrid for vectorised updates
        self._lon_grid, self._lat_grid = np.meshgrid(self._lons, self._lats)

    def update_rssi(self, update: RSSIUpdate) -> None:
        """Apply a ring constraint from an RSSI observation.

        Models the distance uncertainty as a Gaussian centred on the
        FSPL-inverted range with sigma = 30% of that range.
        """
        estimated_m = _fspl_range_m(update.rssi_dbm, update.freq_hz, update.eirp_dbm)
        sigma_m = max(estimated_m * 0.30, 500.0)

        # Vectorised haversine to every grid cell
        dlat = np.radians(self._lat_grid - update.obs_lat)
        dlon = np.radians(self._lon_grid - update.obs_lon)
        a = (np.sin(dlat / 2) ** 2
             + math.cos(math.radians(update.obs_lat))
             * np.cos(np.radians(self._lat_grid))
             * np.sin(dlon / 2) ** 2)
        dist_m = 6_371_000.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        log_likelihood = -0.5 * ((dist_m - estimated_m) / sigma_m) ** 2
        self._log_p += log_likelihood
        self._observation_count += 1

    def update_bearing(self, update: BearingUpdate) -> None:
        """Apply a wedge constraint from a bearing observation.

        Models bearing uncertainty as a wrapped Gaussian (von Mises
        approximation) with the given sigma_deg.
        """
        kappa = (180.0 / (math.pi * max(update.sigma_deg, 1.0))) ** 2
        kappa = min(kappa, 500.0)  # cap to avoid numerical overflow

        # Bearing from observer to each grid cell
        dlon = np.radians(self._lon_grid - update.obs_lon)
        y = np.sin(dlon) * np.cos(np.radians(self._lat_grid))
        x = (math.cos(math.radians(update.obs_lat))
             * np.sin(np.radians(self._lat_grid))
             - math.sin(math.radians(update.obs_lat))
             * np.cos(np.radians(self._lat_grid))
             * np.cos(dlon))
        cell_bearing_rad = np.arctan2(y, x)
        obs_bearing_rad = math.radians(update.bearing_deg)

        # Angular difference (wrapped)
        diff = cell_bearing_rad - obs_bearing_rad
        diff = np.arctan2(np.sin(diff), np.cos(diff))   # wrap to (-π, π]

        log_likelihood = kappa * np.cos(diff)   # von Mises log-likelihood (unnormalised)
        self._log_p += log_likelihood
        self._observation_count += 1

    def peak_estimate(self) -> tuple[float, float, float]:
        """Return (lat, lon, probability) of the MAP estimate cell."""
        idx = np.unravel_index(np.argmax(self._log_p), self._log_p.shape)
        prob = float(np.exp(self._log_p[idx] - np.max(self._log_p)))
        return float(self._lats[idx[0]]), float(self._lons[idx[1]]), prob

    def posterior_geojson(self, top_k: int = 500) -> dict[str, Any]:
        """Return a GeoJSON FeatureCollection of the normalised posterior.

        Each feature is a grid-cell square with a `probability` property
        suitable for MapLibre fill-color / heatmap rendering.

        Only the top_k cells by probability are returned to keep the
        response payload manageable.
        """
        # Numerically stable softmax normalisation
        lp = self._log_p - np.max(self._log_p)
        p = np.exp(lp)
        p /= p.sum()

        half_lat = (self.resolution_km / 111.32) / 2.0
        cos_lat = max(math.cos(math.radians(self.center_lat)), 1e-6)
        half_lon = (self.resolution_km / (111.32 * cos_lat)) / 2.0

        # Flatten and take top-k cells
        flat_p = p.ravel()
        indices = np.argsort(flat_p)[::-1][:top_k]

        features = []
        for flat_idx in indices:
            row, col = divmod(int(flat_idx), self._n)
            lat = float(self._lats[row])
            lon = float(self._lons[col])
            prob = float(flat_p[flat_idx])
            if prob < 1e-9:
                continue
            # Cell polygon (square)
            coords = [[
                [lon - half_lon, lat - half_lat],
                [lon + half_lon, lat - half_lat],
                [lon + half_lon, lat + half_lat],
                [lon - half_lon, lat + half_lat],
                [lon - half_lon, lat - half_lat],
            ]]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": coords},
                "properties": {
                    "probability": round(prob, 8),
                    "log_p": round(float(self._log_p[row, col]), 4),
                },
            })

        peak_lat, peak_lon, _ = self.peak_estimate()
        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "center_lat": self.center_lat,
                "center_lon": self.center_lon,
                "radius_km": self.radius_km,
                "resolution_km": self.resolution_km,
                "observation_count": self._observation_count,
                "peak_lat": round(peak_lat, 6),
                "peak_lon": round(peak_lon, 6),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise grid state for caching between API calls."""
        return {
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "radius_km": self.radius_km,
            "resolution_km": self.resolution_km,
            "log_p": self._log_p.tolist(),
            "observation_count": self._observation_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BayesGrid":
        grid = cls(d["center_lat"], d["center_lon"], d["radius_km"], d["resolution_km"])
        grid._log_p = np.array(d["log_p"], dtype=np.float64)
        grid._observation_count = d.get("observation_count", 0)
        return grid


# ── Module-level in-memory grid cache ────────────────────────────────────────
# Keyed by (center_lat_rounded, center_lon_rounded, radius_km, resolution_km)
_grid_cache: dict[str, BayesGrid] = {}


def get_or_create_grid(
    center_lat: float,
    center_lon: float,
    radius_km: float = 50.0,
    resolution_km: float = 2.0,
) -> BayesGrid:
    """Return a cached BayesGrid or create a new one."""
    key = f"{center_lat:.3f}_{center_lon:.3f}_{radius_km:.1f}_{resolution_km:.1f}"
    if key not in _grid_cache:
        _grid_cache[key] = BayesGrid(center_lat, center_lon, radius_km, resolution_km)
    return _grid_cache[key]


def reset_grid(
    center_lat: float,
    center_lon: float,
    radius_km: float = 50.0,
    resolution_km: float = 2.0,
) -> BayesGrid:
    """Clear and reinitialise the cached grid for this region."""
    key = f"{center_lat:.3f}_{center_lon:.3f}_{radius_km:.1f}_{resolution_km:.1f}"
    _grid_cache[key] = BayesGrid(center_lat, center_lon, radius_km, resolution_km)
    return _grid_cache[key]
