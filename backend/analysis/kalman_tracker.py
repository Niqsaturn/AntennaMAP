"""2D position-only Kalman filter for emitter location smoothing.

Each speculative feature in the map store gets its own PositionKalman instance.
State is serialised to/from JSON so the filter is continuous across server restarts.
"""
from __future__ import annotations

import json
import math


_PROC_NOISE_M = 50.0        # assumed static emitter positional drift (meters / cycle)
_DEG_PER_M_LAT = 1.0 / 111_320.0
_INIT_UNCERTAINTY_M = 5_000.0


def _m_to_deg_lat(meters: float) -> float:
    return meters * _DEG_PER_M_LAT


def _m_to_deg_lon(meters: float, lat_deg: float) -> float:
    cos_lat = math.cos(math.radians(lat_deg))
    return meters * _DEG_PER_M_LAT / max(cos_lat, 1e-6)


class PositionKalman:
    """Scalar 2D position Kalman filter (independent lat / lon channels)."""

    def __init__(
        self,
        init_lat: float,
        init_lon: float,
        init_uncertainty_m: float = _INIT_UNCERTAINTY_M,
    ) -> None:
        self.lat = init_lat
        self.lon = init_lon
        # Variance in degrees² (one per axis, diagonal covariance)
        sigma_lat = _m_to_deg_lat(init_uncertainty_m)
        sigma_lon = _m_to_deg_lon(init_uncertainty_m, init_lat)
        self.P_lat = sigma_lat ** 2
        self.P_lon = sigma_lon ** 2

    def predict(self, dt_seconds: float = 60.0) -> None:
        """Propagate state forward — adds process noise (static emitter model)."""
        q_lat = _m_to_deg_lat(_PROC_NOISE_M) ** 2
        q_lon = _m_to_deg_lon(_PROC_NOISE_M, self.lat) ** 2
        self.P_lat += q_lat
        self.P_lon += q_lon

    def update(
        self,
        meas_lat: float,
        meas_lon: float,
        uncertainty_m: float,
    ) -> tuple[float, float, float]:
        """Fuse a new position measurement.

        Returns (smoothed_lat, smoothed_lon, updated_uncertainty_m).
        """
        sigma_lat = _m_to_deg_lat(uncertainty_m)
        sigma_lon = _m_to_deg_lon(uncertainty_m, self.lat)
        R_lat = sigma_lat ** 2
        R_lon = sigma_lon ** 2

        # Kalman gains
        K_lat = self.P_lat / (self.P_lat + R_lat)
        K_lon = self.P_lon / (self.P_lon + R_lon)

        # State update
        self.lat = self.lat + K_lat * (meas_lat - self.lat)
        self.lon = self.lon + K_lon * (meas_lon - self.lon)

        # Covariance update (Joseph form for numerical stability)
        self.P_lat = (1.0 - K_lat) * self.P_lat
        self.P_lon = (1.0 - K_lon) * self.P_lon

        # Convert posterior variance back to meters
        updated_m = math.sqrt(max(self.P_lat, self.P_lon)) / _DEG_PER_M_LAT
        return round(self.lat, 7), round(self.lon, 7), round(updated_m, 1)

    def to_json(self) -> str:
        return json.dumps({
            "lat": self.lat,
            "lon": self.lon,
            "P_lat": self.P_lat,
            "P_lon": self.P_lon,
        })

    @classmethod
    def from_json(cls, data: str) -> "PositionKalman":
        d = json.loads(data)
        inst = cls.__new__(cls)
        inst.lat = d["lat"]
        inst.lon = d["lon"]
        inst.P_lat = d["P_lat"]
        inst.P_lon = d["P_lon"]
        return inst

    @classmethod
    def for_feature(cls, feature_id: str, lat: float, lon: float) -> "PositionKalman":
        """Load existing Kalman state from map_store, or create a new one."""
        try:
            from backend.storage.map_store import get_kalman_state
            state_json = get_kalman_state(feature_id)
            if state_json:
                return cls.from_json(state_json)
        except Exception:
            pass
        return cls(lat, lon)

    def save(self, feature_id: str) -> None:
        """Persist Kalman state to map_store."""
        try:
            from backend.storage.map_store import set_kalman_state
            set_kalman_state(feature_id, self.to_json())
        except Exception:
            pass
