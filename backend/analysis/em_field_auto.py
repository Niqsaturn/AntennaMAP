"""Automatic EM field and beam pattern overlay generation.

Triggered after any feature is upserted with confidence > 0.5.
Calls em_field_grid() and the array pattern calculators to produce
GeoJSON fill layers stored in map_store and broadcast via SSE.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Any

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.5
_DEFAULT_EIRP_DBM = 43.0   # 20 W typical land mobile
_DEFAULT_BW_DEG = 60.0


def maybe_generate_em_field(feature: dict) -> None:
    """Non-blocking: spawn background thread to compute and store EM overlay."""
    props = feature.get("properties", {})
    if not props:
        return
    confidence = float(props.get("confidence", 0.0) or 0.0)
    if confidence < _CONFIDENCE_THRESHOLD:
        return
    coords = feature.get("geometry", {}).get("coordinates")
    if not coords or len(coords) < 2:
        return
    threading.Thread(
        target=_compute_and_store,
        args=(feature,),
        daemon=True,
    ).start()


def _compute_and_store(feature: dict) -> None:
    props = feature.get("properties", {})
    coords = feature.get("geometry", {}).get("coordinates", [])
    site_lat = float(coords[1])
    site_lon = float(coords[0])
    freq_hz = float(props.get("freq_hz", 100e6) or 100e6)
    feature_id = props.get("id", "")
    azimuth_deg = float(props.get("azimuth_deg") or props.get("bearing_deg") or 0.0)
    beamwidth_deg = float(props.get("beamwidth_deg") or _DEFAULT_BW_DEG)
    if beamwidth_deg >= 360:
        beamwidth_deg = _DEFAULT_BW_DEG

    try:
        geojson = _build_em_polygon(
            lat=site_lat, lon=site_lon,
            freq_hz=freq_hz,
            azimuth_deg=azimuth_deg,
            beamwidth_deg=beamwidth_deg,
            eirp_dbm=float(props.get("eirp_dbm") or _DEFAULT_EIRP_DBM),
        )
        if not geojson:
            return

        _store_em_field(feature_id, geojson)
        _publish_em_event(feature_id, geojson)
    except Exception as exc:
        logger.debug("em_field_auto: %s (feature=%s)", exc, feature_id)


def _build_em_polygon(
    lat: float,
    lon: float,
    freq_hz: float,
    azimuth_deg: float,
    beamwidth_deg: float,
    eirp_dbm: float,
) -> dict | None:
    """Build a wedge/circle GeoJSON polygon for the EM coverage area."""
    try:
        from backend.rf_propagation import em_field_grid, ModelName
        grid = em_field_grid(
            tx_lat=lat, tx_lon=lon,
            frequency_hz=freq_hz,
            tx_eirp_dbm=eirp_dbm,
            model=ModelName.FSPL,
            rx_sensitivity_dbm=-100.0,
            n_radials=24,
            n_rings=8,
        )
        # em_field_grid returns a GeoJSON FeatureCollection; take first feature
        features = grid.get("features", [])
        if features:
            return features[0].get("geometry")
    except Exception:
        pass

    # Fallback: build a simple wedge polygon
    return _wedge_polygon(lat, lon, azimuth_deg, beamwidth_deg, radius_m=_range_estimate_m(freq_hz, eirp_dbm))


def _wedge_polygon(
    lat: float,
    lon: float,
    azimuth_deg: float,
    beamwidth_deg: float,
    radius_m: float,
) -> dict:
    m_per_deg_lat = 111_320.0
    cos_lat = math.cos(math.radians(lat))
    m_per_deg_lon = m_per_deg_lat * max(cos_lat, 1e-6)

    half = beamwidth_deg / 2.0
    start_az = azimuth_deg - half
    end_az = azimuth_deg + half
    n_steps = max(8, int(beamwidth_deg / 5))

    coords = [[lon, lat]]
    for i in range(n_steps + 1):
        az = math.radians(start_az + (end_az - start_az) * i / n_steps)
        dlat = radius_m * math.cos(az) / m_per_deg_lat
        dlon = radius_m * math.sin(az) / m_per_deg_lon
        coords.append([lon + dlon, lat + dlat])
    coords.append([lon, lat])

    return {"type": "Polygon", "coordinates": [coords]}


def _range_estimate_m(freq_hz: float, eirp_dbm: float) -> float:
    """Rough FSPL-based range estimate for polygon sizing."""
    rx_sens_dbm = -100.0
    path_loss_db = eirp_dbm - rx_sens_dbm
    freq_mhz = freq_hz / 1e6
    # FSPL: PL = 20*log10(d) + 20*log10(f_MHz) + 32.44
    log_d = (path_loss_db - 20.0 * math.log10(max(freq_mhz, 1.0)) - 32.44) / 20.0
    dist_km = 10 ** log_d
    return min(max(dist_km * 1000.0, 500.0), 500_000.0)


def _store_em_field(feature_id: str, geometry: dict) -> None:
    try:
        from backend.storage.map_store import upsert_feature
        from datetime import datetime, timezone
        upsert_feature({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "id": f"em_{feature_id}",
                "kind": "em_field",
                "parent_feature_id": feature_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        })
    except Exception as exc:
        logger.debug("em_field_auto: store error: %s", exc)


def _publish_em_event(feature_id: str, geometry: dict) -> None:
    try:
        from backend.foxhunt.auto_loop import event_bus
        from datetime import datetime, timezone
        event_bus.publish({
            "type": "em_field_updated",
            "ts": datetime.now(timezone.utc).isoformat(),
            "feature_id": feature_id,
            "geometry": geometry,
        })
    except Exception:
        pass
