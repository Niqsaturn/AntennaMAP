"""Speculative antenna array feature generator.

Converts raw AI signal detections into GeoJSON features with kind='speculative',
applies antenna classification, enriches with overlay geometries, and uses
real RSSI-based range estimation instead of hardcoded distances.
"""
from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime, timezone
from typing import Any

from backend.analysis.signal_detector import SignalDetection
from backend.geometry.rf_overlays import build_overlay_geometries
from backend.rf.antenna_classifier import classify_antenna

_log = logging.getLogger(__name__)

# Band-appropriate fallback distances when RSSI range estimation unavailable
_BAND_FALLBACK_M: dict[str, float] = {
    "ELF": 200_000, "SLF": 200_000, "ULF": 100_000,
    "VLF": 100_000, "LF": 50_000,
    "AM Broadcast": 30_000, "MF": 30_000,
    "HF / Shortwave": 15_000, "HF": 15_000,
    "VHF": 5_000, "Aviation VHF": 5_000, "VHF Land Mobile": 3_000,
    "FM Broadcast": 20_000,
    "UHF": 1_000, "UHF Land Mobile": 800, "Cellular / LTE": 500,
    "2.4 GHz": 300, "5.8 GHz": 150, "SHF": 200, "EHF": 50,
}
_DEFAULT_FALLBACK_M = 2_000.0


_BEARING_CLUSTER_DEG = 15.0  # detections within 15° of each other → same cluster


def _bearing_diff(a: float, b: float) -> float:
    d = abs((a - b + 180) % 360 - 180)
    return d


def _cluster_detections(
    detections: list[SignalDetection],
) -> list[list[SignalDetection]]:
    """Group detections by (freq_band, bearing proximity)."""
    clusters: list[list[SignalDetection]] = []
    used = [False] * len(detections)
    for i, d in enumerate(detections):
        if used[i]:
            continue
        cluster = [d]
        used[i] = True
        for j in range(i + 1, len(detections)):
            if used[j]:
                continue
            other = detections[j]
            if other.freq_band != d.freq_band:
                continue
            if d.bearing_deg is not None and other.bearing_deg is not None:
                if _bearing_diff(d.bearing_deg, other.bearing_deg) > _BEARING_CLUSTER_DEG:
                    continue
            cluster.append(other)
            used[j] = True
        clusters.append(cluster)
    return clusters


def _estimate_position_from_bearing(
    operator_lat: float,
    operator_lon: float,
    bearing_deg: float,
    estimated_distance_m: float,
) -> tuple[float, float]:
    """Dead-reckoning: walk bearing_deg from operator at estimated_distance_m."""
    R = 6371000.0
    lat1 = math.radians(operator_lat)
    lon1 = math.radians(operator_lon)
    br = math.radians(bearing_deg)
    d_R = estimated_distance_m / R
    lat2 = math.asin(math.sin(lat1) * math.cos(d_R) + math.cos(lat1) * math.sin(d_R) * math.cos(br))
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(d_R) * math.cos(lat1),
        math.cos(d_R) - math.sin(lat1) * math.sin(lat2),
    )
    return round(math.degrees(lat2), 6), round(math.degrees(lon2), 6)


def _estimated_range_m(
    rep: SignalDetection,
    cluster: list[SignalDetection],
) -> float:
    """Compute best available range estimate for this detection cluster.

    Priority:
    1. RSSI-based inversion using range_estimator (requires rssi + freq)
    2. Band-appropriate fallback distance
    """
    # Try to extract RSSI/freq from notes (signal_detector embeds them)
    avg_rssi: float | None = None
    avg_snr: float | None = None
    freq_mhz: float | None = None

    for d in cluster:
        notes = d.notes or ""
        if "snr=" in notes:
            try:
                snr_str = notes.split("snr=")[1].split("dB")[0]
                avg_snr = float(snr_str)
            except Exception:
                pass
        if "range≈" in notes:
            # Already computed by signal_detector; extract it
            try:
                range_str = notes.split("range≈")[1].split("km")[0]
                return float(range_str) * 1000.0
            except Exception:
                pass

    # Try RSSI-based inversion
    if avg_rssi is not None and freq_mhz is not None:
        try:
            from backend.analysis.range_estimator import estimate_range_km
            dist_km, _ = estimate_range_km(
                rssi_dbm=avg_rssi,
                freq_mhz=freq_mhz,
                snr_db=avg_snr or 10.0,
                bandwidth_hz=25_000.0,
            )
            return dist_km * 1000.0
        except Exception:
            pass

    # Band-appropriate fallback
    freq_band = rep.freq_band or ""
    return _BAND_FALLBACK_M.get(freq_band, _DEFAULT_FALLBACK_M)


def _feature_id(lat: float, lon: float, freq_band: str) -> str:
    key = f"{lat:.4f}_{lon:.4f}_{freq_band}"
    return "spec_" + hashlib.md5(key.encode()).hexdigest()[:10]


def speculate_from_detections(
    detections: list[SignalDetection],
    operator_lat: float,
    operator_lon: float,
    existing_features: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert AI detections into kind='speculative' GeoJSON features.

    For each detection cluster:
    - Use AI-provided lat/lon if available
    - Otherwise dead-reckon from bearing at 2 km range
    - Classify antenna type using existing classifier
    - Enrich with overlay geometries (ray, sector, ellipse)
    """
    existing_features = existing_features or []
    clusters = _cluster_detections(detections)
    now = datetime.now(timezone.utc).isoformat()
    features: list[dict[str, Any]] = []

    for cluster in clusters:
        # Representative detection for the cluster (highest confidence)
        rep = max(cluster, key=lambda d: d.confidence)

        # Determine position
        range_m = _estimated_range_m(rep, cluster)
        if rep.estimated_lat is not None and rep.estimated_lon is not None:
            lat, lon = rep.estimated_lat, rep.estimated_lon
        elif rep.bearing_deg is not None:
            lat, lon = _estimate_position_from_bearing(
                operator_lat, operator_lon, rep.bearing_deg, range_m
            )
        else:
            continue

        # Classify antenna
        props_for_classifier = {
            "directionality": (
                "omni" if rep.antenna_type == "omni" else
                "sector" if rep.antenna_type == "sector_panel" else "unknown"
            ),
            "azimuth_deg": rep.azimuth_deg,
            "rf_min_mhz": None,
            "rf_max_mhz": None,
        }
        telemetry_for_classifier = [
            {"bearing_deg": d.bearing_deg} for d in cluster if d.bearing_deg is not None
        ]
        cls = classify_antenna(props_for_classifier, telemetry_for_classifier)

        confidence = round(sum(d.confidence for d in cluster) / len(cluster), 4)
        beamwidth = rep.beamwidth_deg or cls.estimated_elements.get("estimated_beamwidth_deg", 80)
        azimuth = rep.azimuth_deg or cls.estimated_elements.get("array_orientation_deg") or (
            rep.bearing_deg or 0.0
        )
        fid = _feature_id(lat, lon, rep.freq_band)

        props: dict[str, Any] = {
            "id": fid,
            "signal_id": rep.signal_id,
            "kind": "speculative",
            "name": f"Speculative {cls.antenna_type} ({rep.freq_band})",
            "freq_band": rep.freq_band,
            "antenna_type": cls.antenna_type,
            "classifier_confidence": cls.confidence,
            "confidence": confidence,
            "analysis_count": len(cluster),
            "azimuth_deg": round(float(azimuth), 1),
            "beamwidth_deg": round(float(beamwidth), 1),
            "ray_length_m": max(200, int(range_m * 0.35)),
            "wedge_radius_m": max(300, int(range_m * 0.45)),
            "estimated_range_m": round(range_m),
            "notes": rep.notes,
            "timestamp": now,
            **{k: v for k, v in cls.estimated_elements.items()},
        }

        feature: dict[str, Any] = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props,
        }

        # Enrich with overlay geometries
        try:
            feature = _enrich(feature)
        except Exception as exc:
            _log.warning("overlay geometry enrichment failed for %s: %s", fid, exc)

        features.append(feature)

    return features


def _enrich(feature: dict[str, Any]) -> dict[str, Any]:
    enriched = {**feature}
    props = {**feature.get("properties", {})}
    if feature.get("geometry", {}).get("type") == "Point":
        props["overlay_geometries"] = build_overlay_geometries(
            {"type": "Feature", "geometry": feature["geometry"], "properties": props}
        )
    enriched["properties"] = props
    return enriched
