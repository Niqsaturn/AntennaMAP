"""Speculative antenna array feature generator.

Converts raw AI signal detections into GeoJSON features with kind='speculative',
applies antenna classification, and enriches with overlay geometries.
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any

from backend.analysis.signal_detector import SignalDetection
from backend.geometry.rf_overlays import build_overlay_geometries
from backend.rf.antenna_classifier import classify_antenna


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
    estimated_distance_m: float = 2000.0,
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
        if rep.estimated_lat is not None and rep.estimated_lon is not None:
            lat, lon = rep.estimated_lat, rep.estimated_lon
        elif rep.bearing_deg is not None:
            lat, lon = _estimate_position_from_bearing(
                operator_lat, operator_lon, rep.bearing_deg
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
            "kind": "speculative",
            "name": f"Speculative {cls.antenna_type} ({rep.freq_band})",
            "freq_band": rep.freq_band,
            "antenna_type": cls.antenna_type,
            "classifier_confidence": cls.confidence,
            "confidence": confidence,
            "analysis_count": len(cluster),
            "azimuth_deg": round(float(azimuth), 1),
            "beamwidth_deg": round(float(beamwidth), 1),
            "ray_length_m": 750,
            "wedge_radius_m": 900,
            "estimated_range_m": 2000,
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
        except Exception:
            pass

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
