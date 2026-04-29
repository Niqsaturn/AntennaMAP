from __future__ import annotations

from statistics import mean
from typing import Any


def _deterministic_confidence(window: list[dict[str, Any]]) -> float:
    if not window:
        return 0.0
    snr = mean(row.get("snr_db", 0.0) for row in window)
    return round(max(0.0, min(1.0, (snr + 20.0) / 80.0)), 4)


def estimate_single_operator(window: list[dict[str, Any]]) -> dict[str, Any]:
    """Sequential deterministic baseline estimator.

    Uses simple weighted centroid where stronger RSSI/SNR carries more weight.
    """
    if not window:
        return {
            "method": "single_triangulation_baseline",
            "estimate": None,
            "num_samples": 0,
            "confidence": 0.0,
        }

    weighted = []
    for row in window:
        rssi = row.get("rssi_dbm", -120.0)
        snr = row.get("snr_db", -20.0)
        weight = max(0.01, (snr + 20.0) / 80.0) * max(0.01, (140.0 + rssi) / 120.0)
        weighted.append((row["lat"], row["lon"], weight))

    weight_sum = sum(w for _, _, w in weighted)
    est_lat = sum(lat * w for lat, _, w in weighted) / weight_sum
    est_lon = sum(lon * w for _, lon, w in weighted) / weight_sum

    return {
        "method": "single_triangulation_baseline",
        "estimate": {"lat": round(est_lat, 7), "lon": round(est_lon, 7)},
        "num_samples": len(window),
        "confidence": _deterministic_confidence(window),
    }
