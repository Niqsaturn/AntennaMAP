from __future__ import annotations

from typing import Any


def build_training_rows(samples: list[dict[str, Any]], window_size: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    for idx in range(window_size, len(samples) + 1):
        window = samples[idx - window_size : idx]
        rows.append(
            {
                "window_start": window[0]["timestamp"],
                "window_end": window[-1]["timestamp"],
                "sample_count": len(window),
                "avg_snr_db": sum(s.get("snr_db", 0.0) for s in window) / len(window),
                "avg_rssi_dbm": sum(s.get("rssi_dbm", 0.0) for s in window) / len(window),
                "window": window,
            }
        )
    return rows
