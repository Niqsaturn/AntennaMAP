"""Mathematical spectrum generator for software-defined virtual receivers.

All spectrum data is computed from propagation models and public RF data — no
physical hardware connections are needed or used.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def thermal_noise_floor_dbm(bandwidth_hz: float, noise_figure_db: float = 5.0) -> float:
    """Thermal noise floor: kTB + noise figure."""
    ktb = -174.0 + 10.0 * math.log10(max(bandwidth_hz, 1.0))
    return ktb + noise_figure_db


def fspl_db(freq_mhz: float, distance_km: float) -> float:
    d = max(distance_km, 0.001)
    return 32.44 + 20.0 * math.log10(freq_mhz) + 20.0 * math.log10(d)


def computed_psd(
    center_freq_hz: float,
    sample_rate_hz: float,
    n_bins: int = 64,
    known_signals: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> list[float]:
    """Return a synthetic PSD (dBm) array for a virtual receiver.

    known_signals: list of {"freq_hz", "power_dbm", "bandwidth_hz"} dicts
    config keys: noise_figure_db, noise_floor_dbm
    """
    config = config or {}
    bin_bw = max(sample_rate_hz / n_bins, 1.0)
    noise_fig = float(config.get("noise_figure_db", 5.0))
    floor = thermal_noise_floor_dbm(bin_bw, noise_fig)
    if "noise_floor_dbm" in config:
        floor = max(floor, float(config["noise_floor_dbm"]))

    # Reproducible per-frequency noise seed so the same config always returns
    # the same baseline (deterministic for testing / caching).
    seed = int(center_freq_hz / 1e3) % (2**31)
    rng = np.random.default_rng(seed)
    psd = floor + rng.normal(0.0, 1.8, n_bins).astype(float)

    for sig in known_signals or []:
        freq_hz = float(sig.get("freq_hz", center_freq_hz))
        power_dbm = float(sig.get("power_dbm", -80.0))
        bw_hz = float(sig.get("bandwidth_hz", sample_rate_hz / 16))
        offset_bins = (freq_hz - center_freq_hz) / bin_bw
        center_bin = n_bins // 2 + int(round(offset_bins))
        half_span = max(1, int(bw_hz / bin_bw / 2))
        for b in range(max(0, center_bin - half_span), min(n_bins, center_bin + half_span + 1)):
            roll = 1.0 - abs(b - center_bin) / max(1.0, half_span + 1)
            db_roll = 20.0 * math.log10(max(roll, 1e-3))
            psd[b] = max(psd[b], power_dbm + db_roll)

    return [round(float(v), 2) for v in psd]


def psd_to_metrics(psd_db: list[float]) -> tuple[float, float]:
    """Derive RSSI and SNR from a PSD array."""
    if not psd_db:
        return -100.0, 0.0
    arr = sorted(psd_db)
    noise_est = float(np.mean(arr[: max(1, len(arr) // 4)]))
    rssi = round(float(np.mean(arr)), 2)
    snr = round(max(0.0, max(psd_db) - noise_est), 2)
    return rssi, snr
