"""Multipath and Doppler detection from spectral peaks.

When a PSD shows two closely spaced peaks at nominally the same frequency,
this module decides whether the separation is caused by Doppler shift from
a moving object or by multipath propagation delay.
"""
from __future__ import annotations

from backend.analysis.waterfall_analyzer import SpectralPeak

_C_M_PER_S = 299_792_458.0   # speed of light


def detect_multipath(
    peaks: list[SpectralPeak],
    freq_hz: float,
) -> dict | None:
    """Analyse a pair of nearby spectral peaks for multipath or Doppler signature.

    Returns a dict with keys:
      - path_length_diff_m: additional path length of reflected ray
      - velocity_estimate_mps: apparent velocity (non-None only for Doppler case)
      - interpretation: "doppler" | "multipath" | "unknown"
      - confidence: 0.0–1.0

    Returns None if no pair of similar-frequency peaks is found.
    """
    if len(peaks) < 2 or freq_hz <= 0:
        return None

    # Find the pair with smallest frequency separation
    best_pair = None
    min_sep = float("inf")
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            sep = abs(peaks[i].center_freq_hz - peaks[j].center_freq_hz)
            if sep < min_sep:
                min_sep = sep
                best_pair = (peaks[i], peaks[j])

    if best_pair is None or min_sep == 0:
        return None

    delta_f = min_sep  # Hz
    lambda_m = _C_M_PER_S / freq_hz

    # Power similarity check — close power levels suggest coherent interference
    power_diff_db = abs(best_pair[0].peak_dbm - best_pair[1].peak_dbm)
    power_similar = power_diff_db < 10.0

    # Thresholds
    if delta_f < 1.0 and power_similar:
        # Very small Δf — most likely Doppler from slow-moving reflector
        velocity_mps = (delta_f * lambda_m) / 2.0   # monostatic Doppler: Δf = 2v/λ
        path_length = _C_M_PER_S / max(delta_f, 1e-6)   # ill-defined at ~0 Hz
        confidence = min(0.7, 0.3 + power_similar * 0.4)
        return {
            "path_length_diff_m": round(path_length, 1),
            "velocity_estimate_mps": round(velocity_mps, 3),
            "delta_f_hz": round(delta_f, 4),
            "interpretation": "doppler",
            "confidence": round(confidence, 3),
        }

    if 1.0 <= delta_f <= 10_000 and power_similar:
        # Multipath: delay spread Δt = 1/Δf → path length difference Δd = c/Δf
        path_diff_m = _C_M_PER_S / delta_f
        confidence = min(0.75, 0.4 + (1.0 - min(delta_f / 10_000, 1.0)) * 0.35)
        return {
            "path_length_diff_m": round(path_diff_m, 1),
            "velocity_estimate_mps": None,
            "delta_f_hz": round(delta_f, 2),
            "interpretation": "multipath",
            "confidence": round(confidence, 3),
        }

    # Separation too large or powers dissimilar — likely independent signals
    return {
        "path_length_diff_m": None,
        "velocity_estimate_mps": None,
        "delta_f_hz": round(delta_f, 2),
        "interpretation": "unknown",
        "confidence": 0.1,
    }
