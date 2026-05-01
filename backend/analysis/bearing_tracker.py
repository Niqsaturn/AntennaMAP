"""Synthesized bearing accuracy model.

Estimates bearing measurement uncertainty from SDR type, SNR, frequency, and
movement geometry.  Also provides a SNR-weighted circular mean of bearings that
handles 359° → 0° wrap correctly.
"""
from __future__ import annotations

import math


# SDR quality multipliers (lower = more accurate)
_SDR_QUALITY: dict[str, float] = {
    "kiwisdr": 0.85,
    "airspy": 0.80,
    "rtlsdr": 1.00,
    "hackrf": 1.10,
    "default": 1.00,
}


def bearing_uncertainty_deg(
    snr_db: float,
    freq_mhz: float,
    bandwidth_hz: float,
    sdr_type: str = "default",
    movement_geometry: float = 0.5,
) -> float:
    """Return 1-sigma bearing measurement uncertainty in degrees.

    Model:
    - Base: 30° at 0 dB SNR, narrows with SNR (asymptotic ~3°)
    - Frequency: higher frequency → effectively narrower beam → less ambiguity
    - SDR quality: hardware-specific multiplier
    - Movement geometry: poor geometry (operator hardly moved) inflates uncertainty
    - Floor: 2° (single-element receiver physical minimum)
    """
    snr_clamped = max(0.0, min(snr_db, 60.0))
    sigma_base = 30.0 / (1.0 + snr_clamped / 10.0)

    freq_factor = math.sqrt(max(300.0, freq_mhz) / freq_mhz) if freq_mhz > 0 else 1.0
    sigma_freq = sigma_base * freq_factor

    quality = _SDR_QUALITY.get(sdr_type.lower(), _SDR_QUALITY["default"])
    sigma_sdr = sigma_freq * quality

    # Poor movement geometry = observer barely moved = line-of-bearing uncertainty inflated
    geo_penalty = 1.0 if movement_geometry >= 0.5 else 1.0 + (0.5 - movement_geometry) * 1.0
    sigma = sigma_sdr * geo_penalty

    return round(max(2.0, sigma), 2)


def estimate_bearing(
    samples: list[dict],
    freq_mhz: float = 100.0,
    sdr_type: str = "default",
) -> tuple[float, float]:
    """Weighted circular mean of bearing observations.

    Returns (bearing_deg, sigma_deg).

    Uses SNR-weighting and sine/cosine decomposition to handle wrap-around.
    """
    valid = [s for s in samples if s.get("bearing_deg") is not None and s.get("snr_db") is not None]
    if not valid:
        return 0.0, 45.0   # maximum uncertainty

    # Weighted circular mean via unit-vector sum
    sin_sum = 0.0
    cos_sum = 0.0
    total_w = 0.0
    snrs = []
    for s in valid:
        bearing = float(s["bearing_deg"])
        snr = float(s.get("snr_db", 0.0))
        w = max(0.1, (snr + 20.0) / 40.0)   # weight ∝ SNR
        sin_sum += w * math.sin(math.radians(bearing))
        cos_sum += w * math.cos(math.radians(bearing))
        total_w += w
        snrs.append(snr)

    bearing_rad = math.atan2(sin_sum / total_w, cos_sum / total_w)
    bearing_deg = (math.degrees(bearing_rad) + 360.0) % 360.0

    avg_snr = sum(snrs) / len(snrs)
    sigma = bearing_uncertainty_deg(
        snr_db=avg_snr,
        freq_mhz=freq_mhz,
        bandwidth_hz=0.0,
        sdr_type=sdr_type,
        movement_geometry=1.0,   # movement geometry handled externally
    )
    return round(bearing_deg, 2), sigma
