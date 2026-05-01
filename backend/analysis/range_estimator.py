"""Synthesized RSSI → distance inversion.

Converts received signal strength into an estimated distance by binary-searching
the forward path-loss model until it matches the observed RSSI.  Band-appropriate
propagation models are selected automatically from the carrier frequency.
"""
from __future__ import annotations

import math

from backend.rf_propagation import path_loss_db, ionospheric_skip_distance_km, atmospheric_absorption_db

# Ground-wave attenuation lookup: ITU-R P.368 simplified table
# {freq_mhz: db_per_km at 100 km reference} — interpolated for other distances
_GROUNDWAVE_ATT: list[tuple[float, float]] = [
    (0.003,  1.0), (0.01,  1.2), (0.03,  1.5), (0.1,  2.5),
    (0.3,   4.0), (1.0,  8.0), (3.0, 18.0),
]

# Band-specific fallback distances (km) when RSSI/EIRP inversion fails
_BAND_FALLBACK_KM: list[tuple[float, float, float]] = [
    # (freq_low_mhz, freq_high_mhz, fallback_km)
    (0.000,   3.0,  200.0),  # VLF/LF: ground wave, very long range
    (3.0,    30.0,   30.0),  # HF: skip/skywave variable
    (30.0,  300.0,    5.0),  # VHF
    (300.0, 1000.0,   1.0),  # UHF
    (1000.0, 30000.0, 0.3),  # SHF
    (30000.0, 1e9,   0.05),  # EHF
]


def _groundwave_loss_db(freq_mhz: float, distance_km: float) -> float:
    """Approximate ITU-R P.368 ground-wave path loss (dB)."""
    # Interpolate attenuation rate from lookup table
    rate = _GROUNDWAVE_ATT[0][1]
    for i in range(len(_GROUNDWAVE_ATT) - 1):
        f0, a0 = _GROUNDWAVE_ATT[i]
        f1, a1 = _GROUNDWAVE_ATT[i + 1]
        if f0 <= freq_mhz <= f1:
            frac = (freq_mhz - f0) / (f1 - f0)
            rate = a0 + frac * (a1 - a0)
            break
        if freq_mhz > _GROUNDWAVE_ATT[-1][0]:
            rate = _GROUNDWAVE_ATT[-1][1]

    # Free-space base + ground-wave excess loss (rate × distance)
    fspl = 32.44 + 20 * math.log10(max(freq_mhz, 0.001)) + 20 * math.log10(max(distance_km, 0.001))
    gw_excess = rate * distance_km
    return fspl + gw_excess


def _select_model(freq_mhz: float) -> str:
    if freq_mhz < 3.0:
        return "groundwave"
    if freq_mhz < 30.0:
        return "hf_skywave"   # ionospheric sky-wave dominates HF
    if freq_mhz < 1000.0:
        return "hata_urban"
    return "fspl"


def _hf_skywave_loss_db(freq_mhz: float, distance_km: float) -> float:
    """Simplified ITU-R P.533 HF sky-wave path loss.

    PL = 32.44 + 20·log10(f_MHz) + 20·log10(d_km) + F_sky
    F_sky ≈ 6·log10(max(d_km,1000)/1000) — extra absorption at long ground range
    """
    d = max(distance_km, 1.0)
    f_sky = 6.0 * math.log10(max(d, 1000.0) / 1000.0)
    return 32.44 + 20 * math.log10(freq_mhz) + 20 * math.log10(d) + f_sky


def _forward_loss(freq_mhz: float, distance_km: float, model: str, tx_height_m: float) -> float:
    """Compute path loss (dB) for the given model and distance."""
    if model == "groundwave":
        return _groundwave_loss_db(freq_mhz, distance_km)
    if model == "hf_skywave":
        return _hf_skywave_loss_db(freq_mhz, distance_km)
    if model in ("hata_urban", "hata_suburban"):
        return path_loss_db(model, freq_mhz, distance_km, tx_height_m)
    # fspl + atmospheric absorption for SHF/EHF
    fspl = path_loss_db("fspl", freq_mhz, distance_km, tx_height_m)
    if freq_mhz >= 10_000:
        atm = atmospheric_absorption_db(freq_mhz / 1000.0, distance_km)
        fspl += atm.get("total_absorption_db", 0.0)
    return fspl


def _fallback_distance_km(freq_mhz: float) -> float:
    for lo, hi, fallback in _BAND_FALLBACK_KM:
        if lo <= freq_mhz < hi:
            return fallback
    return 1.0


def _default_eirp(freq_mhz: float) -> float:
    """Typical EIRP (dBm) from spectrum allocations, or generic fallback."""
    try:
        from backend.datasources.spectrum_allocations import allocations_for_freq
        allocs = allocations_for_freq(freq_mhz)
        if allocs:
            return float(allocs[0].get("typical_eirp_dbm", 47.0))
    except Exception:
        pass
    # Generic defaults by band
    if freq_mhz < 3:
        return 90.0   # ELF/VLF transmitters are enormous
    if freq_mhz < 30:
        return 60.0   # HF shortwave broadcast
    if freq_mhz < 300:
        return 50.0   # VHF broadcast
    if freq_mhz < 1000:
        return 46.0   # UHF cellular
    return 40.0       # SHF/EHF


def estimate_range_km(
    rssi_dbm: float,
    freq_mhz: float,
    snr_db: float,
    bandwidth_hz: float,
    path_model: str = "auto",
    tx_height_m: float = 30.0,
    rx_height_m: float = 1.5,
    typical_eirp_dbm: float | None = None,
) -> tuple[float, float]:
    """Estimate transmitter range from received RSSI.

    Returns (distance_km, uncertainty_km).  Uses binary search on the forward
    path-loss model until the predicted RSSI matches the observed value.

    For HF bands, also checks whether the observed RSSI is consistent with
    sky-wave propagation and constrains the minimum range to the skip distance.
    """
    freq_mhz = max(freq_mhz, 0.001)
    eirp = typical_eirp_dbm if typical_eirp_dbm is not None else _default_eirp(freq_mhz)

    if path_model == "auto":
        model = _select_model(freq_mhz)
    else:
        model = path_model

    # RX gain assumed 0 dBi (isotropic)
    # RSSI = EIRP - PathLoss → PathLoss = EIRP - RSSI
    target_pl = eirp - rssi_dbm
    if target_pl <= 0:
        # Signal stronger than EIRP would predict at 0 distance — clamp to 0.1 km
        dist = 0.1
    else:
        # Binary search on distance
        lo, hi = 0.01, 5000.0
        for _ in range(50):
            mid = (lo + hi) / 2
            pl = _forward_loss(freq_mhz, mid, model, tx_height_m)
            if pl < target_pl:
                lo = mid
            else:
                hi = mid
        dist = (lo + hi) / 2

    # HF skip constraint: distance can't be less than skip distance
    if 3.0 <= freq_mhz <= 30.0:
        try:
            skip = ionospheric_skip_distance_km(freq_mhz)
            skip_dist = skip.get("skip_distance_km", 0.0)
            if skip.get("propagation_possible") and dist < skip_dist:
                dist = skip_dist
        except Exception:
            pass

    # Uncertainty: larger at low SNR, larger at long distances
    snr_factor = max(0.1, 1.0 - min(snr_db, 40.0) / 40.0)
    uncertainty = dist * (0.35 * snr_factor + 0.1)

    # If result is far outside physical limits, fall back to band default
    max_range = 10_000.0 if freq_mhz < 30 else (500.0 if freq_mhz < 300 else 50.0)
    if dist > max_range or not math.isfinite(dist):
        dist = _fallback_distance_km(freq_mhz)
        uncertainty = dist * 0.5

    return round(dist, 3), round(uncertainty, 3)
