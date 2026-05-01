"""Spectral peak extraction from raw PSD bins.

Converts waterfall/spectrogram data into physically meaningful signal features:
peak frequency, 3 dB bandwidth, SNR, and a modulation hint based on bandwidth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SpectralPeak:
    center_freq_hz: float
    bandwidth_3db_hz: float
    peak_dbm: float
    snr_db: float
    noise_floor_dbm: float
    modulation_hint: str   # "CW"|"SSB"|"AM"|"FM"|"WBFM"|"data"|"unknown"
    bin_index: int         # index in original PSD array


def _noise_floor(psd_bins_db: list[float], noise_figure_db: float) -> float:
    """Bottom 5th-percentile mean of PSD bins + NF correction.

    Using 5th percentile (not 20th) to avoid bias from strong in-band signals
    elevating the estimated noise floor.
    """
    if not psd_bins_db:
        return -100.0 + noise_figure_db
    sorted_bins = sorted(psd_bins_db)
    cutoff = max(1, len(sorted_bins) // 20)
    return sum(sorted_bins[:cutoff]) / cutoff + noise_figure_db


def _modulation_hint(bw_hz: float) -> str:
    if bw_hz < 500:
        return "CW"
    if bw_hz < 6_000:
        return "SSB"
    if bw_hz < 15_000:
        return "AM"
    if bw_hz < 300_000:
        return "FM"
    return "WBFM"


def _bin_freq_hz(bin_index: int, n_bins: int, center_freq_hz: float, sample_rate_hz: float) -> float:
    """Map bin index to absolute frequency."""
    offset_hz = (bin_index - n_bins / 2) * (sample_rate_hz / n_bins)
    return center_freq_hz + offset_hz


def analyze_psd(
    psd_bins_db: list[float],
    center_freq_hz: float,
    sample_rate_hz: float,
    noise_figure_db: float = 5.0,
    min_snr_db: float = 6.0,
) -> list[SpectralPeak]:
    """Detect signal peaks in a PSD array and return SpectralPeak objects.

    Uses a threshold 6 dB above the noise floor. Adjacent above-threshold bins
    are merged into a single peak. Bandwidth is measured at the -3 dB point
    walking left/right from each peak maximum.
    """
    if not psd_bins_db or len(psd_bins_db) < 3:
        return []

    n = len(psd_bins_db)
    floor = _noise_floor(psd_bins_db, noise_figure_db)
    threshold = floor + min_snr_db

    # Find above-threshold runs
    in_peak = False
    runs: list[tuple[int, int]] = []   # (start, end) inclusive
    start = 0
    for i, v in enumerate(psd_bins_db):
        if v >= threshold:
            if not in_peak:
                start = i
                in_peak = True
        else:
            if in_peak:
                runs.append((start, i - 1))
                in_peak = False
    if in_peak:
        runs.append((start, n - 1))

    peaks: list[SpectralPeak] = []
    for r_start, r_end in runs:
        # Find maximum within run
        peak_idx = max(range(r_start, r_end + 1), key=lambda i: psd_bins_db[i])
        peak_dbm = psd_bins_db[peak_idx]
        half_power = peak_dbm - 3.0

        # Walk left to find -3 dB edge
        left = peak_idx
        while left > 0 and psd_bins_db[left - 1] >= half_power:
            left -= 1

        # Walk right to find -3 dB edge
        right = peak_idx
        while right < n - 1 and psd_bins_db[right + 1] >= half_power:
            right += 1

        left_freq = _bin_freq_hz(left, n, center_freq_hz, sample_rate_hz)
        right_freq = _bin_freq_hz(right, n, center_freq_hz, sample_rate_hz)
        center_freq = _bin_freq_hz(peak_idx, n, center_freq_hz, sample_rate_hz)
        bw_hz = max(sample_rate_hz / n, right_freq - left_freq)   # at least 1 bin wide
        snr = peak_dbm - floor

        peaks.append(SpectralPeak(
            center_freq_hz=round(center_freq, 1),
            bandwidth_3db_hz=round(bw_hz, 1),
            peak_dbm=round(peak_dbm, 2),
            snr_db=round(snr, 2),
            noise_floor_dbm=round(floor, 2),
            modulation_hint=_modulation_hint(bw_hz),
            bin_index=peak_idx,
        ))

    return peaks


def doppler_shift_hz(prev_peak: SpectralPeak, curr_peak: SpectralPeak) -> float:
    """Signed frequency shift between two measurements of the same signal (Hz).

    Positive = frequency increased (emitter approaching), negative = receding.
    """
    return curr_peak.center_freq_hz - prev_peak.center_freq_hz


def freq_hopping_detected(peaks: list[SpectralPeak], sample_rate_hz: float) -> bool:
    """Return True if the peaks pattern suggests frequency hopping.

    Heuristic: ≥3 peaks spread across >1% of the sample rate with similar power.
    """
    if len(peaks) < 3:
        return False
    span = max(p.center_freq_hz for p in peaks) - min(p.center_freq_hz for p in peaks)
    if span < sample_rate_hz * 0.01:
        return False
    powers = [p.peak_dbm for p in peaks]
    power_spread = max(powers) - min(powers)
    return power_spread < 15.0   # similar power levels (within 15 dB)
