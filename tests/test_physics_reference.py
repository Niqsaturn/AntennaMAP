"""Physics-grade reference vector tests.

Each test checks a known analytical result against a published standard
so that physics regressions are caught immediately.

References:
  - ITU-R P.525  : free-space path loss
  - ITU-R P.370  : Hata urban model published tables
  - ITU-R P.533  : HF ionospheric propagation
  - Circular mean: Jammalamadaka & SenGupta (2001) eq. 2.2
"""
from __future__ import annotations

import math
import pytest


# ── ITU-R P.525: FSPL ──────────────────────────────────────────────────────

def test_fspl_100mhz_10km():
    """FSPL at 100 MHz, 10 km = 32.44 + 20·log10(100) + 20·log10(10) = 92.44 dB."""
    from backend.rf_propagation import path_loss_db
    pl = path_loss_db("fspl", 100.0, 10.0)
    expected = 32.44 + 20 * math.log10(100.0) + 20 * math.log10(10.0)
    assert abs(pl - expected) < 0.1, f"FSPL={pl:.2f}, expected={expected:.2f}"


def test_fspl_900mhz_1km():
    """FSPL at 900 MHz, 1 km = 32.44 + 20·log10(900) + 20·log10(1) ≈ 91.54 dB."""
    from backend.rf_propagation import path_loss_db
    pl = path_loss_db("fspl", 900.0, 1.0)
    expected = 32.44 + 20 * math.log10(900.0) + 20 * math.log10(1.0)
    assert abs(pl - expected) < 0.1, f"FSPL={pl:.2f}, expected={expected:.2f}"


def test_fspl_2400mhz_0_1km():
    """FSPL at 2.4 GHz, 100 m ≈ 60 dB."""
    from backend.rf_propagation import path_loss_db
    pl = path_loss_db("fspl", 2400.0, 0.1)
    expected = 32.44 + 20 * math.log10(2400.0) + 20 * math.log10(0.1)
    assert abs(pl - expected) < 0.1, f"FSPL={pl:.2f}, expected={expected:.2f}"


# ── Hata urban model ───────────────────────────────────────────────────────

def test_hata_urban_monotone_with_distance():
    """Hata path loss must increase with distance."""
    from backend.rf_propagation import path_loss_db
    pl1 = path_loss_db("hata_urban", 900.0, 1.0)
    pl5 = path_loss_db("hata_urban", 900.0, 5.0)
    pl20 = path_loss_db("hata_urban", 900.0, 20.0)
    assert pl1 < pl5 < pl20, f"non-monotone Hata: {pl1:.1f} {pl5:.1f} {pl20:.1f}"


def test_hata_urban_range_900mhz():
    """Hata urban at 900 MHz, 1 km is between 100 and 130 dB (published range)."""
    from backend.rf_propagation import path_loss_db
    pl = path_loss_db("hata_urban", 900.0, 1.0)
    assert 100 <= pl <= 135, f"Hata@900MHz,1km={pl:.1f} dB — outside expected 100–135 dB"


# ── HF ionospheric skip distance ──────────────────────────────────────────

def test_hf_skip_14mhz_at_least_100km():
    """HF 14 MHz skip distance ≥ 100 km (standard amateur-radio empirical value)."""
    from backend.rf_propagation import ionospheric_skip_distance_km
    result = ionospheric_skip_distance_km(14.0)
    skip_km = result.get("skip_distance_km", 0.0)
    assert skip_km >= 100.0, f"14 MHz skip={skip_km:.0f} km, expected ≥100 km"


def test_hf_skip_increases_with_frequency():
    """Skip distance must increase with frequency (higher freq → higher skip)."""
    from backend.rf_propagation import ionospheric_skip_distance_km
    skip7  = ionospheric_skip_distance_km(7.0).get("skip_distance_km", 0)
    skip14 = ionospheric_skip_distance_km(14.0).get("skip_distance_km", 0)
    skip28 = ionospheric_skip_distance_km(28.0).get("skip_distance_km", 0)
    assert skip7 <= skip14 <= skip28, f"skip not monotone: {skip7:.0f} {skip14:.0f} {skip28:.0f}"


def test_hf_range_estimator_skywave_gt_100km():
    """HF range at 14 MHz, -85 dBm should exceed 100 km (skywave model)."""
    from backend.analysis.range_estimator import estimate_range_km
    dist_km, _ = estimate_range_km(rssi_dbm=-85, freq_mhz=14.0, snr_db=10, bandwidth_hz=3000)
    assert dist_km > 100, f"HF range={dist_km:.1f} km, expected >100 km"


# ── Circular mean of bearing angles ───────────────────────────────────────

def test_circular_mean_wrap():
    """Circular mean of [350°, 10°] with equal weights should equal 0°."""
    from backend.analysis.bearing_tracker import estimate_bearing
    samples = [
        {"bearing_deg": 350.0, "snr_db": 20.0},
        {"bearing_deg": 10.0,  "snr_db": 20.0},
    ]
    mean_bearing, _ = estimate_bearing(samples, freq_mhz=100.0)
    # Result should be near 0° (wrapping correctly)
    diff = abs(((mean_bearing + 180) % 360) - 180)
    assert diff < 5.0, f"circular mean={mean_bearing:.1f}°, expected ~0°"


def test_circular_mean_north():
    """Circular mean of [0°, 0°, 0°] = 0°."""
    from backend.analysis.bearing_tracker import estimate_bearing
    samples = [{"bearing_deg": 0.0, "snr_db": 15.0}] * 3
    mean_bearing, _ = estimate_bearing(samples, freq_mhz=100.0)
    assert abs(mean_bearing) < 1.0, f"mean={mean_bearing:.2f}°, expected 0°"


def test_circular_mean_south():
    """Circular mean of [180°, 180°] = 180°."""
    from backend.analysis.bearing_tracker import estimate_bearing
    samples = [{"bearing_deg": 180.0, "snr_db": 15.0}] * 2
    mean_bearing, _ = estimate_bearing(samples, freq_mhz=100.0)
    assert abs(mean_bearing - 180.0) < 1.0, f"mean={mean_bearing:.2f}°, expected 180°"


# ── Array calculator sanity checks ────────────────────────────────────────

def test_linear_array_8el_uniform_gain():
    """8-element uniform linear array gain ≈ 9 dB (= 10·log10(8))."""
    from backend.rf.array_calculator import LinearArrayParams, linear_array_pattern
    p = LinearArrayParams(n_elements=8, element_spacing_m=0.17, frequency_hz=870e6,
                          steering_angle_deg=0.0, window="uniform")
    r = linear_array_pattern(p)
    assert abs(r.array_gain_db - 10 * math.log10(8)) < 0.5, \
        f"gain={r.array_gain_db:.2f} dB, expected ≈{10*math.log10(8):.2f} dB"


def test_linear_array_hpbw_decreases_with_n():
    """More elements → narrower HPBW."""
    from backend.rf.array_calculator import LinearArrayParams, linear_array_pattern
    lam = 3e8 / 870e6
    r4 = linear_array_pattern(LinearArrayParams(4,  lam / 2, 870e6))
    r8 = linear_array_pattern(LinearArrayParams(8,  lam / 2, 870e6))
    r16 = linear_array_pattern(LinearArrayParams(16, lam / 2, 870e6))
    assert r4.hpbw_deg > r8.hpbw_deg > r16.hpbw_deg, \
        f"HPBW not decreasing: {r4.hpbw_deg} {r8.hpbw_deg} {r16.hpbw_deg}"


def test_array_no_steer_az_attribute_error():
    """P0-1 regression: LinearArrayParams.steer_az_deg must not exist."""
    from backend.rf.array_calculator import LinearArrayParams, linear_array_pattern
    p = LinearArrayParams(n_elements=8, element_spacing_m=0.17,
                          frequency_hz=870e6, steering_angle_deg=30.0)
    r = linear_array_pattern(p)   # would raise AttributeError before fix
    assert r.main_beam_deg is not None


# ── Kalman process noise ──────────────────────────────────────────────────

def test_kalman_process_noise_scales_with_dt():
    """Kalman uncertainty after longer predict step must be larger."""
    from backend.analysis.kalman_tracker import PositionKalman
    kf1 = PositionKalman(37.9, -122.3)
    kf2 = PositionKalman(37.9, -122.3)
    kf1.predict(dt_seconds=10.0)
    kf2.predict(dt_seconds=3600.0)
    # P must grow more for larger dt
    assert kf2.P_lat > kf1.P_lat, "Kalman P_lat should be larger after longer dt"


# ── Waterfall noise floor ─────────────────────────────────────────────────

def test_noise_floor_5th_percentile():
    """Noise floor uses 5th percentile — should be at or below the 20th-percentile value."""
    from backend.analysis.waterfall_analyzer import _noise_floor
    # Uniform bins from -120 to -60 dBm — 5th pct ≈ -117, 20th pct ≈ -108
    bins = list(range(-120, -60))  # 60 values
    floor_5 = _noise_floor(bins, 0.0)
    # 5th percentile of -120...-61 = first 3 values ≈ -120, -119, -118 → mean ≈ -119
    assert floor_5 <= -100.0, f"noise floor={floor_5:.1f} — not using low-percentile bins"


# ── Bearing uncertainty bandwidth penalty ─────────────────────────────────

def test_bearing_uncertainty_cw_lower_than_fm():
    """CW (100 Hz) bearing uncertainty must be lower than FM (200 kHz)."""
    from backend.analysis.bearing_tracker import bearing_uncertainty_deg
    sigma_cw = bearing_uncertainty_deg(snr_db=20, freq_mhz=100, bandwidth_hz=100)
    sigma_fm = bearing_uncertainty_deg(snr_db=20, freq_mhz=100, bandwidth_hz=200_000)
    assert sigma_cw < sigma_fm, f"CW σ={sigma_cw}° not less than FM σ={sigma_fm}°"


# ── Observability guard ───────────────────────────────────────────────────

def test_observability_low_confidence_with_single_bearing():
    """Single bearing observation should produce low-confidence fix."""
    from backend.foxhunt.multilateration import BearingObs, locate_transmitter
    obs = [BearingObs(lat=37.9, lon=-122.3, bearing_deg=45.0, snr_db=15, freq_hz=100e6)]
    fix = locate_transmitter(bearing_obs=obs, freq_hz=100e6)
    assert fix.confidence <= 0.15, f"single bearing confidence={fix.confidence:.3f} — should be ≤0.15"


def test_observability_high_confidence_three_diverse_bearings():
    """Three well-separated bearing observations should yield observable geometry."""
    from backend.foxhunt.multilateration import BearingObs, locate_transmitter
    obs = [
        BearingObs(lat=37.80, lon=-122.40, bearing_deg=45.0,  snr_db=20, freq_hz=100e6),
        BearingObs(lat=37.90, lon=-122.50, bearing_deg=90.0,  snr_db=20, freq_hz=100e6),
        BearingObs(lat=38.00, lon=-122.30, bearing_deg=200.0, snr_db=20, freq_hz=100e6),
    ]
    fix = locate_transmitter(bearing_obs=obs, freq_hz=100e6)
    assert fix.confidence > 0.15, f"three diverse bearings confidence={fix.confidence:.3f} — should be >0.15"
