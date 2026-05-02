from backend.foxhunt.auto_loop import (
    _BASE_PIN_CONFIDENCE_THRESHOLD,
    _SINGLE_DIRECTIONAL_PIN_CONFIDENCE_THRESHOLD,
    _certainty_level,
    _observability_class,
    _pin_confidence_threshold,
    FoxTarget,
)
from backend.foxhunt.multilateration import FusedFix


def _fix(conf: float, unc: float = 800.0) -> FusedFix:
    return FusedFix(
        lat=0.0,
        lon=0.0,
        uncertainty_m=unc,
        confidence=conf,
        methods_used=["bearing_intersection"],
        per_method=[],
        ellipse_major_m=unc,
        ellipse_minor_m=unc * 0.6,
        ellipse_angle_deg=0.0,
    )


def _target(bearings: int) -> FoxTarget:
    t = FoxTarget(freq_hz=144_390_000.0, band_label="VHF", modulation_hint="FM", first_seen="2026-05-01T00:00:00Z")
    t.bearing_obs = [object() for _ in range(bearings)]  # count-only for policy helpers
    return t


def test_single_directional_source_uses_strict_pin_threshold():
    assert _pin_confidence_threshold(_target(1)) == _SINGLE_DIRECTIONAL_PIN_CONFIDENCE_THRESHOLD
    assert _pin_confidence_threshold(_target(0)) == _SINGLE_DIRECTIONAL_PIN_CONFIDENCE_THRESHOLD


def test_multi_bearing_uses_base_pin_threshold():
    assert _pin_confidence_threshold(_target(2)) == _BASE_PIN_CONFIDENCE_THRESHOLD


def test_observability_and_certainty_labels():
    assert _observability_class(_target(1), _fix(0.9, 500.0)) == "single_directional_source"
    assert _observability_class(_target(3), _fix(0.9, 700.0)) == "well_observed"
    assert _certainty_level(0.9) == "high"
    assert _certainty_level(0.6) == "medium"
    assert _certainty_level(0.2) == "low"
