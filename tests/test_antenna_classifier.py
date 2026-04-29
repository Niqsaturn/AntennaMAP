from backend.rf.antenna_classifier import TAXONOMY, classify_antenna


def test_taxonomy_includes_expected_classes():
    assert TAXONOMY == ("omni", "sector_panel", "dish", "phased_array", "unknown")


def test_omni_classification_from_directionality():
    props = {"directionality": "Omni", "rf_min_mhz": 800, "rf_max_mhz": 900}
    result = classify_antenna(props, [{"bearing_deg": 0}, {"bearing_deg": 180}, {"bearing_deg": 270}])
    assert result.antenna_type == "omni"
    assert result.confidence >= 0.8
    assert result.estimated_elements["estimated_beamwidth_deg"] == 360.0


def test_sector_panel_classification_and_inferred_fields():
    props = {"directionality": "Sector", "azimuth_deg": 120, "rf_min_mhz": 1700, "rf_max_mhz": 2100}
    result = classify_antenna(props, [{"bearing_deg": 110}, {"bearing_deg": 125}, {"bearing_deg": 140}])
    assert result.antenna_type == "sector_panel"
    assert result.estimated_elements["array_orientation_deg"] == 120.0
    assert result.estimated_elements["polarization_class"] == "dual"
    assert result.estimated_elements["sector_count"] >= 3


def test_dish_classification_for_high_freq_narrow_beam():
    props = {"rf_min_mhz": 3500, "rf_max_mhz": 4200}
    result = classify_antenna(props, [{"bearing_deg": 15}, {"bearing_deg": 20}, {"bearing_deg": 21}])
    assert result.antenna_type == "dish"
    assert result.estimated_elements["gain_bucket"] == "high"
