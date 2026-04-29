from __future__ import annotations

from dataclasses import dataclass
from math import fmod
from statistics import mean

AntennaType = str

TAXONOMY: tuple[AntennaType, ...] = (
    "omni",
    "sector_panel",
    "dish",
    "phased_array",
    "unknown",
)


@dataclass(frozen=True)
class ClassificationResult:
    antenna_type: AntennaType
    confidence: float
    estimated_elements: dict


def _circular_spread_deg(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    norm = sorted(v % 360 for v in values)
    gaps = [norm[i + 1] - norm[i] for i in range(len(norm) - 1)]
    gaps.append(360 - norm[-1] + norm[0])
    largest_gap = max(gaps)
    return max(0.0, 360 - largest_gap)


def classify_antenna(properties: dict, telemetry_samples: list[dict] | None = None) -> ClassificationResult:
    telemetry_samples = telemetry_samples or []
    bearings = [float(s["bearing_deg"]) for s in telemetry_samples if s.get("bearing_deg") is not None]
    spread = _circular_spread_deg(bearings)
    explicit_polarization = properties.get("polarization")

    rf_min = properties.get("rf_min_mhz")
    rf_max = properties.get("rf_max_mhz")
    freq_center = None
    if isinstance(rf_min, (int, float)) and isinstance(rf_max, (int, float)):
        freq_center = (rf_min + rf_max) / 2

    directionality = (properties.get("directionality") or "").lower()
    beamwidth = None
    antenna_type = "unknown"
    confidence = 0.4

    if directionality == "omni" or (spread is not None and spread > 220):
        antenna_type = "omni"
        beamwidth = 360
        confidence = 0.88 if directionality == "omni" else 0.74
    elif spread is not None and spread < 25 and freq_center and freq_center >= 3000:
        antenna_type = "dish"
        beamwidth = 6 + min(10, spread / 4)
        confidence = 0.81
    elif spread is not None and spread < 55:
        antenna_type = "phased_array" if freq_center and freq_center >= 24000 else "sector_panel"
        beamwidth = max(12, min(65, spread * 1.5))
        confidence = 0.77
    elif directionality == "sector" or (spread is not None and spread < 140):
        antenna_type = "sector_panel"
        beamwidth = 65 if spread is None else max(40, min(120, spread))
        confidence = 0.72

    azimuth = properties.get("azimuth_deg")
    if azimuth is None and bearings:
        azimuth = mean(bearings)

    if beamwidth is None:
        beamwidth = 90 if antenna_type == "unknown" else 60

    if beamwidth >= 300:
        sector_count = 1
    else:
        sector_count = max(1, min(12, round(360 / beamwidth)))

    tilt = -6 if antenna_type in {"sector_panel", "phased_array"} else -1
    polarization = explicit_polarization or (
        "dual" if antenna_type in {"sector_panel", "phased_array"} else "vertical"
    )

    gain_bucket = "high" if antenna_type in {"dish", "phased_array"} else "medium" if antenna_type == "sector_panel" else "low"

    estimated_elements = {
        "estimated_beamwidth_deg": round(float(beamwidth), 1),
        "array_orientation_deg": None if azimuth is None else round(fmod(float(azimuth), 360), 1),
        "sector_count": int(sector_count),
        "tilt_estimate_deg": tilt,
        "polarization_class": polarization,
        "gain_bucket": gain_bucket,
        "bearing_spread_deg": None if spread is None else round(spread, 1),
        "stability_class": "stable" if telemetry_samples and len(telemetry_samples) >= 3 else "unknown",
        "frequency_regime": (
            "mmwave" if freq_center and freq_center >= 24000 else "midband" if freq_center and freq_center >= 1000 else "lowband"
        ),
    }
    return ClassificationResult(antenna_type=antenna_type, confidence=round(confidence, 2), estimated_elements=estimated_elements)
