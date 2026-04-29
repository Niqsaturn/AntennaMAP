from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CalibrationModel:
    mean_error_m: float
    std_error_m: float
    confidence_bias: float


class FoxHuntTrainer:
    def __init__(self) -> None:
        self.model = CalibrationModel(mean_error_m=50.0, std_error_m=30.0, confidence_bias=0.0)

    def fit(self, historical_sessions: list[dict]) -> CalibrationModel:
        errors = [s.get("error_m") for s in historical_sessions if s.get("error_m") is not None]
        confidences = [s.get("confidence_score") for s in historical_sessions if s.get("confidence_score") is not None]
        if errors:
            mean_error = float(np.mean(errors))
            std_error = float(np.std(errors))
        else:
            mean_error, std_error = self.model.mean_error_m, self.model.std_error_m

        if confidences:
            confidence_bias = float(np.mean(confidences) - 0.5)
        else:
            confidence_bias = self.model.confidence_bias

        self.model = CalibrationModel(mean_error_m=mean_error, std_error_m=std_error, confidence_bias=confidence_bias)
        return self.model

    def calibrate_confidence(self, raw_confidence: float) -> float:
        return float(np.clip(raw_confidence - self.model.confidence_bias * 0.2, 0.0, 1.0))
