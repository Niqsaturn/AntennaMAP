from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SdrFrameSource(BaseModel):
    node: str
    host: str
    port: int = 8073


class SdrCalibrationOffsets(BaseModel):
    freq_offset_hz: float = 0.0
    gain_offset_db: float = 0.0
    noise_floor_offset_db: float = 0.0


class SdrFrameEvent(BaseModel):
    type: str = Field(default="sdr_frame")
    source: SdrFrameSource
    frame_schema_version: int = 1
    frame_seq: int = Field(ge=0)
    center_freq_hz: float
    span_hz: float = Field(gt=0)
    sample_rate_hz: float = Field(gt=0)
    fft_bins: list[float] = Field(min_length=4)
    fft_bin_count: int = Field(ge=4)
    calibration_offsets: SdrCalibrationOffsets
    timestamp: str

    @classmethod
    def validate_payload(cls, payload: dict) -> "SdrFrameEvent":
        model = cls.model_validate(payload)
        if model.fft_bin_count != len(model.fft_bins):
            raise ValueError("fft_bin_count must equal len(fft_bins)")
        datetime.fromisoformat(model.timestamp.replace("Z", "+00:00"))
        return model

