from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SpectrumWindow:
    """Provider-neutral spectrum snapshot."""

    timestamp: str
    center_freq_hz: float
    sample_rate_hz: float
    psd_bins_db: list[float]


@dataclass
class SignalMetrics:
    """Provider-neutral signal quality metrics."""

    timestamp: str
    rssi_dbm: float | None = None
    snr_db: float | None = None


@dataclass
class DeviceMetadata:
    """Provider-neutral device metadata, including optional GPS."""

    provider: str
    device_id: str | None = None
    serial: str | None = None
    gain_db: float | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedSdrReading:
    """Unified ingestion schema consumed by downstream pipeline."""

    timestamp: str
    center_freq_hz: float
    sample_rate_hz: float
    rssi_dbm: float | None
    snr_db: float | None
    psd_bins_db: list[float]
    gps: dict[str, float] | None
    provider: str
    metadata: dict[str, Any]


class BaseSdrAdapter(ABC):
    """Common interface every SDR provider adapter must implement."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.connected = False

    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def read_spectrum_window(self) -> SpectrumWindow:
        ...

    @abstractmethod
    def read_signal_metrics(self) -> SignalMetrics:
        ...

    @abstractmethod
    def read_device_metadata(self) -> DeviceMetadata:
        ...

    def read_normalized(self) -> NormalizedSdrReading:
        """Compose provider reads into a single normalized schema."""
        spectrum = self.read_spectrum_window()
        metrics = self.read_signal_metrics()
        metadata = self.read_device_metadata()

        gps = None
        if metadata.gps_lat is not None and metadata.gps_lon is not None:
            gps = {"lat": metadata.gps_lat, "lon": metadata.gps_lon}

        timestamp = spectrum.timestamp or metrics.timestamp or datetime.now(timezone.utc).isoformat()

        return NormalizedSdrReading(
            timestamp=timestamp,
            center_freq_hz=spectrum.center_freq_hz,
            sample_rate_hz=spectrum.sample_rate_hz,
            rssi_dbm=metrics.rssi_dbm,
            snr_db=metrics.snr_db,
            psd_bins_db=spectrum.psd_bins_db,
            gps=gps,
            provider=metadata.provider,
            metadata={
                "device_id": metadata.device_id,
                "serial": metadata.serial,
                "gain_db": metadata.gain_db,
                **metadata.extras,
            },
        )
