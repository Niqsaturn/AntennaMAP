from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from backend.sdr.base import BaseSdrAdapter, DeviceMetadata, SignalMetrics, SpectrumWindow

try:
    import pyhackrf as _pyhackrf
    _HAS_HACKRF = True
except ImportError:
    _HAS_HACKRF = False


class HackrfAdapter(BaseSdrAdapter):
    """HackRF One adapter. Uses pyhackrf when installed; falls back to mock data."""

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_spectrum_window(self) -> SpectrumWindow:
        if _HAS_HACKRF:
            try:
                samples = _pyhackrf.read_samples(
                    center_freq=int(self.config.get("center_freq_hz", 915e6)),
                    sample_rate=int(self.config.get("sample_rate_hz", 8e6)),
                    num_samples=131072,
                )
                psd = np.abs(np.fft.fftshift(np.fft.fft(np.frombuffer(samples, dtype=np.complex64), n=64))) ** 2
                psd_db = [round(10 * math.log10(max(v, 1e-20)), 2) for v in psd]
                return SpectrumWindow(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    center_freq_hz=float(self.config.get("center_freq_hz", 915e6)),
                    sample_rate_hz=float(self.config.get("sample_rate_hz", 8e6)),
                    psd_bins_db=psd_db,
                )
            except Exception:
                pass
        return SpectrumWindow(
            timestamp=datetime.now(timezone.utc).isoformat(),
            center_freq_hz=float(self.config.get("center_freq_hz", 915e6)),
            sample_rate_hz=float(self.config.get("sample_rate_hz", 8e6)),
            psd_bins_db=list(self.config.get("psd_bins_db", [-78.6, -77.9, -80.4, -81.3])),
        )

    def read_signal_metrics(self) -> SignalMetrics:
        return SignalMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            rssi_dbm=float(self.config.get("rssi_dbm", -70.5)),
            snr_db=float(self.config.get("snr_db", 14.8)),
        )

    def read_device_metadata(self) -> DeviceMetadata:
        return DeviceMetadata(
            provider="hackrf",
            device_id=self.config.get("device_id", "hackrf-one"),
            serial=self.config.get("serial"),
            gain_db=self.config.get("gain_db", 24.0),
            gps_lat=self.config.get("gps_lat"),
            gps_lon=self.config.get("gps_lon"),
            extras={"amp_enabled": self.config.get("amp_enabled", False), "library_available": _HAS_HACKRF},
        )
