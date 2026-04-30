from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from backend.sdr.base import BaseSdrAdapter, DeviceMetadata, SignalMetrics, SpectrumWindow

try:
    import airspy as _airspy
    _HAS_AIRSPY = True
except ImportError:
    _HAS_AIRSPY = False


class AirspyAdapter(BaseSdrAdapter):
    """Airspy R2/Mini adapter. Uses airspy-python when installed; falls back to mock data."""

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_spectrum_window(self) -> SpectrumWindow:
        if _HAS_AIRSPY:
            try:
                dev = _airspy.Device()
                dev.set_sample_rate(int(self.config.get("sample_rate_hz", 10e6)))
                dev.set_center_freq(int(self.config.get("center_freq_hz", 433.92e6)))
                samples = dev.read_samples(65536)
                dev.close()
                arr = np.array(samples, dtype=np.complex64)
                psd = np.abs(np.fft.fftshift(np.fft.fft(arr, n=64))) ** 2
                psd_db = [round(10 * math.log10(max(v, 1e-20)), 2) for v in psd]
                return SpectrumWindow(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    center_freq_hz=float(self.config.get("center_freq_hz", 433.92e6)),
                    sample_rate_hz=float(self.config.get("sample_rate_hz", 10e6)),
                    psd_bins_db=psd_db,
                )
            except Exception:
                pass
        return SpectrumWindow(
            timestamp=datetime.now(timezone.utc).isoformat(),
            center_freq_hz=float(self.config.get("center_freq_hz", 433.92e6)),
            sample_rate_hz=float(self.config.get("sample_rate_hz", 10e6)),
            psd_bins_db=list(self.config.get("psd_bins_db", [-86.2, -85.1, -84.7, -86.9])),
        )

    def read_signal_metrics(self) -> SignalMetrics:
        return SignalMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            rssi_dbm=float(self.config.get("rssi_dbm", -68.0)),
            snr_db=float(self.config.get("snr_db", 22.0)),
        )

    def read_device_metadata(self) -> DeviceMetadata:
        return DeviceMetadata(
            provider="airspy",
            device_id=self.config.get("device_id", "airspy-r2"),
            serial=self.config.get("serial"),
            gain_db=self.config.get("gain_db", 15.0),
            gps_lat=self.config.get("gps_lat"),
            gps_lon=self.config.get("gps_lon"),
            extras={"bias_tee": self.config.get("bias_tee", False), "library_available": _HAS_AIRSPY},
        )
