from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from backend.sdr.base import BaseSdrAdapter, DeviceMetadata, SignalMetrics, SpectrumWindow

try:
    from rtlsdr import RtlSdr as _RtlSdr
    _HAS_RTLSDR = True
except ImportError:
    _HAS_RTLSDR = False


def _rtlsdr_read(config: dict) -> dict:
    sdr = _RtlSdr(device_index=int(config.get("device_index", 0)))
    try:
        sdr.sample_rate = float(config.get("sample_rate_hz", 2.4e6))
        sdr.center_freq = float(config.get("center_freq_hz", 1090e6))
        sdr.gain = config.get("gain_db", "auto")
        sdr.freq_correction = int(config.get("ppm_error", 0))
        samples = sdr.read_samples(256 * 1024)
    finally:
        sdr.close()

    psd = np.abs(np.fft.fftshift(np.fft.fft(samples, n=64))) ** 2
    psd_db = [round(10 * math.log10(max(v, 1e-20)), 2) for v in psd]
    noise = sorted(psd_db)[: len(psd_db) // 4]
    rssi = round(float(np.mean(psd_db)), 2)
    snr = round(max(0.0, max(psd_db) - float(np.mean(noise))), 2)
    return {"psd_db": psd_db, "rssi_dbm": rssi, "snr_db": snr}


class RtlSdrAdapter(BaseSdrAdapter):
    """RTL-SDR adapter. Uses pyrtlsdr when installed; falls back to mock data."""

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_spectrum_window(self) -> SpectrumWindow:
        if _HAS_RTLSDR:
            try:
                data = _rtlsdr_read(self.config)
                return SpectrumWindow(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    center_freq_hz=float(self.config.get("center_freq_hz", 1090e6)),
                    sample_rate_hz=float(self.config.get("sample_rate_hz", 2.4e6)),
                    psd_bins_db=data["psd_db"],
                )
            except Exception:
                pass
        return SpectrumWindow(
            timestamp=datetime.now(timezone.utc).isoformat(),
            center_freq_hz=float(self.config.get("center_freq_hz", 1090e6)),
            sample_rate_hz=float(self.config.get("sample_rate_hz", 2.4e6)),
            psd_bins_db=list(self.config.get("psd_bins_db", [-80.0, -79.3, -81.1, -83.2])),
        )

    def read_signal_metrics(self) -> SignalMetrics:
        if _HAS_RTLSDR:
            try:
                data = _rtlsdr_read(self.config)
                return SignalMetrics(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    rssi_dbm=data["rssi_dbm"],
                    snr_db=data["snr_db"],
                )
            except Exception:
                pass
        return SignalMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            rssi_dbm=float(self.config.get("rssi_dbm", -72.1)),
            snr_db=float(self.config.get("snr_db", 18.0)),
        )

    def read_device_metadata(self) -> DeviceMetadata:
        return DeviceMetadata(
            provider="rtlsdr",
            device_id=self.config.get("device_id", "rtl-0"),
            serial=self.config.get("serial"),
            gain_db=self.config.get("gain_db", 20.7),
            gps_lat=self.config.get("gps_lat"),
            gps_lon=self.config.get("gps_lon"),
            extras={"ppm_error": self.config.get("ppm_error", 0), "library_available": _HAS_RTLSDR},
        )
