from __future__ import annotations

from datetime import datetime, timezone

from backend.sdr.base import BaseSdrAdapter, DeviceMetadata, SignalMetrics, SpectrumWindow


class RtlSdrAdapter(BaseSdrAdapter):
    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_spectrum_window(self) -> SpectrumWindow:
        return SpectrumWindow(
            timestamp=datetime.now(timezone.utc).isoformat(),
            center_freq_hz=float(self.config.get("center_freq_hz", 1090e6)),
            sample_rate_hz=float(self.config.get("sample_rate_hz", 2.4e6)),
            psd_bins_db=list(self.config.get("psd_bins_db", [-80.0, -79.3, -81.1, -83.2])),
        )

    def read_signal_metrics(self) -> SignalMetrics:
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
            extras={"ppm_error": self.config.get("ppm_error", 0)},
        )
