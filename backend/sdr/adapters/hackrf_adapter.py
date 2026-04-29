from __future__ import annotations

from datetime import datetime, timezone

from backend.sdr.base import BaseSdrAdapter, DeviceMetadata, SignalMetrics, SpectrumWindow


class HackrfAdapter(BaseSdrAdapter):
    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_spectrum_window(self) -> SpectrumWindow:
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
            extras={"amp_enabled": self.config.get("amp_enabled", False)},
        )
