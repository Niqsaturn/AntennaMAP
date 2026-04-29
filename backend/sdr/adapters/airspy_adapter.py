from __future__ import annotations

from datetime import datetime, timezone

from backend.sdr.base import BaseSdrAdapter, DeviceMetadata, SignalMetrics, SpectrumWindow


class AirspyAdapter(BaseSdrAdapter):
    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_spectrum_window(self) -> SpectrumWindow:
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
            device_id=self.config.get("device_id", "airspy-mini"),
            serial=self.config.get("serial"),
            gain_db=self.config.get("gain_db", 15.0),
            gps_lat=self.config.get("gps_lat"),
            gps_lon=self.config.get("gps_lon"),
            extras={"bias_tee": self.config.get("bias_tee", False)},
        )
