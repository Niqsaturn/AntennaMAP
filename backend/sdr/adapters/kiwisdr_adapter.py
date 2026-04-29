from __future__ import annotations

from datetime import datetime, timezone

from backend.sdr.base import BaseSdrAdapter, DeviceMetadata, SignalMetrics, SpectrumWindow


class KiwiSdrAdapter(BaseSdrAdapter):
    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_spectrum_window(self) -> SpectrumWindow:
        return SpectrumWindow(
            timestamp=datetime.now(timezone.utc).isoformat(),
            center_freq_hz=float(self.config.get("center_freq_hz", 7.1e6)),
            sample_rate_hz=float(self.config.get("sample_rate_hz", 12_000)),
            psd_bins_db=list(self.config.get("psd_bins_db", [-92.4, -90.8, -88.1, -89.0])),
        )

    def read_signal_metrics(self) -> SignalMetrics:
        return SignalMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            rssi_dbm=float(self.config.get("rssi_dbm", -89.5)),
            snr_db=float(self.config.get("snr_db", 11.2)),
        )

    def read_device_metadata(self) -> DeviceMetadata:
        return DeviceMetadata(
            provider="kiwisdr",
            device_id=self.config.get("device_id", "kiwi-node"),
            serial=self.config.get("serial"),
            gain_db=self.config.get("gain_db"),
            gps_lat=self.config.get("gps_lat"),
            gps_lon=self.config.get("gps_lon"),
            extras={"endpoint": self.config.get("endpoint", "localhost:8073")},
        )
