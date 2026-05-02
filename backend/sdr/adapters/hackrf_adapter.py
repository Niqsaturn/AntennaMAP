"""HackRF One virtual receiver model.

Generates mathematically computed spectrum data — no physical hardware or
USB drivers required.  Covers 1 MHz – 6 GHz with configurable bandwidth up to
20 MHz.
"""
from __future__ import annotations

from datetime import datetime, timezone

from backend.sdr.base import BaseSdrAdapter, DeviceMetadata, SignalMetrics, SpectrumWindow
from backend.sdr.computed_spectrum import computed_psd, psd_to_metrics


class HackrfAdapter(BaseSdrAdapter):
    """HackRF One model: 1 MHz – 6 GHz, up to 20 MHz BW."""

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_spectrum_window(self) -> SpectrumWindow:
        center = float(self.config.get("center_freq_hz", 915e6))
        sr = float(self.config.get("sample_rate_hz", 8e6))
        live_bins = self.config.get("live_api_bins_db")
        if isinstance(live_bins, list) and live_bins:
            psd = [float(v) for v in live_bins]
            provenance = "live_api"
        else:
            psd = computed_psd(
                center_freq_hz=center,
                sample_rate_hz=sr,
                n_bins=64,
                known_signals=self.config.get("known_signals"),
                config=self.config,
            )
            provenance = "synthetic"
        return SpectrumWindow(
            timestamp=datetime.now(timezone.utc).isoformat(),
            center_freq_hz=center,
            sample_rate_hz=sr,
            psd_bins_db=psd,
            source_provenance=provenance,
        )

    def read_signal_metrics(self) -> SignalMetrics:
        window = self.read_spectrum_window()
        rssi, snr = psd_to_metrics(window.psd_bins_db)
        return SignalMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            rssi_dbm=rssi,
            snr_db=snr,
        )

    def read_device_metadata(self) -> DeviceMetadata:
        return DeviceMetadata(
            provider="hackrf",
            device_id=self.config.get("device_id", "hackrf-virtual"),
            serial=self.config.get("serial"),
            gain_db=self.config.get("gain_db", 24.0),
            gps_lat=self.config.get("gps_lat"),
            gps_lon=self.config.get("gps_lon"),
            extras={
                "mode": "computed",
                "freq_range_mhz": "1–6000",
                "noise_figure_db": self.config.get("noise_figure_db", 8.0),
                "amp_enabled": self.config.get("amp_enabled", False),
            },
        )
