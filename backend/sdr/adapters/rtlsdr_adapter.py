"""RTL-SDR virtual receiver model.

Generates mathematically computed spectrum data — no physical hardware or
USB drivers required.  All signal levels are derived from propagation models
and any known-signal metadata supplied in the adapter config.
"""
from __future__ import annotations

from datetime import datetime, timezone

from backend.sdr.base import BaseSdrAdapter, DeviceMetadata, SignalMetrics, SpectrumWindow
from backend.sdr.computed_spectrum import computed_psd, psd_to_metrics


class RtlSdrAdapter(BaseSdrAdapter):
    """RTL-SDR model: VHF/UHF coverage 24 MHz – 1.766 GHz, 2.4 MHz default BW."""

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_spectrum_window(self) -> SpectrumWindow:
        center = float(self.config.get("center_freq_hz", 433.92e6))
        sr = float(self.config.get("sample_rate_hz", 2.4e6))
        psd = computed_psd(
            center_freq_hz=center,
            sample_rate_hz=sr,
            n_bins=64,
            known_signals=self.config.get("known_signals"),
            config=self.config,
        )
        return SpectrumWindow(
            timestamp=datetime.now(timezone.utc).isoformat(),
            center_freq_hz=center,
            sample_rate_hz=sr,
            psd_bins_db=psd,
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
            provider="rtlsdr",
            device_id=self.config.get("device_id", "rtlsdr-virtual"),
            serial=self.config.get("serial"),
            gain_db=self.config.get("gain_db", 20.7),
            gps_lat=self.config.get("gps_lat"),
            gps_lon=self.config.get("gps_lon"),
            extras={
                "mode": "computed",
                "freq_range_mhz": "24–1766",
                "noise_figure_db": self.config.get("noise_figure_db", 5.0),
            },
        )
