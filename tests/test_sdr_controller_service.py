from __future__ import annotations

from datetime import datetime, timezone

from backend.sdr.base import BaseSdrAdapter, DeviceMetadata, SignalMetrics, SpectrumWindow
from backend.sdr.controller_service import SdrControllerService


class DummyAdapter(BaseSdrAdapter):
    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_spectrum_window(self) -> SpectrumWindow:
        center = float(self.config.get("center_freq_hz", 100e6))
        sr = float(self.config.get("sample_rate_hz", 1e6))
        psd = [-95.0, -80.0, -70.0, -85.0]
        return SpectrumWindow(
            timestamp=datetime.now(timezone.utc).isoformat(),
            center_freq_hz=center,
            sample_rate_hz=sr,
            psd_bins_db=psd,
        )

    def read_signal_metrics(self) -> SignalMetrics:
        return SignalMetrics(timestamp=datetime.now(timezone.utc).isoformat(), rssi_dbm=-74.0, snr_db=22.0)

    def read_device_metadata(self) -> DeviceMetadata:
        return DeviceMetadata(provider="dummy", device_id="dummy-1")


def test_controller_reads_capabilities_sweeps_and_emits_normalized_observations():
    emitted: list[dict] = []
    adapter = DummyAdapter(
        config={
            "sample_rate_hz": 2e6,
            "max_span_hz": 1e6,
            "waterfall_resolution_bins": 4,
            "band_limits_hz": [
                {"name": "vhf", "start_hz": 118e6, "end_hz": 120e6},
                {"name": "uhf", "start_hz": 430e6, "end_hz": 432e6},
            ],
        }
    )

    service = SdrControllerService(adapter, emitted.append, dwell_seconds=0.001)

    first = service.process_once()
    second = service.process_once()

    assert first["emitted"] >= 1
    assert second["emitted"] >= 1
    assert emitted

    sample = emitted[0]
    assert sample["capabilities"]["max_span_hz"] == 1e6
    assert sample["capabilities"]["waterfall_resolution_bins"] == 4
    assert sample["band"] in {"vhf", "uhf"}
    assert sample["frames"]
    assert "waterfall_psd_bins_db" in sample["frames"][0]


def test_controller_tracks_quality_and_recency_per_band_for_scheduler_rebalancing():
    adapter = DummyAdapter(
        config={
            "sample_rate_hz": 2e6,
            "max_span_hz": 2e6,
            "band_limits_hz": [{"name": "wide", "start_hz": 100e6, "end_hz": 104e6}],
        }
    )

    service = SdrControllerService(adapter, lambda _: None, dwell_seconds=0.001)
    service.process_once()

    state = service.quality_state()
    assert "wide" in state
    assert state["wide"]["frame_count"] > 0
    assert state["wide"]["quality"] >= 0.0
    assert state["wide"]["last_seen_at"] is not None
    assert state["wide"]["recency_seconds"] >= 0.0
