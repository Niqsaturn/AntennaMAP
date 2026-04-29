import pytest

from backend.sdr.adapters.airspy_adapter import AirspyAdapter
from backend.sdr.adapters.hackrf_adapter import HackrfAdapter
from backend.sdr.adapters.kiwisdr_adapter import KiwiSdrAdapter
from backend.sdr.adapters.rtlsdr_adapter import RtlSdrAdapter
from backend.sdr.registry import create_sdr_adapter, create_sdr_adapter_from_config


@pytest.mark.parametrize(
    ("provider", "expected_cls"),
    [
        ("kiwisdr", KiwiSdrAdapter),
        ("rtlsdr", RtlSdrAdapter),
        ("airspy", AirspyAdapter),
        ("hackrf", HackrfAdapter),
    ],
)
def test_registry_selects_expected_adapter(provider, expected_cls):
    adapter = create_sdr_adapter(provider, config={"center_freq_hz": 100e6})
    assert isinstance(adapter, expected_cls)


def test_registry_builds_from_config_blob():
    adapter = create_sdr_adapter_from_config({"type": "rtlsdr", "settings": {"sample_rate_hz": 2.048e6}})
    assert isinstance(adapter, RtlSdrAdapter)


def test_registry_raises_for_unknown_provider():
    with pytest.raises(ValueError):
        create_sdr_adapter("unsupported")


@pytest.mark.parametrize("provider", ["kiwisdr", "rtlsdr", "airspy", "hackrf"])
def test_normalized_schema_is_compatible_across_adapters(provider):
    adapter = create_sdr_adapter(provider, config={"gps_lat": 40.0, "gps_lon": -105.0})
    adapter.connect()
    reading = adapter.read_normalized()

    assert isinstance(reading.timestamp, str)
    assert reading.center_freq_hz > 0
    assert reading.sample_rate_hz > 0
    assert isinstance(reading.psd_bins_db, list) and reading.psd_bins_db
    assert reading.rssi_dbm is None or isinstance(reading.rssi_dbm, float)
    assert reading.snr_db is None or isinstance(reading.snr_db, float)
    assert reading.gps == {"lat": 40.0, "lon": -105.0}
    assert reading.provider == provider
    assert "device_id" in reading.metadata

    adapter.disconnect()
    assert adapter.connected is False
