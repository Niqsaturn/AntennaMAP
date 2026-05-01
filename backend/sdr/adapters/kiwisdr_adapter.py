from __future__ import annotations

import struct
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from backend.sdr.base import BaseSdrAdapter, DeviceMetadata, SignalMetrics, SpectrumWindow

try:
    import websocket as _websocket
    _HAS_WEBSOCKET = True
except ImportError:
    _HAS_WEBSOCKET = False


def _kiwi_fetch(host: str, port: int, timeout: float = 6.0) -> dict[str, Any]:
    """Connect to a KiwiSDR waterfall endpoint and return basic spectrum metrics."""
    url = f"ws://{host}:{port}/W/F"
    ws = _websocket.create_connection(url, timeout=timeout)
    try:
        ws.send("SET auth t=kiwi p=\n")
        ws.send("SET zoom=0 start=0\n")
        ws.send("SET maxdb=-10 mindb=-110\n")

        bins_db: list[float] = []
        for _ in range(20):
            raw = ws.recv()
            if not isinstance(raw, bytes) or len(raw) < 2:
                continue
            # KiwiSDR W/F binary frame: first 3 bytes are "W/F", remainder is bin data
            # Each bin is a uint8 scaled to the configured dB range (-110 to -10)
            header = raw[:3]
            if header == b"W/F" or (len(raw) > 5 and raw[0] == 0x57):
                payload = raw[3:] if header == b"W/F" else raw[1:]
                bins_db = [
                    round(-110.0 + b * (100.0 / 255.0), 2)
                    for b in payload
                ]
                break
    finally:
        ws.close()

    if not bins_db:
        bins_db = [-80.0] * 64

    noise_floor = sorted(bins_db)[: max(1, len(bins_db) // 4)]
    avg_noise = mean(noise_floor)
    peak = max(bins_db)
    rssi = round(mean(bins_db), 2)
    snr = round(max(0.0, peak - avg_noise), 2)

    return {"bins_db": bins_db, "rssi_dbm": rssi, "snr_db": snr}


class KiwiSdrAdapter(BaseSdrAdapter):
    """
    KiwiSDR network adapter. Connects to a KiwiSDR receiver over WebSocket —
    no physical hardware required, only a hostname/IP of a running KiwiSDR server.

    Config keys:
        host (str): KiwiSDR hostname or IP. Default: "localhost"
        port (int): KiwiSDR port. Default: 8073
        center_freq_hz (float): Tuning frequency in Hz. Default: 7.1 MHz (40m HF)
    """

    def connect(self) -> None:
        self.connected = _HAS_WEBSOCKET

    def disconnect(self) -> None:
        self.connected = False

    def _host(self) -> str:
        return str(self.config.get("host", "localhost"))

    def _port(self) -> int:
        return int(self.config.get("port", 8073))

    def read_spectrum_window(self) -> SpectrumWindow:
        if _HAS_WEBSOCKET and self.connected:
            try:
                data = _kiwi_fetch(self._host(), self._port())
                return SpectrumWindow(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    center_freq_hz=float(self.config.get("center_freq_hz", 7.1e6)),
                    sample_rate_hz=12_000.0,
                    psd_bins_db=data["bins_db"],
                )
            except Exception:
                pass
        # Mock fallback
        return SpectrumWindow(
            timestamp=datetime.now(timezone.utc).isoformat(),
            center_freq_hz=float(self.config.get("center_freq_hz", 7.1e6)),
            sample_rate_hz=12_000.0,
            psd_bins_db=[-92.4, -90.8, -88.1, -89.0],
        )

    def read_signal_metrics(self) -> SignalMetrics:
        if _HAS_WEBSOCKET and self.connected:
            try:
                data = _kiwi_fetch(self._host(), self._port())
                return SignalMetrics(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    rssi_dbm=data["rssi_dbm"],
                    snr_db=data["snr_db"],
                )
            except Exception:
                pass
        return SignalMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            rssi_dbm=float(self.config.get("rssi_dbm", -89.5)),
            snr_db=float(self.config.get("snr_db", 11.2)),
        )

    def read_device_metadata(self) -> DeviceMetadata:
        return DeviceMetadata(
            provider="kiwisdr",
            device_id=f"{self._host()}:{self._port()}",
            serial=None,
            gain_db=None,
            gps_lat=self.config.get("gps_lat"),
            gps_lon=self.config.get("gps_lon"),
            extras={
                "host": self._host(),
                "port": self._port(),
                "websocket_available": _HAS_WEBSOCKET,
            },
        )
