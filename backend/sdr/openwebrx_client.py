"""OpenWebRX WebSocket client for VHF/UHF spectrum data.

Connects to public OpenWebRX nodes, parses FFT frames, and produces
SpectralPeak lists compatible with the KiwiSDR pipeline.

OpenWebRX WebSocket protocol:
  - Client sends: "SERVER DE CLIENT client=openwebrx.js type=receiver"
  - Server sends: JSON frames with type "receiver_details" (config) and
                  binary frames (header byte + float32 FFT data)
  - Binary frame header byte 1 = FFT data (secondary_fft in newer versions)
"""
from __future__ import annotations

import base64
import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Known public OpenWebRX nodes (host, port, description)
_PUBLIC_NODES: list[tuple[str, int, str]] = [
    ("websdr.ewi.utwente.nl", 8901, "UTwente 0-29MHz HF"),
    ("kiwisdr.local", 8073, "Local KiwiSDR"),
]

_CONNECT_TIMEOUT_S = 8.0
_READ_TIMEOUT_S = 15.0


@dataclass
class OpenWebRXNode:
    host: str
    port: int
    description: str = ""
    center_freq_hz: float = 145e6
    bandwidth_hz: float = 2.4e6
    lat: float | None = None
    lon: float | None = None
    last_peaks: list[dict] = field(default_factory=list)
    last_update: float = 0.0
    reachable: bool = False


class OpenWebRXPool:
    """Manages connections to multiple OpenWebRX nodes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._nodes: list[OpenWebRXNode] = []
        self._initialized = False

    def _default_nodes(self) -> None:
        for host, port, desc in _PUBLIC_NODES:
            self._nodes.append(OpenWebRXNode(host=host, port=port, description=desc))

    def scan_peaks_all(self, min_snr_db: float = 6.0) -> list[dict]:
        """Return combined spectral peaks from all reachable OpenWebRX nodes."""
        with self._lock:
            nodes = list(self._nodes)
        if not nodes:
            return []

        results: list[dict] = []
        threads = []
        lock = threading.Lock()

        def _scan(node: OpenWebRXNode) -> None:
            try:
                peaks = _fetch_peaks(node, min_snr_db=min_snr_db)
                with lock:
                    results.extend(peaks)
                    node.last_peaks = peaks
                    node.last_update = time.monotonic()
                    node.reachable = True
            except Exception as exc:
                logger.debug("openwebrx %s:%d scan error: %s", node.host, node.port, exc)
                node.reachable = False

        for node in nodes:
            t = threading.Thread(target=_scan, args=(node,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=_READ_TIMEOUT_S + 2.0)
        return results

    def list_nodes(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "host": n.host, "port": n.port,
                    "description": n.description,
                    "center_freq_mhz": n.center_freq_hz / 1e6,
                    "bandwidth_mhz": n.bandwidth_hz / 1e6,
                    "reachable": n.reachable,
                    "last_peak_count": len(n.last_peaks),
                }
                for n in self._nodes
            ]

    def add_node(
        self,
        host: str,
        port: int = 8073,
        lat: float | None = None,
        lon: float | None = None,
        description: str = "",
        center_freq_hz: float = 145e6,
        bandwidth_hz: float = 2.4e6,
    ) -> None:
        with self._lock:
            self._nodes.append(OpenWebRXNode(
                host=host, port=port, description=description,
                center_freq_hz=center_freq_hz, bandwidth_hz=bandwidth_hz,
                lat=lat, lon=lon,
            ))


def _fetch_peaks(node: OpenWebRXNode, min_snr_db: float = 6.0) -> list[dict]:
    """Connect to one OpenWebRX node and pull a single FFT frame."""
    import websocket as _ws

    url = f"ws://{node.host}:{node.port}/ws/"
    bins_db: list[float] | None = None
    center_freq = node.center_freq_hz
    bandwidth = node.bandwidth_hz

    received = threading.Event()

    def on_message(ws: Any, message: Any) -> None:
        nonlocal bins_db, center_freq, bandwidth
        if isinstance(message, str):
            try:
                data = json.loads(message)
                if data.get("type") == "receiver_details":
                    center_freq = float(data.get("center_freq", center_freq))
                    bandwidth = float(data.get("samp_rate", bandwidth))
            except Exception:
                pass
        elif isinstance(message, bytes) and len(message) > 1:
            # Binary frame: first byte is message type; 1 = FFT
            msg_type = message[0]
            if msg_type == 1:
                n_bins = (len(message) - 1) // 4
                if n_bins >= 64:
                    bins_db = list(struct.unpack_from(f">{n_bins}f", message, 1))
                    received.set()
                    ws.close()

    def on_error(ws: Any, error: Any) -> None:
        received.set()

    def on_open(ws: Any) -> None:
        ws.send("SERVER DE CLIENT client=openwebrx.js type=receiver")

    ws_app = _ws.WebSocketApp(
        url,
        on_message=on_message,
        on_error=on_error,
        on_open=on_open,
    )
    t = threading.Thread(
        target=lambda: ws_app.run_forever(
            ping_interval=0,
            sockopt=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
        ),
        daemon=True,
    )
    t.start()
    received.wait(timeout=_READ_TIMEOUT_S)
    ws_app.close()

    if not bins_db:
        return []

    from backend.analysis.waterfall_analyzer import analyze_psd
    peaks = analyze_psd(bins_db, center_freq, bandwidth)
    return [
        {
            "freq_hz": p.center_freq_hz,
            "bandwidth_hz": p.bandwidth_3db_hz,
            "snr_db": p.snr_db,
            "peak_dbm": p.peak_dbm,
            "modulation_hint": p.modulation_hint,
            "source": "openwebrx",
            "node": f"{node.host}:{node.port}",
            "lat": node.lat,
            "lon": node.lon,
        }
        for p in peaks
        if p.snr_db >= min_snr_db
    ]


# Module-level pool
openwebrx_pool = OpenWebRXPool()
