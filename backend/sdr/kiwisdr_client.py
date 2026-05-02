"""Full KiwiSDR WebSocket streaming client.

Handles the KiwiSDR binary waterfall protocol:
  - Connects to ws://host:port/W/F
  - Sends auth + zoom commands
  - Parses binary W/F frames → 1024-bin dB arrays
  - Provides synchronous RSSI sampling and peak scanning
  - Supports background streaming with a callback

Multi-node scanning feeds RSSI observations into multilateration.
"""
from __future__ import annotations

import asyncio
import logging
import math
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import websocket as _ws
    _HAS_WS = True
except ImportError:
    _HAS_WS = False

# KiwiSDR full HF passband
_KIWI_PASSBAND_HZ = 30_000_000.0
_KIWI_BINS = 1024

# dB mapping: KiwiSDR uint8 bin → dBFS
# uint8 0 → -127 dBFS, 255 → -13 dBFS  (empirical for maxdb=-10 mindb=-127)
_DB_MIN = -127.0
_DB_RANGE = 114.0   # _DB_MIN + _DB_RANGE = -13 dBm


@dataclass
class WaterfallFrame:
    timestamp: str
    passband_hz: float
    bins_db: list[float]          # 1024 values
    noise_floor_db: float
    peak_db: float
    peak_freq_hz: float
    host: str = ""
    port: int = 8073


@dataclass
class SignalPeak:
    freq_hz: float
    bandwidth_hz: float
    peak_dbm: float
    snr_db: float
    modulation_hint: str


@dataclass
class NodeStatus:
    host: str
    port: int
    reachable: bool
    last_seen: str = ""
    lat: float | None = None
    lon: float | None = None
    gps_locked: bool = False
    description: str = ""


# ── Frame parsing ─────────────────────────────────────────────────────────────

def _parse_wf_frame(raw: bytes) -> list[float] | None:
    """Extract 1024 dB-scaled bins from a KiwiSDR binary W/F frame.

    KiwiSDR waterfall frame layout (varies by firmware, two known variants):
      Variant A: b"W/F" + 1 flag byte + content
      Variant B: single 0x57 ('W') byte + content
    Content ends with exactly 1024 uint8 bin values (the last 1024 bytes).
    """
    if not raw or len(raw) < 16:
        return None

    # Locate the bin payload: last 1024 bytes after any header
    if raw[:3] == b"W/F":
        payload = raw[3:]
    elif raw[0] == 0x57:   # 'W'
        payload = raw[1:]
    else:
        payload = raw

    if len(payload) < 1024:
        return None

    bins_raw = payload[-1024:]
    return [round(_DB_MIN + b * (_DB_RANGE / 255.0), 2) for b in bins_raw]


def _bins_to_frame(bins: list[float], host: str = "", port: int = 8073) -> WaterfallFrame:
    noise_sample = sorted(bins)[: max(1, len(bins) // 5)]
    noise_floor = mean(noise_sample)
    peak_db = max(bins)
    peak_idx = bins.index(peak_db)
    peak_freq = peak_idx * (_KIWI_PASSBAND_HZ / _KIWI_BINS)
    return WaterfallFrame(
        timestamp=datetime.now(timezone.utc).isoformat(),
        passband_hz=_KIWI_PASSBAND_HZ,
        bins_db=bins,
        noise_floor_db=round(noise_floor, 2),
        peak_db=round(peak_db, 2),
        peak_freq_hz=round(peak_freq, 0),
        host=host,
        port=port,
    )


def _rssi_at_freq(frames: list[WaterfallFrame], freq_hz: float, window_bins: int = 5) -> float | None:
    """Average dB value at freq_hz across frames, ±window_bins around centre bin."""
    if not frames:
        return None
    bin_width = _KIWI_PASSBAND_HZ / _KIWI_BINS
    centre = int(freq_hz / bin_width)
    centre = max(window_bins, min(centre, _KIWI_BINS - 1 - window_bins))
    vals = []
    for f in frames:
        vals.extend(f.bins_db[centre - window_bins: centre + window_bins + 1])
    return round(mean(vals), 2) if vals else None


# ── Single KiwiSDR client ─────────────────────────────────────────────────────

class KiwiSdrClient:
    """Connect to a single KiwiSDR node and stream waterfall frames."""

    def __init__(self, host: str, port: int = 8073, timeout: float = 8.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.gps_lat: float | None = None
        self.gps_lon: float | None = None
        self.gps_locked: bool = False
        self._frames: list[WaterfallFrame] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/W/F"

    def _init_ws(self, ws: Any) -> None:
        ws.send("SET auth t=kiwi p=\n")
        time.sleep(0.1)
        ws.send("SET zoom=0 start=0\n")
        ws.send("SET maxdb=-10 mindb=-127\n")

    def _recv_frames(self, count: int, ws: Any) -> list[WaterfallFrame]:
        frames: list[WaterfallFrame] = []
        attempts = 0
        while len(frames) < count and attempts < count * 10:
            attempts += 1
            try:
                raw = ws.recv()
            except Exception:
                break
            if isinstance(raw, bytes):
                bins = _parse_wf_frame(raw)
                if bins:
                    frames.append(_bins_to_frame(bins, self.host, self.port))
            elif isinstance(raw, str) and "gps" in raw.lower():
                # Parse GPS position from MSG frame if present
                try:
                    parts = dict(p.split("=") for p in raw.split() if "=" in p)
                    if "lat" in parts and "lon" in parts:
                        self.gps_lat = float(parts["lat"])
                        self.gps_lon = float(parts["lon"])
                        self.gps_locked = True
                except Exception:
                    pass
        return frames

    def read_frames(self, count: int = 4) -> list[WaterfallFrame]:
        """Synchronously read `count` waterfall frames."""
        if not _HAS_WS:
            return []
        try:
            ws = _ws.create_connection(self.url, timeout=self.timeout)
            self._init_ws(ws)
            frames = self._recv_frames(count, ws)
            ws.close()
            return frames
        except Exception as exc:
            logger.debug("kiwisdr %s:%s read_frames: %s", self.host, self.port, exc)
            return []

    def read_rssi(self, freq_hz: float) -> float | None:
        """Sample RSSI (dBm proxy) at a specific frequency from this node."""
        frames = self.read_frames(count=3)
        return _rssi_at_freq(frames, freq_hz)

    def scan_peaks(self, min_snr_db: float = 6.0) -> list[SignalPeak]:
        """Scan full 0–30 MHz band and return signal peaks above noise."""
        frames = self.read_frames(count=4)
        if not frames:
            return []
        avg_bins = [mean(f.bins_db[i] for f in frames) for i in range(_KIWI_BINS)]
        try:
            from backend.analysis.waterfall_analyzer import analyze_psd
            raw_peaks = analyze_psd(
                avg_bins,
                center_freq_hz=_KIWI_PASSBAND_HZ / 2,
                sample_rate_hz=_KIWI_PASSBAND_HZ,
            )
            return [
                SignalPeak(
                    freq_hz=p.center_freq_hz,
                    bandwidth_hz=p.bandwidth_3db_hz,
                    peak_dbm=p.peak_dbm,
                    snr_db=p.snr_db,
                    modulation_hint=p.modulation_hint,
                )
                for p in raw_peaks
                if p.snr_db >= min_snr_db
            ]
        except Exception as exc:
            logger.debug("kiwisdr %s:%s scan_peaks: %s", self.host, self.port, exc)
            return []

    def check_reachable(self) -> tuple[bool, float | None]:
        """TCP-probe the node. Returns (reachable, latency_ms)."""
        t0 = time.monotonic()
        try:
            sock = socket.create_connection((self.host, self.port), timeout=3.0)
            sock.close()
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            return True, latency_ms
        except OSError as exc:
            logger.debug("kiwisdr %s:%s unreachable: %s", self.host, self.port, exc)
            return False, None

    def start_streaming(self, callback: Callable[[WaterfallFrame], None]) -> None:
        """Background-thread streaming: calls callback for each parsed frame."""
        if not _HAS_WS or self._running:
            return
        self._running = True

        def _loop() -> None:
            while self._running:
                try:
                    ws = _ws.create_connection(self.url, timeout=self.timeout)
                    self._init_ws(ws)
                    while self._running:
                        try:
                            raw = ws.recv()
                        except Exception:
                            break
                        if isinstance(raw, bytes):
                            bins = _parse_wf_frame(raw)
                            if bins:
                                frame = _bins_to_frame(bins, self.host, self.port)
                                with self._lock:
                                    self._frames.append(frame)
                                    if len(self._frames) > 300:
                                        self._frames = self._frames[-300:]
                                try:
                                    callback(frame)
                                except Exception as exc:
                                    logger.warning("kiwisdr %s callback error: %s", self.host, exc)
                    ws.close()
                except Exception as exc:
                    if self._running:
                        logger.warning("kiwisdr %s stream reconnect: %s", self.host, exc)
                        time.sleep(3.0)

        self._thread = threading.Thread(target=_loop, daemon=True, name=f"kiwi-{self.host}")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def latest_frames(self, n: int = 20) -> list[WaterfallFrame]:
        with self._lock:
            return list(self._frames[-n:])

    def avg_bins(self, n: int = 5) -> list[float] | None:
        """Average of the last n waterfall frames (for display and peak detection)."""
        frames = self.latest_frames(n)
        if not frames:
            return None
        return [mean(f.bins_db[i] for f in frames) for i in range(_KIWI_BINS)]


# ── Multi-node manager ────────────────────────────────────────────────────────

class KiwiNodePool:
    """Manages a set of KiwiSDR nodes and queries them in parallel.

    Default nodes include a few well-known publicly accessible KiwiSDR
    receivers.  Users can add their own nodes via the API.
    """

    # Curated seed list of known-good public KiwiSDR nodes — geographic spread
    # across NA, SA, EU, AF, AS, AU so the map has global coverage from startup.
    DEFAULT_NODES: list[dict] = [
        # North America
        {"host": "kiwi.k9an.net",          "port": 8073, "lat": 41.85,  "lon": -88.21, "description": "Illinois USA"},
        {"host": "kiwisdr.wi2xsr.net",      "port": 8073, "lat": 43.07,  "lon": -89.39, "description": "Wisconsin USA"},
        {"host": "kiwi.w4ax.com",           "port": 8073, "lat": 33.75,  "lon": -84.39, "description": "Georgia USA"},
        {"host": "sdr.w3pm.net",            "port": 8073, "lat": 40.44,  "lon": -79.99, "description": "Pittsburgh USA"},
        {"host": "kiwi.ve3ien.ca",          "port": 8073, "lat": 43.70,  "lon": -79.41, "description": "Toronto CA"},
        {"host": "sdr.kf5hf.com",           "port": 8073, "lat": 32.76,  "lon": -97.33, "description": "Dallas USA"},
        {"host": "kiwisdr.k7mdl.com",       "port": 8073, "lat": 47.51,  "lon": -122.04,"description": "Seattle USA"},
        {"host": "ka7oei-kiwi.duckdns.org", "port": 8073, "lat": 40.76,  "lon": -111.89,"description": "Salt Lake City USA"},
        # South America
        {"host": "kiwisdr.lu9dce.com",      "port": 8073, "lat": -34.60, "lon": -58.37, "description": "Buenos Aires AR"},
        {"host": "kiwi.py2rrb.com",         "port": 8073, "lat": -22.91, "lon": -43.17, "description": "Rio de Janeiro BR"},
        # Europe
        {"host": "websdr.ewi.utwente.nl",   "port": 8073, "lat": 52.238, "lon": 6.857,  "description": "Enschede NL"},
        {"host": "rx.linkfanel.net",         "port": 8073, "lat": 48.86,  "lon": 2.35,   "description": "Paris FR"},
        {"host": "kiwi.f5uii.net",           "port": 8073, "lat": 43.30,  "lon": 5.37,   "description": "Marseille FR"},
        {"host": "sdr.dk5ec.de",             "port": 8073, "lat": 53.55,  "lon": 10.00,  "description": "Hamburg DE"},
        {"host": "ka1mdo-kiwi.mddns.eu",    "port": 8073, "lat": 50.08,  "lon": 14.43,  "description": "Prague CZ"},
        {"host": "kiwisdr.ddns.net",         "port": 8073, "lat": 37.98,  "lon": 23.72,  "description": "Athens GR"},
        {"host": "kiwi.g8jnj.net",          "port": 8073, "lat": 51.46,  "lon": -2.60,  "description": "Bristol UK"},
        {"host": "sdr.sm2byc.se",           "port": 8073, "lat": 63.83,  "lon": 20.26,  "description": "Umea SE"},
        # Africa
        {"host": "kiwi.zs6bkw.net",        "port": 8073, "lat": -25.74, "lon": 28.18,  "description": "Pretoria ZA"},
        # Middle East / Asia
        {"host": "sdr.4x1rf.com",           "port": 8073, "lat": 31.77,  "lon": 35.21,  "description": "Jerusalem IL"},
        {"host": "kiwisdr.oh6bgs.com",      "port": 8073, "lat": 25.20,  "lon": 55.27,  "description": "Dubai AE"},
        {"host": "kiwi.vk6fh.com",          "port": 8073, "lat": -31.95, "lon": 115.86, "description": "Perth AU"},
        {"host": "sdr.vk2kfj.com",          "port": 8073, "lat": -33.87, "lon": 151.21, "description": "Sydney AU"},
        {"host": "kiwisdr.ja8cjy.com",      "port": 8073, "lat": 43.06,  "lon": 141.35, "description": "Sapporo JP"},
        {"host": "kiwi.bd7mq.com",          "port": 8073, "lat": 23.13,  "lon": 113.26, "description": "Guangzhou CN"},
    ]

    def __init__(self) -> None:
        self._nodes: list[dict] = []
        self._lock = threading.Lock()
        self._frame_seq_by_node: dict[str, int] = {}

    def add_node(
        self,
        host: str,
        port: int = 8073,
        lat: float | None = None,
        lon: float | None = None,
        description: str = "",
    ) -> None:
        with self._lock:
            # Deduplicate by host:port
            existing = {f"{n['host']}:{n['port']}" for n in self._nodes}
            if f"{host}:{port}" not in existing:
                self._nodes.append(
                    {"host": host, "port": port, "lat": lat, "lon": lon,
                     "description": description}
                )

    def remove_node(self, host: str, port: int = 8073) -> None:
        with self._lock:
            self._nodes = [n for n in self._nodes if not (n["host"] == host and n["port"] == port)]

    def list_nodes(self) -> list[dict]:
        with self._lock:
            return list(self._nodes)

    def _next_frame_seq(self, node_key: str) -> int:
        with self._lock:
            seq = self._frame_seq_by_node.get(node_key, 0) + 1
            self._frame_seq_by_node[node_key] = seq
        return seq

    def scan_rssi(self, freq_hz: float, timeout: float = 8.0) -> list[dict]:
        """Query all configured nodes in parallel and return RSSI readings.

        Returns: [{"host", "port", "lat", "lon", "rssi_dbm", "timestamp"}, ...]
        """
        nodes = self.list_nodes()
        results: list[dict] = []
        result_lock = threading.Lock()

        def _query(node: dict) -> None:
            client = KiwiSdrClient(node["host"], node["port"], timeout=timeout)
            rssi = client.read_rssi(freq_hz)
            if rssi is not None:
                with result_lock:
                    results.append({
                        "host": node["host"],
                        "port": node["port"],
                        "lat": node.get("lat"),
                        "lon": node.get("lon"),
                        "rssi_dbm": rssi,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "freq_hz": freq_hz,
                    })

        threads = [threading.Thread(target=_query, args=(n,), daemon=True) for n in nodes]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout + 1.0)
        return results

    def scan_peaks_all(self, min_snr_db: float = 6.0) -> list[dict]:
        """Scan all nodes for signal peaks; union the results."""
        nodes = self.list_nodes()
        all_peaks: list[dict] = []
        peak_lock = threading.Lock()

        def _query(node: dict) -> None:
            client = KiwiSdrClient(node["host"], node["port"])
            frames = client.read_frames(count=4)
            if frames:
                avg_bins = [mean(f.bins_db[i] for f in frames) for i in range(_KIWI_BINS)]
                try:
                    from backend.foxhunt.auto_loop import event_bus
                    node_key = f"{node['host']}:{node['port']}"
                    frame_seq = self._next_frame_seq(node_key)
                    event_bus.publish({
                        "type": "sdr_frame",
                        "source": {
                            "node": node["host"],
                            "host": node["host"],
                            "port": node["port"],
                        },
                        "frame_schema_version": 1,
                        "frame_seq": frame_seq,
                        "center_freq_hz": _KIWI_PASSBAND_HZ / 2,
                        "span_hz": _KIWI_PASSBAND_HZ,
                        "sample_rate_hz": _KIWI_PASSBAND_HZ,
                        "fft_bins": avg_bins,
                        "fft_bin_count": len(avg_bins),
                        "calibration_offsets": {
                            "freq_offset_hz": 0.0,
                            "gain_offset_db": 0.0,
                            "noise_floor_offset_db": 0.0,
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception:
                    pass
            peaks = client.scan_peaks(min_snr_db)
            with peak_lock:
                for p in peaks:
                    all_peaks.append({
                        "host": node["host"], "port": node["port"],
                        "lat": node.get("lat"), "lon": node.get("lon"),
                        "freq_hz": p.freq_hz, "bw_hz": p.bandwidth_hz,
                        "peak_dbm": p.peak_dbm, "snr_db": p.snr_db,
                        "modulation_hint": p.modulation_hint,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

        threads = [threading.Thread(target=_query, args=(n,), daemon=True) for n in nodes]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=12.0)
        return all_peaks

    def check_all(self) -> list[NodeStatus]:
        """Probe reachability of all configured nodes via TCP."""
        nodes = self.list_nodes()
        statuses: list[NodeStatus] = []
        lock = threading.Lock()

        def _check(node: dict) -> None:
            client = KiwiSdrClient(node["host"], node["port"], timeout=4.0)
            ok, latency_ms = client.check_reachable()
            with lock:
                statuses.append(NodeStatus(
                    host=node["host"], port=node["port"],
                    reachable=ok,
                    last_seen=datetime.now(timezone.utc).isoformat() if ok else "",
                    lat=node.get("lat"), lon=node.get("lon"),
                    description=node.get("description", ""),
                ))

        threads = [threading.Thread(target=_check, args=(n,), daemon=True) for n in nodes]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=6.0)
        return statuses

    async def auto_populate(self, limit: int = 150) -> int:
        """Fetch the public KiwiSDR directory and add new nodes.

        Returns the count of newly added nodes.
        """
        discovered = await fetch_public_directory(limit)
        added = 0
        for nd in discovered:
            before = len(self._nodes)
            self.add_node(nd["host"], nd["port"], nd.get("lat"), nd.get("lon"), nd.get("description", ""))
            if len(self._nodes) > before:
                added += 1
        if discovered:
            logger.info("KiwiSDR auto-populate: +%d new nodes (%d total)", added, len(self._nodes))
        return added


async def fetch_public_directory(limit: int = 150) -> list[dict]:
    """Fetch the public KiwiSDR receiver directory from rx.linkfanel.net.

    Returns list of dicts with host, port, lat, lon, description.
    Sorted by fewest current users (most available first).
    """
    try:
        import httpx as _httpx
        url = "https://rx.linkfanel.net/receivers.json"
        async with _httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            entries = resp.json()
    except Exception as exc:
        logger.debug("KiwiSDR directory fetch failed: %s", exc)
        return []

    nodes = []
    for entry in entries:
        try:
            raw_url = entry.get("url", "")
            # url is like "http://host:port" or "http://host:port/"
            raw_url = raw_url.rstrip("/")
            if "://" in raw_url:
                raw_url = raw_url.split("://", 1)[1]
            if ":" in raw_url:
                host, port_str = raw_url.rsplit(":", 1)
                port = int(port_str)
            else:
                host = raw_url
                port = 8073
            if not host:
                continue
            gps = entry.get("gps") or []
            lat = float(gps[0]) if len(gps) >= 1 else None
            lon = float(gps[1]) if len(gps) >= 2 else None
            users = int(entry.get("users", 0))
            users_max = int(entry.get("users_max", 4))
            if users >= users_max:
                continue  # skip full receivers
            nodes.append({
                "host": host, "port": port,
                "lat": lat, "lon": lon,
                "description": entry.get("name", ""),
                "_users": users,
            })
        except Exception:
            continue

    nodes.sort(key=lambda n: n.get("_users", 0))
    for nd in nodes:
        nd.pop("_users", None)
    return nodes[:limit]


# ── Module-level singleton ────────────────────────────────────────────────────
node_pool = KiwiNodePool()
# Seed with defaults
for _nd in KiwiNodePool.DEFAULT_NODES:
    node_pool.add_node(_nd["host"], _nd["port"], _nd.get("lat"), _nd.get("lon"), _nd.get("description", ""))
