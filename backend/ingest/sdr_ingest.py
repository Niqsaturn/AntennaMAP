from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from backend.analysis.track_pipeline import derive_track_candidates

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class RFMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frequency_hz: float = Field(gt=0)
    bandwidth_hz: float = Field(gt=0)
    gain_db: float | None = None
    rssi_dbm: float
    snr_db: float


class SpectralSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    psd_bins_db: list[float] = Field(min_length=1)
    waterfall_avg_db: float
    waterfall_peak_db: float


class LocationHeading(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    heading_deg: float = Field(ge=0, le=360)
    fox_session_id: str | None = None


class SDRTelemetryPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: datetime
    device_id: str = Field(min_length=1)
    rf: RFMetadata
    spectral: SpectralSummary
    location: LocationHeading | None = None


@dataclass
class SDRStoragePaths:
    raw_jsonl: Path
    aggregates_jsonl: Path
    reject_jsonl: Path
    sqlite_file: Path


class SDRIngestService:
    def __init__(self, adapter_fetcher: Callable[[], list[dict[str, Any]]], storage: SDRStoragePaths, poll_interval_s: float = 1.0) -> None:
        self._adapter_fetcher = adapter_fetcher
        self._storage = storage
        self._poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_started_at: str | None = None
        self._last_stopped_at: str | None = None
        self._ingested_count = 0

    def _ensure_sqlite(self) -> None:
        self._storage.sqlite_file.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self._storage.sqlite_file)
        try:
            con.execute("CREATE TABLE IF NOT EXISTS sdr_raw (id INTEGER PRIMARY KEY AUTOINCREMENT, ingested_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
            con.execute(
                "CREATE TABLE IF NOT EXISTS sdr_aggregates (id INTEGER PRIMARY KEY AUTOINCREMENT, window_start TEXT NOT NULL, window_end TEXT NOT NULL, sample_count INTEGER NOT NULL, avg_rssi_dbm REAL, avg_snr_db REAL, avg_waterfall_db REAL, payload_json TEXT NOT NULL)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS track_candidates (track_id TEXT PRIMARY KEY, last_seen TEXT NOT NULL, confidence REAL NOT NULL, payload_json TEXT NOT NULL)"
            )
            con.commit()
        finally:
            con.close()

    @staticmethod
    def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, default=str))
                f.write("\n")

    def _persist(self, accepted: list[dict[str, Any]], aggregate: dict[str, Any] | None, tracks: list[dict[str, Any]]) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        raw_rows = [{"ingested_at": now, "payload": row} for row in accepted]
        self._append_jsonl(self._storage.raw_jsonl, raw_rows)
        if aggregate:
            self._append_jsonl(self._storage.aggregates_jsonl, [aggregate])
        if tracks:
            self._append_jsonl(self._storage.aggregates_jsonl, [{"track_candidates": tracks, "logged_at": now}])

        self._ensure_sqlite()
        con = sqlite3.connect(self._storage.sqlite_file)
        try:
            for row in raw_rows:
                con.execute("INSERT INTO sdr_raw (ingested_at, payload_json) VALUES (?, ?)", (row["ingested_at"], json.dumps(row["payload"], sort_keys=True, default=str)))
            if aggregate:
                con.execute(
                    "INSERT INTO sdr_aggregates (window_start, window_end, sample_count, avg_rssi_dbm, avg_snr_db, avg_waterfall_db, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        aggregate["window_start"],
                        aggregate["window_end"],
                        aggregate["sample_count"],
                        aggregate.get("avg_rssi_dbm"),
                        aggregate.get("avg_snr_db"),
                        aggregate.get("avg_waterfall_db"),
                        json.dumps(aggregate, sort_keys=True, default=str),
                    ),
                )
            for t in tracks:
                con.execute(
                    """INSERT INTO track_candidates (track_id, last_seen, confidence, payload_json) VALUES (?, ?, ?, ?)
                    ON CONFLICT(track_id) DO UPDATE SET last_seen=excluded.last_seen, confidence=excluded.confidence, payload_json=excluded.payload_json""",
                    (t["track_id"], t["timestamp"], t["confidence"], json.dumps(t, sort_keys=True, default=str)),
                )
            con.commit()
        finally:
            con.close()

    def _build_aggregate(self, accepted: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not accepted:
            return None
        timestamps = [datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) for r in accepted]
        return {
            "window_start": min(timestamps).isoformat(),
            "window_end": max(timestamps).isoformat(),
            "sample_count": len(accepted),
            "avg_rssi_dbm": round(mean(r["rf"]["rssi_dbm"] for r in accepted), 3),
            "avg_snr_db": round(mean(r["rf"]["snr_db"] for r in accepted), 3),
            "avg_waterfall_db": round(mean(r["spectral"]["waterfall_avg_db"] for r in accepted), 3),
        }

    def _process_once(self) -> dict[str, int]:
        packets = self._adapter_fetcher()
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for packet in packets:
            try:
                parsed = SDRTelemetryPacket.model_validate(packet)
                accepted.append(parsed.model_dump(mode="json"))
            except ValidationError as exc:
                rejected.append({"logged_at": datetime.now(tz=timezone.utc).isoformat(), "reason": "schema_validation", "errors": exc.errors(), "packet": packet})
        self._append_jsonl(self._storage.reject_jsonl, rejected)
        aggregate = self._build_aggregate(accepted)
        tracks = [t.__dict__ for t in derive_track_candidates(accepted)]
        self._persist(accepted, aggregate, tracks)
        self._ingested_count += len(accepted)
        return {"accepted": len(accepted), "rejected": len(rejected)}

    def _run(self) -> None:
        while self._running:
            with self._lock:
                self._process_once()
            time.sleep(self._poll_interval_s)

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._running:
                return {"running": True, "message": "already_running"}
            self._running = True
            self._last_started_at = datetime.now(tz=timezone.utc).isoformat()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return {"running": True, "message": "started", "started_at": self._last_started_at}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._running:
                return {"running": False, "message": "already_stopped"}
            self._running = False
            self._last_stopped_at = datetime.now(tz=timezone.utc).isoformat()
            return {"running": False, "message": "stopped", "stopped_at": self._last_stopped_at}

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "poll_interval_s": self._poll_interval_s,
            "ingested_count": self._ingested_count,
            "last_started_at": self._last_started_at,
            "last_stopped_at": self._last_stopped_at,
        }
