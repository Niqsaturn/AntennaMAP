"""SQLite-backed persistent map feature store.

Replaces static GeoJSON for speculative/estimated features so the dataset
can grow to US-wide scale without loading everything into memory.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "backend" / "storage" / "data" / "map_store.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    freq_band TEXT,
    confidence REAL DEFAULT 0.5,
    analysis_count INTEGER DEFAULT 1,
    properties_json TEXT NOT NULL,
    kalman_state_json TEXT,
    calibration_error_m REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_geo ON features(lat, lon);
CREATE INDEX IF NOT EXISTS idx_kind ON features(kind);

CREATE TABLE IF NOT EXISTS coverage_tiles (
    tile_id TEXT PRIMARY KEY,
    lat_min REAL NOT NULL,
    lon_min REAL NOT NULL,
    lat_max REAL NOT NULL,
    lon_max REAL NOT NULL,
    status TEXT DEFAULT 'unanalyzed',
    feature_count INTEGER DEFAULT 0,
    seeded_at TEXT,
    analyzed_at TEXT
);

CREATE TABLE IF NOT EXISTS analysis_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model TEXT,
    provider TEXT,
    input_summary TEXT,
    detections_count INTEGER DEFAULT 0,
    detections_json TEXT,
    raw_response TEXT,
    processing_ms REAL
);
"""

_MIGRATIONS = [
    "ALTER TABLE features ADD COLUMN kalman_state_json TEXT",
    "ALTER TABLE features ADD COLUMN calibration_error_m REAL",
]


def _apply_migrations(con: sqlite3.Connection) -> None:
    """Add new columns to existing databases without dropping data."""
    for stmt in _MIGRATIONS:
        try:
            con.execute(stmt)
        except sqlite3.OperationalError:
            pass  # column already exists


def _ensure_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(_SCHEMA)
        _apply_migrations(con)
        yield con
        con.commit()
    finally:
        con.close()


def upsert_feature(feature: dict[str, Any]) -> None:
    """Insert or update a GeoJSON feature (confidence accumulation via EMA)."""
    props = feature.get("properties", {})
    geom = feature.get("geometry", {})
    coords = geom.get("coordinates", [0.0, 0.0])
    lon, lat = float(coords[0]), float(coords[1])
    fid = props.get("id") or props.get("feature_id") or f"feat_{lat:.4f}_{lon:.4f}"
    kind = str(props.get("kind", "speculative"))
    freq_band = props.get("freq_band") or props.get("band")
    new_confidence = float(props.get("confidence", 0.5))
    now = datetime.now(timezone.utc).isoformat()

    with _conn() as con:
        existing = con.execute(
            "SELECT confidence, analysis_count FROM features WHERE id = ?", (fid,)
        ).fetchone()
        if existing:
            old_conf = existing["confidence"]
            count = existing["analysis_count"] + 1
            # EMA: blend new confidence toward old
            merged_conf = old_conf + (new_confidence - old_conf) * 0.3
            # Promotion: high confidence + enough observations
            if merged_conf >= 0.85 and count >= 3 and kind == "speculative":
                kind = "estimate"
            # Dismissal: very low confidence after multiple analyses
            if merged_conf < 0.1 and count >= 5:
                con.execute("DELETE FROM features WHERE id = ?", (fid,))
                return
            props["confidence"] = round(merged_conf, 4)
            props["analysis_count"] = count
            con.execute(
                """UPDATE features SET kind=?, lat=?, lon=?, freq_band=?,
                   confidence=?, analysis_count=?, properties_json=?, updated_at=?
                   WHERE id=?""",
                (kind, lat, lon, freq_band, round(merged_conf, 4), count,
                 json.dumps(props), now, fid),
            )
        else:
            props["confidence"] = new_confidence
            props["analysis_count"] = 1
            con.execute(
                """INSERT INTO features
                   (id, kind, lat, lon, freq_band, confidence, analysis_count,
                    properties_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (fid, kind, lat, lon, freq_band, new_confidence,
                 json.dumps(props), now, now),
            )


def features_in_bounds(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    kind: str | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    """Return GeoJSON FeatureCollection for features inside a bounding box."""
    with _conn() as con:
        if kind:
            rows = con.execute(
                """SELECT * FROM features
                   WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? AND kind=?
                   LIMIT ?""",
                (lat_min, lat_max, lon_min, lon_max, kind, limit),
            ).fetchall()
        else:
            rows = con.execute(
                """SELECT * FROM features
                   WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
                   LIMIT ?""",
                (lat_min, lat_max, lon_min, lon_max, limit),
            ).fetchall()

    features = []
    for row in rows:
        props = json.loads(row["properties_json"])
        props["id"] = row["id"]
        props["kind"] = row["kind"]
        props["confidence"] = row["confidence"]
        props["analysis_count"] = row["analysis_count"]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def get_all_features(kind: str | None = None, limit: int = 2000) -> dict[str, Any]:
    """Return all features (or by kind) as a GeoJSON FeatureCollection."""
    return features_in_bounds(-90, 90, -180, 180, kind=kind, limit=limit)


def upsert_tile(tile_id: str, lat_min: float, lon_min: float,
                lat_max: float, lon_max: float, status: str = "unanalyzed",
                feature_count: int = 0) -> None:
    now = datetime.now(timezone.utc).isoformat()
    seeded_at = now if status in ("seeded", "analyzed") else None
    analyzed_at = now if status == "analyzed" else None
    with _conn() as con:
        con.execute(
            """INSERT INTO coverage_tiles
               (tile_id, lat_min, lon_min, lat_max, lon_max, status, feature_count,
                seeded_at, analyzed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(tile_id) DO UPDATE SET
               status=excluded.status, feature_count=excluded.feature_count,
               seeded_at=COALESCE(excluded.seeded_at, coverage_tiles.seeded_at),
               analyzed_at=COALESCE(excluded.analyzed_at, coverage_tiles.analyzed_at)""",
            (tile_id, lat_min, lon_min, lat_max, lon_max, status, feature_count,
             seeded_at, analyzed_at),
        )


def get_coverage_grid() -> dict[str, Any]:
    """Return coverage tiles as a GeoJSON FeatureCollection."""
    with _conn() as con:
        rows = con.execute("SELECT * FROM coverage_tiles").fetchall()
    features = []
    for row in rows:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [row["lon_min"], row["lat_min"]],
                    [row["lon_max"], row["lat_min"]],
                    [row["lon_max"], row["lat_max"]],
                    [row["lon_min"], row["lat_max"]],
                    [row["lon_min"], row["lat_min"]],
                ]],
            },
            "properties": {
                "tile_id": row["tile_id"],
                "status": row["status"],
                "feature_count": row["feature_count"],
                "seeded_at": row["seeded_at"],
                "analyzed_at": row["analyzed_at"],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def get_coverage_progress() -> dict[str, Any]:
    """Return CONUS coverage statistics."""
    with _conn() as con:
        rows = con.execute(
            "SELECT status, COUNT(*) as cnt FROM coverage_tiles GROUP BY status"
        ).fetchall()
    counts = {row["status"]: row["cnt"] for row in rows}
    total_tiles = sum(counts.values())
    analyzed = counts.get("analyzed", 0)
    seeded = counts.get("seeded", 0)
    conus_total = 5850  # 0.5° grid over CONUS
    return {
        "total_tiles_tracked": total_tiles,
        "conus_total_tiles": conus_total,
        "analyzed": analyzed,
        "seeded": seeded,
        "unanalyzed": conus_total - analyzed - seeded,
        "percent_analyzed": round(analyzed / conus_total * 100, 2),
        "percent_seeded": round(seeded / conus_total * 100, 2),
        "status_counts": counts,
    }


def log_analysis(
    model: str,
    provider: str,
    input_summary: str,
    detections_count: int,
    detections_json: str,
    raw_response: str,
    processing_ms: float,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        con.execute(
            """INSERT INTO analysis_log
               (timestamp, model, provider, input_summary, detections_count,
                detections_json, raw_response, processing_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, model, provider, input_summary, detections_count,
             detections_json, raw_response, processing_ms),
        )


def get_analysis_log(limit: int = 20) -> list[dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM analysis_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def update_feature_kind(feature_id: str, kind: str) -> bool:
    """Manually set a feature's kind (e.g. confirm/dismiss)."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        if kind == "dismissed":
            con.execute("DELETE FROM features WHERE id = ?", (feature_id,))
        else:
            con.execute(
                "UPDATE features SET kind=?, updated_at=? WHERE id=?",
                (kind, now, feature_id),
            )
        return con.execute("SELECT changes()").fetchone()[0] > 0 or kind == "dismissed"


def update_tile_status(tile_id: str, status: str, feature_count: int | None = None) -> None:
    """Update a coverage tile's status (e.g. seeded → analyzed)."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        analyzed_at = now if status == "analyzed" else None
        seeded_at = now if status == "seeded" else None
        if feature_count is not None:
            con.execute(
                """UPDATE coverage_tiles
                   SET status=?, feature_count=?,
                   seeded_at=COALESCE(?, seeded_at),
                   analyzed_at=COALESCE(?, analyzed_at)
                   WHERE tile_id=?""",
                (status, feature_count, seeded_at, analyzed_at, tile_id),
            )
        else:
            con.execute(
                """UPDATE coverage_tiles
                   SET status=?,
                   seeded_at=COALESCE(?, seeded_at),
                   analyzed_at=COALESCE(?, analyzed_at)
                   WHERE tile_id=?""",
                (status, seeded_at, analyzed_at, tile_id),
            )


def get_kalman_state(feature_id: str) -> str | None:
    """Return serialised Kalman state JSON for a feature, or None."""
    with _conn() as con:
        row = con.execute(
            "SELECT kalman_state_json FROM features WHERE id=?", (feature_id,)
        ).fetchone()
    return row["kalman_state_json"] if row else None


def set_kalman_state(feature_id: str, state_json: str) -> None:
    """Persist Kalman state JSON for a feature."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        con.execute(
            "UPDATE features SET kalman_state_json=?, updated_at=? WHERE id=?",
            (state_json, now, feature_id),
        )


def record_position_error(feature_id: str, error_m: float) -> None:
    """EMA-blend a confirmed position error into calibration_error_m."""
    with _conn() as con:
        row = con.execute(
            "SELECT calibration_error_m FROM features WHERE id=?", (feature_id,)
        ).fetchone()
        if row and row["calibration_error_m"] is not None:
            blended = row["calibration_error_m"] + (error_m - row["calibration_error_m"]) * 0.4
        else:
            blended = error_m
        con.execute(
            "UPDATE features SET calibration_error_m=? WHERE id=?",
            (round(blended, 1), feature_id),
        )


def get_calibration_stats() -> dict[str, Any]:
    """Return mean / median position error across confirmed features."""
    with _conn() as con:
        rows = con.execute(
            "SELECT calibration_error_m FROM features WHERE calibration_error_m IS NOT NULL"
        ).fetchall()
    errors = [r["calibration_error_m"] for r in rows]
    if not errors:
        return {"count": 0, "mean_error_m": None, "median_error_m": None}
    errors.sort()
    n = len(errors)
    mean_err = sum(errors) / n
    median_err = errors[n // 2] if n % 2 else (errors[n // 2 - 1] + errors[n // 2]) / 2
    return {
        "count": n,
        "mean_error_m": round(mean_err, 1),
        "median_error_m": round(median_err, 1),
        "min_error_m": round(errors[0], 1),
        "max_error_m": round(errors[-1], 1),
    }


def get_uncertain_features(limit: int = 10) -> list[dict[str, Any]]:
    """Return features most in need of confirmation.

    Ranks by lowest (confidence / analysis_count) ratio among speculative
    features that have ≥2 observations — the highest-observed, lowest-confidence
    ones are most informative to confirm or dismiss.
    """
    with _conn() as con:
        rows = con.execute(
            """SELECT * FROM features
               WHERE kind='speculative' AND analysis_count >= 2
               ORDER BY (confidence / analysis_count) ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    result = []
    for row in rows:
        props = json.loads(row["properties_json"])
        props["id"] = row["id"]
        props["kind"] = row["kind"]
        props["confidence"] = row["confidence"]
        props["analysis_count"] = row["analysis_count"]
        result.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": props,
        })
    return result
