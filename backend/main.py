from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.rf_propagation import generate_snapshot

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "public" / "data" / "antenna_data.geojson"

app = FastAPI(title="AntennaMAP API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_geojson() -> dict:
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_telemetry(samples: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        grouped.setdefault(sample.get("band", "unknown"), []).append(sample)
    band_summary = {}
    for band, items in grouped.items():
        snrs = [x.get("snr_db", 0.0) for x in items]
        rssis = [x.get("rssi_dbm", 0.0) for x in items]
        band_summary[band] = {
            "sample_count": len(items),
            "avg_snr_db": round(sum(snrs) / len(snrs), 2),
            "avg_rssi_dbm": round(sum(rssis) / len(rssis), 2),
        }
    return {"band_summary": band_summary}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "antennamap"}


@app.get("/api/features")
def get_features(
    kind: str | None = Query(default=None, pattern="^(infrastructure|estimate)?$"),
    timestamp_lte: str | None = None,
) -> dict:
    data = load_geojson()
    features = data["features"]

    if kind:
        features = [f for f in features if f["properties"].get("kind") == kind]

    if timestamp_lte:
        cutoff = datetime.fromisoformat(timestamp_lte.replace("Z", "+00:00"))
        features = [
            f
            for f in features
            if datetime.fromisoformat(f["properties"]["timestamp"].replace("Z", "+00:00")) <= cutoff
        ]

    return {"type": "FeatureCollection", "features": features}


@app.get("/api/propagation")
def get_propagation(
    site_id: str,
    model: str = Query(default="fspl", pattern="^(fspl|hata_urban|hata_suburban)$"),
    grid_radius_km: float = 5.0,
    grid_resolution: int = 41,
) -> dict:
    features = load_geojson()["features"]
    site = next((f for f in features if f["properties"].get("id") == site_id), None)
    if not site:
        raise HTTPException(status_code=404, detail="site not found")
    props = site["properties"]
    lon, lat = site["geometry"]["coordinates"]

    snapshot = generate_snapshot(
        lat=lat,
        lon=lon,
        eirp_dbm=float(props.get("eirp_dbm", 57.0)),
        frequency_mhz=float(props.get("rf_max_mhz", props.get("rf_min_mhz", 1900))),
        gain_dbi=float(props.get("gain_dbi", 16.0)),
        height_m=float(props.get("height_m", 30.0)),
        beamwidth_deg=float(props.get("beamwidth_deg", 120.0)),
        tilt_deg=float(props.get("tilt_deg", 3.0)),
        orientation_deg=float(props.get("azimuth_deg", 0.0) or 0.0),
        model=model,
        grid_radius_km=grid_radius_km,
        grid_resolution=grid_resolution,
    )
    return {"site_id": site_id, "cache_key": f"{site_id}:{model}:{grid_radius_km}:{grid_resolution}", "snapshot": snapshot}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
