from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.rf.geometry import build_propagation_features

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
    kind: str | None = Query(default=None, pattern="^(infrastructure|estimate)?$"),
    timestamp_lte: str | None = None,
) -> dict:
    source_features = get_features(kind=kind, timestamp_lte=timestamp_lte)["features"]
    propagation_features = []
    for feature in source_features:
        propagation_features.extend(build_propagation_features(feature))
    return {"type": "FeatureCollection", "features": propagation_features}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
