from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.rf.antenna_classifier import classify_antenna

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "public" / "data" / "antenna_data.geojson"
TELEMETRY_FILE = ROOT / "public" / "data" / "telemetry_samples.json"

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


def load_telemetry_samples() -> list[dict]:
    with TELEMETRY_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_telemetry(samples: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample.get("band", "unknown")].append(sample)

    summary = {}
    for band, items in grouped.items():
        avg_snr = sum(float(i.get("snr_db", 0)) for i in items) / len(items)
        summary[band] = {"sample_count": len(items), "avg_snr_db": round(avg_snr, 1)}

    return {"band_summary": summary}


def enrich_feature(feature: dict, telemetry: list[dict]) -> dict:
    props = feature["properties"]
    band = props.get("freq_band")
    telemetry_for_feature = [s for s in telemetry if s.get("band") == band] if band else telemetry
    result = classify_antenna(props, telemetry_for_feature)
    props["antenna_type"] = result.antenna_type
    props["type_confidence"] = result.confidence
    props["estimated_elements"] = result.estimated_elements
    return feature


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "antennamap"}


@app.get("/api/features")
def get_features(
    kind: str | None = Query(default=None, pattern="^(infrastructure|estimate)?$"),
    timestamp_lte: str | None = None,
) -> dict:
    data = load_geojson()
    telemetry = load_telemetry_samples()
    features = [enrich_feature(f, telemetry) for f in data["features"]]

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


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
