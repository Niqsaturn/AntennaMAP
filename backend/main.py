from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "public" / "data" / "antenna_data.geojson"
TELEMETRY_FILE = ROOT / "public" / "data" / "telemetry_samples.json"
OLLAMA_URL = "http://localhost:11434/api/generate"

app = FastAPI(title="AntennaMAP API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

state: dict[str, Any] = {
    "latest_assessment": None,
    "estimated_features": [],
    "loop_running": False,
    "loop_error": None,
}


def load_geojson() -> dict:
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_telemetry() -> list[dict[str, Any]]:
    with TELEMETRY_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_telemetry(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        grouped.setdefault(s["band"], []).append(s)

    band_summary = {}
    for band, rows in grouped.items():
        avg_snr = sum(r["snr_db"] for r in rows) / len(rows)
        avg_rssi = sum(r["rssi_dbm"] for r in rows) / len(rows)
        band_summary[band] = {
            "sample_count": len(rows),
            "avg_snr_db": round(avg_snr, 2),
            "avg_rssi_dbm": round(avg_rssi, 2),
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window": "rolling_local_samples",
        "band_summary": band_summary,
    }


def ask_ollama(summary: dict[str, Any]) -> dict[str, Any]:
    """Privacy-safe model interaction: aggregated telemetry only, no decoded payloads."""
    prompt = (
        "You are an RF analysis assistant. Given aggregated spectrum telemetry and bearings, "
        "suggest up to 2 potential estimated emitter locations with confidence. "
        "Return strict JSON with key 'estimates'. Each estimate requires id, lon, lat, "
        "freq_band, confidence_score(0-1), confidence_major_m, confidence_minor_m, sample_count.\n\n"
        f"Telemetry summary: {json.dumps(summary)}"
    )

    payload = {"model": "llama3.1:8b", "prompt": prompt, "stream": False, "format": "json"}
    res = requests.post(OLLAMA_URL, json=payload, timeout=10)
    res.raise_for_status()
    response = res.json().get("response", "{}")
    return json.loads(response)


def fallback_estimates(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "estimates": [
            {
                "id": "est-ai-001",
                "lon": -80.272,
                "lat": 25.967,
                "freq_band": "1.8 GHz",
                "confidence_score": 0.58,
                "confidence_major_m": 450,
                "confidence_minor_m": 190,
                "sample_count": 12,
            }
        ]
    }


def build_estimate_features(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    features = []
    for est in analysis.get("estimates", [])[:5]:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [est["lon"], est["lat"]]},
                "properties": {
                    "id": est["id"],
                    "kind": "estimate",
                    "name": "AI Estimated Emitter",
                    "freq_band": est["freq_band"],
                    "confidence_major_m": est["confidence_major_m"],
                    "confidence_minor_m": est["confidence_minor_m"],
                    "confidence_score": est["confidence_score"],
                    "sample_count": est["sample_count"],
                    "timestamp": now,
                    "source": "local_llm_assessment",
                },
            }
        )
    return features


def analysis_loop() -> None:
    state["loop_running"] = True
    while True:
        try:
            samples = load_telemetry()
            summary = summarize_telemetry(samples)
            try:
                analysis = ask_ollama(summary)
            except Exception:
                analysis = fallback_estimates(summary)
            state["latest_assessment"] = summary
            state["estimated_features"] = build_estimate_features(analysis)
            state["loop_error"] = None
        except Exception as exc:
            state["loop_error"] = str(exc)
        time.sleep(15)


@app.on_event("startup")
def startup_event() -> None:
    threading.Thread(target=analysis_loop, daemon=True).start()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "antennamap",
        "ai_loop_running": state["loop_running"],
        "ai_loop_error": state["loop_error"],
    }


@app.get("/api/assessment")
def assessment() -> dict:
    return {
        "latest_assessment": state["latest_assessment"],
        "estimated_feature_count": len(state["estimated_features"]),
    }


@app.get("/api/features")
def get_features(kind: str | None = Query(default=None, pattern="^(infrastructure|estimate)?$"), timestamp_lte: str | None = None) -> dict:
    data = load_geojson()
    features = data["features"] + state["estimated_features"]

    if kind:
        features = [f for f in features if f["properties"].get("kind") == kind]
    if timestamp_lte:
        cutoff = datetime.fromisoformat(timestamp_lte.replace("Z", "+00:00"))
        features = [f for f in features if datetime.fromisoformat(f["properties"]["timestamp"].replace("Z", "+00:00")) <= cutoff]

    return {"type": "FeatureCollection", "features": features}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
