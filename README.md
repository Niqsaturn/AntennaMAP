# AntennaMAP Starter Project

This repository now starts a real application with:
- FastAPI API for map features and AI assessment status
- Local continuous RF assessment loop (aggregated telemetry only)
- Optional Ollama integration for local LLM-assisted emitter estimation
- MapLibre-based 3D map UI that polls for refreshed estimated emitters across a Miami city-wide seed dataset

## Privacy boundary

The loop uses only aggregated telemetry metadata (band, RSSI, SNR, bearing, coordinates). It does not decode communication payloads.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional, in another terminal:
# ollama run llama3.1:8b

uvicorn backend.main:app --reload --port 8000
```

Open: `http://localhost:8000`

## Run tests

```bash
pytest -q
```


## City coverage

The seed dataset now includes synthetic infrastructure and estimate points across the Miami city bounding box for full-city preview rendering.
