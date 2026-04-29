# AntennaMAP Starter Project

This repository now starts a real application with:
- FastAPI API for map features and AI assessment status
- Local continuous RF assessment loop (aggregated telemetry only)
- Optional Ollama integration for local LLM-assisted emitter estimation
- MapLibre-based 3D map UI that polls for refreshed estimated emitters across a Miami city-wide seed dataset

## Privacy boundary

The loop uses only aggregated telemetry metadata (band, RSSI, SNR, bearing, coordinates). It does not decode communication payloads.

## One-command install + run (desktop-friendly)

### macOS / Linux terminal

```bash
./scripts/install_and_run.sh
```

### macOS Finder double-click

Double-click:

```text
scripts/install_and_run.command
```

The script will:
1. Create `.venv` if missing
2. Install dependencies
3. Run tests
4. Start the app with uvicorn

Then open: `http://127.0.0.1:8000`

## Manual run (alternative)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional, in another terminal:
# ollama run llama3.1:8b

uvicorn backend.main:app --reload --port 8000
```

## Run tests

```bash
PYTHONPATH=. pytest -q
```

## City coverage

The seed dataset includes synthetic infrastructure and estimate points across the Miami city bounding box for full-city preview rendering.
