# AntennaMAP Starter Project

AntennaMAP now includes a local AI assessment loop designed for **informational-only RF telemetry analysis**.

## What was added

- FastAPI backend endpoints for features, health, and model assessment state.
- Background assessment loop that continuously processes local telemetry summaries.
- Optional local Ollama integration (`http://localhost:11434`) for inference.
- Fallback estimator when Ollama is unavailable.
- Frontend polling every 15 seconds to display AI-estimated emitters on the map.

## Privacy boundary

This starter only processes **aggregated spectrum telemetry** (RSSI/SNR/bearing/band metadata). It does **not** decode or store communication payloads.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional: run ollama separately (example)
# ollama run llama3.1:8b

uvicorn backend.main:app --reload --port 8000
```

Open `http://localhost:8000`.

## API

- `GET /api/health`
- `GET /api/features?kind=infrastructure|estimate&timestamp_lte=<ISO8601>`
- `GET /api/assessment`
