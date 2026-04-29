# AntennaMAP Starter Project

This begins the project with a proper application structure:
- FastAPI backend (`/api/features`, `/api/health`)
- Static frontend using MapLibre for a 3D GPS-style map
- Sample Miramar data for infrastructure + estimated emitters

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Open `http://localhost:8000`.

## Current scope

- Click infrastructure and estimated emitter objects for metadata
- Time cutoff slider for timestamp filtering
- Layer toggles for object categories
- Informational-only dataset with no private payload content
