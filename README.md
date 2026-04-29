# AntennaMAP Starter Project

This begins the project with a proper application structure:
- FastAPI backend (`/api/features`, `/api/health`, `/api/model/*`)
- Static frontend using MapLibre for a 3D GPS-style map
- Sample Miramar data for infrastructure + estimated emitters
- Triangulation ML pipeline module with deterministic solver + model calibration artifacts

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Open `http://localhost:8000`.

## ML / Triangulation Pipeline

- Module: `backend/ml/triangulation_pipeline.py`
- Seed telemetry example: `public/data/telemetry_samples.json` (example-only)
- Dedicated training data directory: `backend/ml/training_data/`
- Model artifacts/versioned metadata: `backend/ml/models/triangulation_<model_version>.json`
- Endpoints:
  - `POST /api/model/train`
  - `GET /api/model/status`
  - `POST /api/model/infer`

### Training data schema

See `backend/ml/training_data/README.md` for full schema details.

### Current scope

- Click infrastructure and estimated emitter objects for metadata
- Time cutoff slider for timestamp filtering
- Layer toggles for object categories
- Informational-only dataset with no private payload content
