# AntennaMAP Starter Project

This begins the project with a proper application structure:
- FastAPI backend (`/api/features`, `/api/health`, `/api/pipeline/ingest`, `/api/model/metrics`)
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

## Ingestion + governance layer

- `backend/pipeline/ingest.py` provides append-only telemetry ingestion with strict schema validation.
- Data quality controls enforce timestamp monotonicity, GPS outlier rejection via speed thresholding, and impossible bearing/signal range filtering.
- Each ingestion run stores governance metadata (`run_id`, `model_version`, `data_window`, and metrics) in `backend/pipeline/data/model_runs.jsonl`.
- `/api/model/metrics` exposes the latest and historical model-run metrics for drift tracking.
- Retraining triggers combine:
  - sample-volume threshold,
  - drift-error threshold,
  - scheduled retrain window.

## Privacy and retention policy

- **No payload decoding**: telemetry payload contents are never decoded or persisted.
- **No personal identifiers**: the pipeline excludes direct personal identifiers (e.g., names, emails, account IDs, device owner IDs, phone numbers).
- **Collected fields**: only operational radio telemetry fields needed for map/inference quality are accepted (time, band, signal, bearing, and coarse location).
- **Append-only logs**: accepted telemetry and run metadata are stored as append-only JSONL to preserve lineage and post-hoc auditability.
- **Retention baseline**: keep raw append-only logs for 90 days by default, and retain aggregate drift/retraining metrics longer for model governance.

## Current scope

- Click infrastructure and estimated emitter objects for metadata
- Time cutoff slider for timestamp filtering
- Layer toggles for object categories
- Informational-only dataset with no private payload content

## Map tile network requirements and offline fallback

The frontend map relies on external style/tile endpoints by default:
- Primary style: `https://demotiles.maplibre.org/style.json`
- Backup style: `https://tiles.openfreemap.org/styles/bright`
- MapLibre assets loaded from `https://unpkg.com`

For full rendering, clients must be able to reach those domains over HTTPS (TCP 443). If access is blocked, the app now shows an in-UI warning panel and switches to the configured backup style.

### Local/offline fallback plan

1. Vendor MapLibre JS/CSS into `frontend/vendor/` and update `index.html` script/link tags to local files.
2. Host a local style JSON plus vector/raster tiles (for example from MBTiles via a local tile server such as `tileserver-gl` or equivalent).
3. Update `frontend/main.js` `MAP_STYLES.primary` to the local style URL (example: `http://localhost:8080/styles/basic/style.json`) and keep an internal backup URL.
4. Optionally pre-cache the style, sprites, glyphs, and tile responses with a service worker for disconnected operation.
