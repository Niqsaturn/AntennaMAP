#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

cd "$ROOT_DIR"

echo "[AntennaMAP] Project root: $ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[AntennaMAP] Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[AntennaMAP] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[AntennaMAP] Running tests..."
PYTHONPATH=. pytest -q

if [[ "${RUN_BASELINE_TRAINING:-0}" == "1" ]]; then
  echo "[AntennaMAP] Running initial baseline training..."
  python -m curl >/dev/null 2>&1 || true
  python - <<'PY2'
from fastapi.testclient import TestClient
from backend.main import app
client = TestClient(app)
client.post("/api/training/start", params={"method":"single_triangulation_baseline"})
print("baseline training complete")
PY2
fi

echo "[AntennaMAP] Starting app at http://${HOST}:${PORT}"
exec uvicorn backend.main:app --host "$HOST" --port "$PORT" --reload
