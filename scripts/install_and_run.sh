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

echo "[AntennaMAP] Starting app at http://${HOST}:${PORT}"
exec uvicorn backend.main:app --host "$HOST" --port "$PORT" --reload
