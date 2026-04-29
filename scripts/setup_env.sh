#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

cd "$ROOT_DIR"

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
