#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${1:-$HOME/Desktop/AntennaMAP}"
BRANCH="${2:-main}"

if [[ ! -d "$INSTALL_DIR" ]]; then
  echo "Install directory not found: $INSTALL_DIR"
  exit 1
fi
if [[ ! -d "$INSTALL_DIR/.git" ]]; then
  echo "Directory exists but is not a git repo: $INSTALL_DIR"
  exit 1
fi

cd "$INSTALL_DIR"

echo "[AntennaMAP] Updating repository in: $INSTALL_DIR"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

echo "[AntennaMAP] Refreshing environment + tests"
./scripts/setup_env.sh

echo "[AntennaMAP] Launching server"
./scripts/run_server.sh
