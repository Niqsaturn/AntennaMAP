#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${1:-}"
TARGET_DIR="${2:-AntennaMAP}"

if [[ -z "$REPO_URL" ]]; then
  echo "Usage: ./scripts/install_from_github.sh <github_repo_url> [target_dir]"
  echo "Example: ./scripts/install_from_github.sh https://github.com/your-org/AntennaMAP.git"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is required but not installed."
  exit 1
fi

if [[ -d "$TARGET_DIR" ]]; then
  echo "Error: target directory '$TARGET_DIR' already exists."
  exit 1
fi

echo "[AntennaMAP] Cloning repository..."
git clone "$REPO_URL" "$TARGET_DIR"

cd "$TARGET_DIR"

echo "[AntennaMAP] Running local installer/runner..."
./scripts/install_and_run.sh
