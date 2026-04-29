#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${1:-}"
TARGET_DIR="${2:-AntennaMAP}"
BRANCH="${3:-main}"

if [[ -z "$REPO_URL" ]]; then
  echo "Usage: ./scripts/install_from_github.sh <github_repo_url> [target_dir] [branch]"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is required but not installed."
  exit 1
fi

if [[ -d "$TARGET_DIR/.git" ]]; then
  echo "[AntennaMAP] Existing clone found. Updating..."
  git -C "$TARGET_DIR" fetch origin "$BRANCH"
  git -C "$TARGET_DIR" checkout "$BRANCH"
  git -C "$TARGET_DIR" pull --ff-only origin "$BRANCH"
else
  if [[ -d "$TARGET_DIR" ]]; then
    echo "Error: '$TARGET_DIR' exists but is not a git clone."
    exit 1
  fi
  echo "[AntennaMAP] Cloning repository..."
  git clone --branch "$BRANCH" "$REPO_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"
./scripts/setup_env.sh
./scripts/run_server.sh
