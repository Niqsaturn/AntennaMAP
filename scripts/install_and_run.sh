#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"

cd "$ROOT_DIR"
./scripts/setup_env.sh
./scripts/run_server.sh
