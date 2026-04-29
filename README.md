# AntennaMAP Starter Project

This repository now starts a real application with:
- FastAPI API for map features and AI assessment status
- Local continuous RF assessment loop (aggregated telemetry only)
- Optional Ollama integration for local LLM-assisted emitter estimation
- MapLibre-based 3D map UI that polls for refreshed estimated emitters across a Miami city-wide seed dataset

## Privacy boundary

The loop uses only aggregated telemetry metadata (band, RSSI, SNR, bearing, coordinates). It does not decode communication payloads.

## One-command install + run (desktop-friendly)

### macOS / Linux terminal

```bash
./scripts/install_and_run.sh
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_and_run.ps1
```

### Windows (Command Prompt)

```bat
scripts\install_and_run.bat
```

### macOS Finder double-click

Double-click:

```text
scripts/install_and_run.command
```

The script will:
1. Create `.venv` if missing
2. Install dependencies
3. Run tests
4. Start the app with uvicorn

Then open: `http://127.0.0.1:8000`

## Install directly from GitHub

### macOS / Linux

```bash
curl -fsSL -o install_from_github.sh https://raw.githubusercontent.com/your-github-username/AntennaMAP/main/scripts/install_from_github.sh
chmod +x install_from_github.sh
./install_from_github.sh https://github.com/your-github-username/AntennaMAP.git
```

### Windows PowerShell

```powershell
Invoke-WebRequest https://raw.githubusercontent.com/your-github-username/AntennaMAP/main/scripts/install_from_github.ps1 -OutFile install_from_github.ps1
powershell -ExecutionPolicy Bypass -File .\install_from_github.ps1 -RepoUrl https://github.com/your-github-username/AntennaMAP.git
```

> Replace `your-github-username` with your GitHub username (repo name stays `AntennaMAP`).

## Manual run (alternative)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional, in another terminal:
# ollama run llama3.1:8b

uvicorn backend.main:app --reload --port 8000
```

## Run tests

```bash
PYTHONPATH=. pytest -q
```

## City coverage

The seed dataset includes synthetic infrastructure and estimate points across the Miami city bounding box for full-city preview rendering.
