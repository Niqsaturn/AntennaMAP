from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_FILE = ROOT / "backend" / "model_providers" / "python_local_models.json"
MODEL_DIR = ROOT / "backend" / "models"


def discover_python_local_models() -> list[str]:
    models: set[str] = set()

    if REGISTRY_FILE.exists():
        try:
            payload = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                entries = payload.get("models", [])
            elif isinstance(payload, list):
                entries = payload
            else:
                entries = []
            for entry in entries:
                if isinstance(entry, str) and entry.strip():
                    models.add(entry.strip())
                elif isinstance(entry, dict):
                    name = str(entry.get("name", "")).strip()
                    if name:
                        models.add(name)
        except json.JSONDecodeError:
            pass

    if MODEL_DIR.exists():
        for artifact in MODEL_DIR.iterdir():
            if artifact.is_dir():
                models.add(artifact.name)
            elif artifact.suffix.lower() in {".pt", ".pth", ".onnx", ".pkl", ".joblib", ".gguf"}:
                models.add(artifact.stem)

    return sorted(models)
