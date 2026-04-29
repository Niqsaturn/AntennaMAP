from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.training.dataset_builder import build_training_rows
from backend.training.triangulation_baseline import estimate_single_operator


def train_single_triangulation_baseline(samples: list[dict[str, Any]], models_dir: Path, provider: str = "local") -> dict[str, Any]:
    datasets = build_training_rows(samples)
    outputs = [estimate_single_operator(row["window"]) for row in datasets]
    avg_conf = sum(o["confidence"] for o in outputs) / len(outputs) if outputs else 0.0

    artifact = {
        "provider": provider,
        "model_name": "deterministic_centroid",
        "method": "single_triangulation_baseline",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "num_windows": len(datasets),
            "avg_confidence": round(avg_conf, 4),
        },
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    filename = f"single_triangulation_baseline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    (models_dir / filename).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact
