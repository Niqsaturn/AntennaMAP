"""Few-shot example manager for Ollama prompt augmentation.

Confirmed detections are stored here and injected as examples in future
Ollama analysis calls so the model learns from operator ground truth over time.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_FILE = ROOT / "backend" / "analysis" / "examples.jsonl"


def add_confirmed_example(
    detection: dict[str, Any],
    true_lat: float,
    true_lon: float,
) -> None:
    """Store a confirmed detection as a few-shot example."""
    example = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "freq_band": detection.get("freq_band"),
        "antenna_type": detection.get("antenna_type"),
        "bearing_deg": detection.get("bearing_deg"),
        "confidence": detection.get("confidence"),
        "true_lat": true_lat,
        "true_lon": true_lon,
        "notes": detection.get("notes", ""),
    }
    EXAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with EXAMPLES_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(example) + "\n")


def get_recent_examples(n: int = 3) -> list[dict[str, Any]]:
    """Return the N most recent confirmed examples for prompt injection."""
    if not EXAMPLES_FILE.exists():
        return []
    lines = EXAMPLES_FILE.read_text(encoding="utf-8").strip().splitlines()
    examples = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(examples) >= n:
            break
    return examples


def format_examples_for_prompt(examples: list[dict[str, Any]]) -> str:
    """Format examples as a few-shot section for the Ollama system prompt."""
    if not examples:
        return ""
    lines = ["Prior confirmed detections (use as reference):"]
    for ex in examples:
        lines.append(
            f"  - {ex.get('freq_band','?')} band, bearing {ex.get('bearing_deg','?')}°, "
            f"type={ex.get('antenna_type','?')}, "
            f"confirmed at ({ex.get('true_lat','?'):.4f}, {ex.get('true_lon','?'):.4f})"
        )
    return "\n".join(lines)


def clear_examples() -> None:
    """Remove all stored examples (reset for testing)."""
    if EXAMPLES_FILE.exists():
        EXAMPLES_FILE.unlink()
