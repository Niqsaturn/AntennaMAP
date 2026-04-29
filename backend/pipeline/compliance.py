from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
from typing import Any

COMPLIANCE_POLICY = {
    "policy_id": "metadata-only-v1",
    "metadata_only": True,
    "do_not_decode_payload": True,
    "allowed_fields": ["timestamp", "band", "rssi_dbm", "snr_db", "bearing_deg", "lat", "lon", "region"],
    "frequency_allowlist_mhz": {
        "US": [[902, 928], [2400, 2483.5], [5150, 5850]],
        "EU": [[863, 870], [2400, 2483.5], [5150, 5875]],
        "JP": [[920, 928], [2400, 2483.5], [5170, 5835]],
    },
    "retention_days": {"raw_telemetry": 90, "aggregated_metrics": 365},
}


def append_audit_event(path: Path, event_type: str, details: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "event_type": event_type,
        "details": details,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True))
        f.write("\n")


def apply_retention(path: Path, retention_days: int, *, now: datetime | None = None) -> None:
    if not path.exists():
        return
    now = now or datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=retention_days)
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            stamp = datetime.fromisoformat(row.get("timestamp", ""))
            if stamp >= cutoff:
                kept.append(json.dumps(row, sort_keys=True))
        except Exception:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def policy_status() -> dict[str, Any]:
    return {
        "status": "enforced",
        "active_policy": COMPLIANCE_POLICY,
    }
