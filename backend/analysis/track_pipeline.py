from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Any

from backend.analysis.waterfall_analyzer import analyze_psd, freq_hopping_detected


@dataclass
class TrackCandidate:
    track_id: str
    timestamp: str
    center_frequency_hz: float
    detected_carriers: int
    occupied_bandwidth_hz: float
    drift_hz_per_s: float
    periodicity_s: float | None
    modulation_hints: list[str]
    interference_signature: str
    confidence: float
    temporal_history: list[dict[str, Any]]


def _dominant_modulation(peaks) -> list[str]:
    hints = [p.modulation_hint for p in peaks]
    if not hints:
        return ["unknown"]
    ranked = [name for name, _ in Counter(hints).most_common(3)]
    return ranked


def _estimate_periodicity_s(samples: list[dict[str, Any]]) -> float | None:
    timestamps = []
    for s in samples:
        ts = s.get("timestamp")
        if not ts:
            continue
        timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    if len(timestamps) < 4:
        return None
    timestamps.sort()
    deltas = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    if not deltas or max(deltas) <= 0:
        return None
    return round(mean(deltas), 3)


def _interference_signature(peaks, psd_bins_db: list[float], sample_rate_hz: float) -> str:
    if not peaks:
        return "none"
    if freq_hopping_detected(peaks, sample_rate_hz):
        return "freq_hopping_or_barrage"
    bw_values = [p.bandwidth_3db_hz for p in peaks]
    if bw_values and max(bw_values) > sample_rate_hz * 0.4:
        return "wideband_noise_jamming"
    if len(peaks) >= 6:
        return "multi_tone_or_comb"
    if pstdev(psd_bins_db) < 2.0:
        return "raised_noise_floor"
    return "none"


def derive_track_candidates(samples: list[dict[str, Any]]) -> list[TrackCandidate]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in samples:
        device_id = str(s.get("device_id", "unknown"))
        freq = int((s.get("rf", {}) or {}).get("frequency_hz", 0) // 1000)
        key = (device_id, str(freq))
        grouped.setdefault(key, []).append(s)

    out: list[TrackCandidate] = []
    for (device_id, freq_key), rows in grouped.items():
        rows = sorted(rows, key=lambda r: r.get("timestamp", ""))
        if not rows:
            continue
        latest = rows[-1]
        rf = latest.get("rf", {}) or {}
        spectral = latest.get("spectral", {}) or {}
        center_freq_hz = float(rf.get("frequency_hz", 0.0))
        sample_rate_hz = max(float(rf.get("bandwidth_hz", 1.0)), 1.0)
        psd = spectral.get("psd_bins_db", []) or []
        peaks = analyze_psd(psd, center_freq_hz=center_freq_hz, sample_rate_hz=sample_rate_hz) if psd else []

        occupied_bw = max((p.bandwidth_3db_hz for p in peaks), default=float(rf.get("bandwidth_hz", 0.0)))
        drift = 0.0
        if len(rows) >= 2:
            prev_rf = rows[-2].get("rf", {}) or {}
            prev_f = float(prev_rf.get("frequency_hz", center_freq_hz))
            t2 = datetime.fromisoformat(rows[-1]["timestamp"].replace("Z", "+00:00")).timestamp()
            t1 = datetime.fromisoformat(rows[-2]["timestamp"].replace("Z", "+00:00")).timestamp()
            dt = max(t2 - t1, 1e-6)
            drift = (center_freq_hz - prev_f) / dt

        periodicity = _estimate_periodicity_s(rows[-20:])
        hints = _dominant_modulation(peaks)
        signature = _interference_signature(peaks, psd, sample_rate_hz) if psd else "none"

        snr = float(rf.get("snr_db", 0.0))
        carriers = len(peaks)
        conf = min(0.99, max(0.05, 0.35 + min(0.4, snr / 50.0) + min(0.2, carriers * 0.03)))

        history = []
        for r in rows[-15:]:
            rf_r = r.get("rf", {}) or {}
            loc = r.get("location") or {}
            history.append({
                "timestamp": r.get("timestamp"),
                "frequency_hz": rf_r.get("frequency_hz"),
                "rssi_dbm": rf_r.get("rssi_dbm"),
                "snr_db": rf_r.get("snr_db"),
                "lat": loc.get("lat"),
                "lon": loc.get("lon"),
            })

        out.append(TrackCandidate(
            track_id=f"track_{device_id}_{freq_key}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            center_frequency_hz=center_freq_hz,
            detected_carriers=carriers,
            occupied_bandwidth_hz=round(float(occupied_bw), 2),
            drift_hz_per_s=round(drift, 5),
            periodicity_s=periodicity,
            modulation_hints=hints,
            interference_signature=signature,
            confidence=round(conf, 4),
            temporal_history=history,
        ))
    return out
