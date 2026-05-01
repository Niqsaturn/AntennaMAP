"""AI-powered RF signal detection.

Calls the selected Ollama or python_local model to analyze recent telemetry
observations and extract structured signal detections.  Audio content is
never decoded — only carrier metadata is analyzed.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib import error, request

# Audio-bearing frequency bands whose content must not be characterized
_AUDIO_BANDS = {
    "AM Broadcast", "FM Broadcast", "Aviation VHF",
    "VHF Land Mobile", "UHF Land Mobile", "HF / Shortwave",
}

_SYSTEM_PROMPT = """\
You are an expert RF signal analyst and antenna engineer specializing in
geolocation and array detection.

Given SDR telemetry observations (frequency band, bearing, RSSI, SNR, GPS),
you will:
1. Identify distinct signal sources by clustering on freq_band + bearing proximity
2. Estimate each source's position from RSSI/bearing triangulation if possible
3. Classify antenna type: omni | sector_panel | phased_array | dish | unknown
4. Rate confidence 0.0–1.0 for each detection
5. Note evidence of multi-element arrays

CRITICAL PRIVACY RULE: Do NOT characterize, describe, or reference any audio
or voice content.  Analyze carrier signal metadata ONLY.

Respond ONLY with a valid JSON object — no markdown, no prose outside JSON:
{
  "detections": [
    {
      "signal_id": "sig_001",
      "freq_band": "<band>",
      "bearing_deg": <float or null>,
      "confidence": <0.0–1.0>,
      "antenna_type": "<type>",
      "estimated_lat": <float or null>,
      "estimated_lon": <float or null>,
      "azimuth_deg": <float or null>,
      "beamwidth_deg": <float or null>,
      "notes": "<brief carrier-only note>"
    }
  ],
  "summary": "<one sentence carrier analysis summary>"
}"""


@dataclass
class SignalDetection:
    signal_id: str
    freq_band: str
    bearing_deg: float | None
    confidence: float
    antenna_type: str
    estimated_lat: float | None
    estimated_lon: float | None
    azimuth_deg: float | None
    beamwidth_deg: float | None
    notes: str = ""


@dataclass
class DetectionResult:
    detections: list[SignalDetection] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""
    processing_ms: float = 0.0
    error: str | None = None


def _scrub_audio_content(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove any fields that could contain demodulated audio content."""
    safe_keys = {
        "timestamp", "band", "freq_band", "frequency_mhz", "frequency_hz",
        "rssi_dbm", "snr_db", "bearing_deg", "lat", "lon", "region",
        "bandwidth_hz", "sample_rate_hz",
    }
    scrubbed = []
    for s in samples:
        clean = {k: v for k, v in s.items() if k in safe_keys}
        # Flag audio-bearing bands explicitly
        band = clean.get("band") or clean.get("freq_band", "")
        if band in _AUDIO_BANDS:
            clean["content_policy"] = "carrier_metadata_only"
        scrubbed.append(clean)
    return scrubbed


def _format_user_prompt(
    samples: list[dict[str, Any]],
    fcc_context: list[dict[str, Any]],
    examples_text: str,
) -> str:
    scrubbed = _scrub_audio_content(samples)
    parts = []
    if examples_text:
        parts.append(examples_text)
    parts.append(f"SDR telemetry observations ({len(scrubbed)} samples):")
    parts.append(json.dumps(scrubbed[:50], indent=2))  # cap at 50
    if fcc_context:
        parts.append(f"\nNearby FCC-licensed transmitters ({len(fcc_context)} found):")
        parts.append(json.dumps(fcc_context[:20], indent=2))
    return "\n".join(parts)


def _call_ollama(model: str, user_prompt: str, timeout: float = 60.0) -> str:
    body = json.dumps({
        "model": model,
        "system": _SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode("utf-8")
    req = request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("response", "")


def _parse_detections(raw: str) -> tuple[list[SignalDetection], str]:
    # Extract JSON block (handle markdown fences)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return [], ""
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [], ""

    detections = []
    for d in data.get("detections", []):
        try:
            detections.append(SignalDetection(
                signal_id=str(d.get("signal_id", "sig_unknown")),
                freq_band=str(d.get("freq_band", "unknown")),
                bearing_deg=_safe_float(d.get("bearing_deg")),
                confidence=float(max(0.0, min(1.0, d.get("confidence", 0.5)))),
                antenna_type=str(d.get("antenna_type", "unknown")),
                estimated_lat=_safe_float(d.get("estimated_lat")),
                estimated_lon=_safe_float(d.get("estimated_lon")),
                azimuth_deg=_safe_float(d.get("azimuth_deg")),
                beamwidth_deg=_safe_float(d.get("beamwidth_deg")),
                notes=str(d.get("notes", "")),
            ))
        except Exception:
            continue
    return detections, str(data.get("summary", ""))


def _python_local_detections(
    samples: list[dict[str, Any]], operator_lat: float, operator_lon: float
) -> DetectionResult:
    """Fallback to rule-based detection when no LLM is available."""
    from backend.ml.triangulation_pipeline import TelemetrySample, parse_samples, solve_weighted_least_squares
    from backend.datasources.spectrum_allocations import allocations_for_freq

    t0 = time.monotonic()
    # Group samples by band
    bands: dict[str, list[dict]] = {}
    for s in samples:
        band = s.get("band") or s.get("freq_band", "unknown")
        bands.setdefault(band, []).append(s)

    detections = []
    sid = 0
    for band, band_samples in bands.items():
        if len(band_samples) < 2:
            continue
        # Estimate mean bearing
        bearings = [s.get("bearing_deg", 0.0) for s in band_samples if s.get("bearing_deg") is not None]
        avg_bearing = sum(bearings) / len(bearings) if bearings else None
        avg_rssi = sum(s.get("rssi_dbm", -100) for s in band_samples) / len(band_samples)
        avg_snr = sum(s.get("snr_db", 0) for s in band_samples) / len(band_samples)
        confidence = min(1.0, max(0.1, (avg_snr + 20) / 80 * len(band_samples) / 5))
        sid += 1
        detections.append(SignalDetection(
            signal_id=f"sig_{sid:03d}",
            freq_band=band,
            bearing_deg=round(avg_bearing, 1) if avg_bearing else None,
            confidence=round(confidence, 3),
            antenna_type="unknown",
            estimated_lat=None,
            estimated_lon=None,
            azimuth_deg=round(avg_bearing, 1) if avg_bearing else None,
            beamwidth_deg=None,
            notes=f"rule-based: {len(band_samples)} obs, avg_snr={avg_snr:.1f}dB",
        ))

    ms = (time.monotonic() - t0) * 1000
    return DetectionResult(
        detections=detections,
        summary=f"Rule-based analysis: {len(detections)} signal groups identified",
        raw_response="",
        processing_ms=round(ms, 1),
    )


def detect_signals(
    samples: list[dict[str, Any]],
    model_config: dict[str, Any],
    fcc_context: list[dict[str, Any]] | None = None,
    examples_text: str = "",
    operator_lat: float = 0.0,
    operator_lon: float = 0.0,
) -> DetectionResult:
    """Analyze telemetry samples and return structured signal detections.

    Uses Ollama when provider='ollama', rule-based fallback otherwise.
    Audio content is scrubbed before any model call.
    """
    provider = model_config.get("provider", "local")
    model = model_config.get("model", "")

    if provider == "ollama" and model:
        t0 = time.monotonic()
        user_prompt = _format_user_prompt(samples, fcc_context or [], examples_text)
        try:
            raw = _call_ollama(model, user_prompt)
            detections, summary = _parse_detections(raw)
            ms = (time.monotonic() - t0) * 1000
            return DetectionResult(
                detections=detections,
                summary=summary,
                raw_response=raw,
                processing_ms=round(ms, 1),
            )
        except (error.URLError, TimeoutError, OSError) as exc:
            return DetectionResult(error=f"Ollama unavailable: {exc}")

    # python_local or local fallback
    return _python_local_detections(samples, operator_lat, operator_lon)


def _safe_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
