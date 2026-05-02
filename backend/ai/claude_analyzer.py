"""Continuous Claude AI analysis loop for RF signal intelligence.

Uses the Anthropic Claude API (claude-sonnet-4-6) with prompt caching on
the static system context. Runs every 30 seconds and writes signal estimates
back to map_store as speculative features.

Falls back to a rule-based classifier when ANTHROPIC_API_KEY is absent.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"
_CYCLE_S = 30.0

_SYSTEM_PROMPT = """\
You are an expert RF signal intelligence analyst. Analyze radio spectrum \
observations and produce structured signal estimates for a geolocation map.

You will receive spectral peaks (frequency, bandwidth, SNR, modulation hint, \
node location), RSSI observations across distributed nodes, existing map \
features, FCC license data, and satellite positions.

CRITICAL PRIVACY RULE: Analyze carrier signal METADATA ONLY. Do NOT \
characterize, describe, or reference any audio, voice, or content.

Scientific method: cite which specific observations support each estimate.

Respond ONLY with a valid JSON object — no markdown fences, no prose outside JSON:
{
  "signals": [
    {
      "freq_hz": <float>,
      "bandwidth_hz": <float>,
      "modulation": "<CW|SSB|AM|FM|WBFM|data|unknown>",
      "bearing_deg": <float or null>,
      "range_km": <float or null>,
      "lat": <float or null>,
      "lon": <float or null>,
      "confidence": <0.0-1.0>,
      "array_type": "<vertical_monopole|dipole|yagi|phased_array|dish|unknown>",
      "band": "<band name>",
      "notes": "<brief carrier-metadata note, max 120 chars>"
    }
  ],
  "corrections": [
    {
      "feature_id": "<id>",
      "new_lat": <float>,
      "new_lon": <float>,
      "delta_m": <float>,
      "reason": "<brief reason>"
    }
  ],
  "em_fields": [
    {
      "site_id": "<feature id>",
      "azimuth_deg": <float>,
      "beamwidth_deg": <float>,
      "pattern": "<omni|directional|sectored>"
    }
  ]
}"""


@dataclass
class AnalysisCycleResult:
    signals: list[dict] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)
    em_fields: list[dict] = field(default_factory=list)
    raw_response: str = ""
    error: str | None = None
    processing_ms: float = 0.0
    provider: str = "claude"


class ClaudeAnalyzer:
    """Async analysis engine wrapping the Anthropic Messages API."""

    def __init__(self) -> None:
        self._client: Any = None
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._last_features_hash = ""

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            return None
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
            return self._client
        except ImportError:
            logger.warning("anthropic SDK not installed — using rule-based fallback")
            return None

    async def analyze_cycle(
        self,
        peaks: list[dict],
        rssi_obs: list[dict],
        features: list[dict],
        fcc_context: list[dict] | None = None,
        sat_positions: list[dict] | None = None,
        op_lat: float = 0.0,
        op_lon: float = 0.0,
    ) -> AnalysisCycleResult:
        t0 = time.monotonic()
        client = self._get_client()
        if client is None:
            result = _rule_based_fallback(peaks, rssi_obs, op_lat, op_lon)
            result.processing_ms = round((time.monotonic() - t0) * 1000, 1)
            return result

        user_content = _build_user_message(
            peaks, rssi_obs, features,
            fcc_context or [], sat_positions or [],
            op_lat, op_lon,
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.messages.create(
                    model=_MODEL,
                    max_tokens=2048,
                    system=[
                        {
                            "type": "text",
                            "text": _SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_content}],
                ),
            )
            raw = response.content[0].text if response.content else "{}"
            parsed = _parse_response(raw)
            parsed.processing_ms = round((time.monotonic() - t0) * 1000, 1)
            parsed.provider = "claude"
            return parsed
        except Exception as exc:
            logger.warning("claude_analyzer: API error: %s", exc)
            result = _rule_based_fallback(peaks, rssi_obs, op_lat, op_lon)
            result.error = str(exc)
            result.processing_ms = round((time.monotonic() - t0) * 1000, 1)
            return result

    async def enrich_estimate(
        self,
        estimate_lat: float,
        estimate_lon: float,
        uncertainty_m: float,
        peaks: list[dict],
        fcc_context: list[dict],
        freq_hz: float,
    ) -> dict:
        """One-shot enrichment called from fox hunt solver.

        Returns corrected lat/lon or the original estimate if Claude is unavailable.
        """
        client = self._get_client()
        if client is None:
            return {"lat": estimate_lat, "lon": estimate_lon, "notes": "rule-based"}

        msg = (
            f"Fox hunt WLS estimate: lat={estimate_lat:.5f}, lon={estimate_lon:.5f}, "
            f"uncertainty={uncertainty_m:.0f}m, freq={freq_hz/1e6:.3f}MHz.\n"
            f"Recent peaks: {json.dumps(peaks[:5])}\n"
            f"FCC licenses nearby: {json.dumps(fcc_context[:5])}\n"
            "Does the WLS estimate appear correct? If not, provide a refined lat/lon "
            "with brief reasoning. Respond with JSON only: "
            '{"lat": <float>, "lon": <float>, "notes": "<reason>"}'
        )
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.messages.create(
                    model=_MODEL,
                    max_tokens=256,
                    system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": msg}],
                ),
            )
            raw = response.content[0].text.strip() if response.content else "{}"
            return json.loads(raw)
        except Exception as exc:
            logger.warning("claude_analyzer: enrich_estimate error: %s", exc)
            return {"lat": estimate_lat, "lon": estimate_lon, "notes": str(exc)}


def _build_user_message(
    peaks: list[dict],
    rssi_obs: list[dict],
    features: list[dict],
    fcc_context: list[dict],
    sat_positions: list[dict],
    op_lat: float,
    op_lon: float,
) -> str:
    lines = [
        f"Operator position: lat={op_lat:.4f}, lon={op_lon:.4f}",
        f"\nSpectral peaks ({len(peaks)} most recent, max 20):",
        json.dumps(peaks[:20], separators=(",", ":")),
        f"\nMulti-node RSSI observations ({len(rssi_obs)}, max 10):",
        json.dumps(rssi_obs[:10], separators=(",", ":")),
        f"\nExisting map features near scan area ({len(features)}, max 10):",
        json.dumps(
            [{"id": f.get("properties", {}).get("id"), "kind": f.get("properties", {}).get("kind"),
              "freq_hz": f.get("properties", {}).get("freq_hz"),
              "confidence": f.get("properties", {}).get("confidence"),
              "lat": f.get("geometry", {}).get("coordinates", [None, None])[1],
              "lon": f.get("geometry", {}).get("coordinates", [None, None])[0]}
             for f in features[:10]],
            separators=(",", ":"),
        ),
        f"\nFCC licenses in area ({len(fcc_context)}, max 10):",
        json.dumps(fcc_context[:10], separators=(",", ":")),
        f"\nSatellite positions overhead ({len(sat_positions)}, max 5):",
        json.dumps(sat_positions[:5], separators=(",", ":")),
    ]
    return "\n".join(lines)


def _parse_response(raw: str) -> AnalysisCycleResult:
    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        return AnalysisCycleResult(
            signals=data.get("signals", []),
            corrections=data.get("corrections", []),
            em_fields=data.get("em_fields", []),
            raw_response=raw[:2000],
        )
    except Exception as exc:
        logger.warning("claude_analyzer: response parse error: %s | raw: %.200s", exc, raw)
        return AnalysisCycleResult(raw_response=raw[:2000], error=str(exc))


def _rule_based_fallback(
    peaks: list[dict],
    rssi_obs: list[dict],
    op_lat: float,
    op_lon: float,
) -> AnalysisCycleResult:
    """Minimal rule-based classifier used when Claude API is unavailable."""
    signals = []
    for p in peaks[:10]:
        freq = float(p.get("freq_hz", 0))
        snr = float(p.get("snr_db", 0))
        bw = float(p.get("bandwidth_hz", 3000))
        confidence = min(0.5, snr / 30.0)
        if confidence < 0.05:
            continue
        signals.append({
            "freq_hz": freq,
            "bandwidth_hz": bw,
            "modulation": p.get("modulation_hint", "unknown"),
            "bearing_deg": None,
            "range_km": None,
            "lat": op_lat if op_lat else None,
            "lon": op_lon if op_lon else None,
            "confidence": round(confidence, 3),
            "array_type": "unknown",
            "band": _freq_to_band(freq),
            "notes": f"Rule-based: SNR={snr:.1f}dB",
        })
    return AnalysisCycleResult(signals=signals, provider="rule_based")


def _freq_to_band(hz: float) -> str:
    mhz = hz / 1e6
    if mhz < 0.03:   return "ELF"
    if mhz < 3:      return "LF/MF"
    if mhz < 30:     return "HF"
    if mhz < 300:    return "VHF"
    if mhz < 3000:   return "UHF"
    return "SHF"


def _signal_to_feature(sig: dict, cycle_ts: str) -> dict:
    """Convert a Claude signal dict to a GeoJSON feature for map_store."""
    freq = float(sig.get("freq_hz", 0))
    lat = sig.get("lat")
    lon = sig.get("lon")
    if lat is None or lon is None:
        return {}
    fid = "claude_" + hashlib.md5(
        f"{lat:.4f}_{lon:.4f}_{freq:.0f}".encode()
    ).hexdigest()[:10]
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "properties": {
            "id": fid,
            "kind": "speculative",
            "name": f"AI: {freq/1e6:.3f} MHz ({sig.get('band', '')})",
            "freq_hz": freq,
            "freq_mhz": round(freq / 1e6, 4),
            "bandwidth_hz": sig.get("bandwidth_hz"),
            "modulation_hint": sig.get("modulation"),
            "bearing_deg": sig.get("bearing_deg"),
            "range_km": sig.get("range_km"),
            "confidence": sig.get("confidence", 0.0),
            "antenna_type": sig.get("array_type", "unknown"),
            "beamwidth_deg": 360,
            "notes": sig.get("notes", ""),
            "source": "claude_analyzer",
            "timestamp": cycle_ts,
        },
    }


# ── Module-level singleton ────────────────────────────────────────────────────

_analyzer = ClaudeAnalyzer()


async def claude_analyzer_loop() -> None:
    """Background loop: analyze spectrum every _CYCLE_S seconds, upsert features."""
    logger.info("claude_analyzer_loop: starting (cycle=%.0fs)", _CYCLE_S)
    while True:
        try:
            await _run_one_cycle()
        except Exception as exc:
            logger.warning("claude_analyzer_loop: cycle error: %s", exc)
        await asyncio.sleep(_CYCLE_S)


async def _run_one_cycle() -> None:
    from backend.foxhunt.auto_loop import event_bus
    from backend.storage.map_store import upsert_feature, get_all_features
    from backend.sdr.kiwisdr_client import node_pool

    # Gather inputs
    try:
        peaks = node_pool.scan_peaks_all(min_snr_db=6.0)
    except Exception:
        peaks = []

    try:
        features_fc = get_all_features(limit=20)
        features = features_fc.get("features", [])
    except Exception:
        features = []

    # Operator position from most recent fox hunt state
    op_lat, op_lon = 0.0, 0.0
    try:
        from backend.foxhunt.auto_loop import auto_loop
        st = auto_loop.status()
        op_lat = st.get("op_lat", 0.0) or 0.0
        op_lon = st.get("op_lon", 0.0) or 0.0
    except Exception:
        pass

    # FCC context
    fcc_context: list[dict] = []
    if op_lat and op_lon:
        try:
            import asyncio as _aio
            fcc_context = await _aio.wait_for(
                _aio.get_event_loop().run_in_executor(
                    None,
                    lambda: [],  # placeholder — real call below
                ),
                timeout=5.0,
            )
        except Exception:
            pass
        try:
            from backend.datasources.fcc import search_licenses_near
            fcc_context = await asyncio.wait_for(
                search_licenses_near(op_lat, op_lon, radius_km=25.0, limit=10),
                timeout=8.0,
            )
        except Exception:
            fcc_context = []

    result = await _analyzer.analyze_cycle(
        peaks=peaks,
        rssi_obs=[],
        features=features,
        fcc_context=fcc_context,
        op_lat=op_lat,
        op_lon=op_lon,
    )

    ts = datetime.now(timezone.utc).isoformat()
    saved = 0
    for sig in result.signals:
        feat = _signal_to_feature(sig, ts)
        if feat:
            try:
                upsert_feature(feat)
                saved += 1
            except Exception as exc:
                logger.debug("claude_analyzer: upsert failed: %s", exc)

    # Apply corrections to existing features
    for corr in result.corrections:
        fid = corr.get("feature_id")
        new_lat = corr.get("new_lat")
        new_lon = corr.get("new_lon")
        if not (fid and new_lat and new_lon):
            continue
        try:
            feat = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(new_lon), float(new_lat)]},
                "properties": {
                    "id": fid,
                    "correction_reason": corr.get("reason", ""),
                    "corrected_at": ts,
                },
            }
            upsert_feature(feat)
        except Exception:
            pass

    event_bus.publish({
        "type": "claude_cycle",
        "ts": ts,
        "signals_found": len(result.signals),
        "saved": saved,
        "corrections": len(result.corrections),
        "provider": result.provider,
        "processing_ms": result.processing_ms,
        "error": result.error,
    })

    if saved or result.corrections:
        logger.info(
            "claude_analyzer: cycle done — %d signals saved, %d corrections, %.0fms",
            saved, len(result.corrections), result.processing_ms,
        )
