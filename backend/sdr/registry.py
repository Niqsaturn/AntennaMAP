from __future__ import annotations

import logging
from typing import Any

from backend.sdr.base import BaseSdrAdapter

logger = logging.getLogger(__name__)

# Always-available adapters (mock fallback built in)
from backend.sdr.adapters.kiwisdr_adapter import KiwiSdrAdapter
from backend.sdr.adapters.rtlsdr_adapter import RtlSdrAdapter
from backend.sdr.adapters.hackrf_adapter import HackrfAdapter
from backend.sdr.adapters.airspy_adapter import AirspyAdapter

ADAPTER_REGISTRY: dict[str, type[BaseSdrAdapter]] = {
    "kiwisdr": KiwiSdrAdapter,
    "rtlsdr": RtlSdrAdapter,
    "airspy": AirspyAdapter,
    "hackrf": HackrfAdapter,
}


def create_sdr_adapter(provider_type: str, config: dict[str, Any] | None = None) -> BaseSdrAdapter:
    try:
        adapter_cls = ADAPTER_REGISTRY[provider_type.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(ADAPTER_REGISTRY))
        raise ValueError(f"Unknown SDR provider '{provider_type}'. Supported: {supported}") from exc
    adapter = adapter_cls(config=config)
    logger.debug("Created SDR adapter: %s (config=%s)", provider_type, config)
    return adapter


def create_sdr_adapter_from_config(config: dict[str, Any]) -> BaseSdrAdapter:
    provider_type = config.get("type")
    if not provider_type:
        raise ValueError("SDR config must include 'type'")
    provider_config = config.get("settings", {})
    return create_sdr_adapter(provider_type=provider_type, config=provider_config)
