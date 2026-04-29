from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.ingest.storage import append_jsonl


class Geometry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(pattern="^Point$")
    coordinates: list[float] = Field(min_length=2, max_length=2)


class FeatureProperties(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    kind: str = Field(pattern="^(infrastructure|estimate)$")
    name: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)


class InfrastructureFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(pattern="^Feature$")
    geometry: Geometry
    properties: FeatureProperties


@dataclass
class InfrastructureIngestResult:
    accepted: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def ingest_infrastructure(features: list[dict[str, Any]], output_path) -> InfrastructureIngestResult:
    accepted: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for raw in features:
        try:
            parsed = InfrastructureFeature.model_validate(raw)
        except ValidationError as exc:
            errors.append({"type": "schema_validation", "errors": exc.errors(), "row": raw})
            continue

        lon, lat = parsed.geometry.coordinates
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            errors.append({"type": "coordinate_out_of_bounds", "row": parsed.model_dump(mode='json')})
            continue

        accepted.append(parsed.model_dump(mode="json"))

    append_jsonl(output_path, accepted)
    return InfrastructureIngestResult(accepted=accepted, errors=errors)
