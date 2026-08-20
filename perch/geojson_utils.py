"""Parse and validate LLM-produced attack-region GeoJSON."""

from __future__ import annotations

import json
import math
import re
from typing import Any


def parse_llm_geojson(llm_output: str) -> dict[str, Any] | None:
    """Extract a FeatureCollection from plain or Markdown-wrapped JSON."""
    match = re.search(
        r"```(?:json|geojson)?\s*\n?(.*?)\n?```",
        str(llm_output),
        re.DOTALL | re.IGNORECASE,
    )
    candidate = match.group(1) if match else str(llm_output)
    data = _decode_json_object(candidate)
    if not isinstance(data, dict):
        return None
    if data.get("type") == "Feature":
        data = {"type": "FeatureCollection", "features": [data]}
    if data.get("type") != "FeatureCollection":
        return None
    return data


def validate_geojson(data: dict[str, Any], max_features: int = 3) -> list[str]:
    """Return contract violations for attack-region FeatureCollections."""
    errors: list[str] = []
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        return ["Root must be a GeoJSON FeatureCollection"]

    features = data.get("features")
    if not isinstance(features, list) or not features:
        return ["FeatureCollection.features must not be empty"]
    if len(features) > max_features:
        errors.append(f"FeatureCollection may contain at most {max_features} regions")

    for index, feature in enumerate(features):
        prefix = f"features[{index}]"
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            errors.append(f"{prefix}.type must be Feature")
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("type") != "Polygon":
            errors.append(f"{prefix}.geometry.type must be Polygon")
            continue
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list) or not coordinates:
            errors.append(f"{prefix}.geometry.coordinates must contain a ring")
            continue
        ring = coordinates[0]
        ring_errors = _validate_ring(ring)
        errors.extend(f"{prefix}: {error}" for error in ring_errors)

        properties = feature.get("properties")
        if not isinstance(properties, dict):
            errors.append(f"{prefix}.properties must be an object")
            continue
        score = properties.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            errors.append(f"{prefix}.properties.score must be numeric")
        elif not 0.0 <= float(score) <= 1.0:
            errors.append(f"{prefix}.properties.score must be in [0, 1]")
        if not str(properties.get("reasoning", "")).strip():
            errors.append(f"{prefix}.properties.reasoning must not be empty")
    return errors


def _decode_json_object(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _validate_ring(ring: Any) -> list[str]:
    if not isinstance(ring, list) or len(ring) < 4:
        return ["polygon ring must contain at least four points"]
    points: list[tuple[float, float]] = []
    errors: list[str] = []
    for index, point in enumerate(ring):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            errors.append(f"ring[{index}] must be [longitude, latitude]")
            continue
        lon, lat = point[0], point[1]
        if (
            not isinstance(lon, (int, float))
            or isinstance(lon, bool)
            or not isinstance(lat, (int, float))
            or isinstance(lat, bool)
            or not math.isfinite(float(lon))
            or not math.isfinite(float(lat))
        ):
            errors.append(f"ring[{index}] coordinates must be finite numbers")
            continue
        lon_f, lat_f = float(lon), float(lat)
        if not (-180.0 <= lon_f <= 180.0 and -90.0 <= lat_f <= 90.0):
            errors.append(f"ring[{index}] is outside geographic bounds")
        points.append((lon_f, lat_f))
    if errors:
        return errors
    if points[0] != points[-1]:
        errors.append("polygon ring must be closed")
    area_twice = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:])
    )
    if abs(area_twice) <= 1e-12:
        errors.append("polygon ring must have non-zero area")
    return errors
