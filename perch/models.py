"""Shared data models for explainable attack-position evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WeaponEnvelope:
    """Launch envelope expressed in scene units."""

    name: str
    min_range: float
    max_range: float
    optimal_range: float
    guidance: tuple[str, ...] = ()
    requires_designator: bool = False


@dataclass(frozen=True)
class ObstacleVolume:
    """Conservative vertical-cylinder representation of an Isaac obstacle."""

    obstacle_id: str
    category: str
    center: np.ndarray = field(compare=False)
    radius: float = 0.0
    base_z: float = 0.0
    top_z: float = 0.0
    blocks_los: bool = True


DEFAULT_WEAPON_ENVELOPES_KM: dict[str, dict[str, object]] = {
    "HF": {
        "min_range_km": 2.0,
        "max_range_km": 8.0,
        "optimal_range_km": 4.0,
        "guidance": ("semi_active_laser",),
        "requires_designator": False,
    },
    "RKT": {
        "min_range_km": 1.0,
        "max_range_km": 5.0,
        "optimal_range_km": 3.0,
        "guidance": ("ballistic",),
        "requires_designator": False,
    },
    "GUN": {
        "min_range_km": 0.5,
        "max_range_km": 2.0,
        "optimal_range_km": 1.2,
        "guidance": ("line_of_sight",),
        "requires_designator": False,
    },
}


def resolve_weapon_envelope_km(
    world: dict[str, Any],
    platform_id: str,
    munition: str,
) -> dict[str, Any]:
    """Resolve the same kilometre-based weapon envelope for LLM and FREA."""
    code = str(munition).upper()
    settings: dict[str, Any] = {}
    platform_settings = world.get("platform_weapon_envelopes", {})
    if platform_id in platform_settings:
        settings = dict(platform_settings[platform_id].get(code, {}))
    if not settings:
        settings = dict(world.get("weapon_envelopes", {}).get(code, {}))
    if not settings:
        settings = dict(DEFAULT_WEAPON_ENVELOPES_KM.get(code, {
            "min_range_km": 1.0,
            "max_range_km": 5.0,
            "optimal_range_km": 3.0,
        }))

    min_range = max(0.0, float(settings.get("min_range_km", 0.2)))
    max_range = max(
        min_range + 1e-6,
        float(settings.get("max_range_km", 5.0)),
    )
    optimal = float(settings.get(
        "optimal_range_km", (min_range + max_range) * 0.5
    ))
    settings.update({
        "name": str(settings.get("name", code)),
        "min_range_km": min_range,
        "max_range_km": max_range,
        "optimal_range_km": min(max(optimal, min_range), max_range),
        "guidance": tuple(settings.get("guidance", ())),
        "requires_designator": bool(
            settings.get("requires_designator", False)
        ),
    })
    return settings
