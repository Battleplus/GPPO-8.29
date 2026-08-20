"""Position / battle-position data models.

Produced by ``perch`` and consumed by the MPPI action-route
planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Position:
    """A selected battle position (attack / recon / support point).

    Attributes:
        pos_id:   Unique position identifier.
        x, y, z:  Scene-unit coordinates.
        kind:     ``"attack"``, ``"recon"``, or ``"support"``.
        metadata: Optional per-position diagnostics (mask score, range
                  score, constraint violations, Pareto rank, …).
    """

    pos_id: str
    x: float
    y: float
    z: float = 0.0
    kind: str = "attack"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)
