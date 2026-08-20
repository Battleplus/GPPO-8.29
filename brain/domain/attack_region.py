"""Attack-region data model used between LLM/RAG and Perch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttackRegion:
    """A candidate polygon where Perch should search attack positions."""

    target_id: str
    score: float
    reasoning: str
    polygon_km: list[list[float]]
    polygon_scene: list[list[float]]
    source: str = "demo"
    rank: int = 0
    raw_feature: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "score": float(self.score),
            "reasoning": self.reasoning,
            "polygon_km": [list(point) for point in self.polygon_km],
            "polygon_scene": [list(point) for point in self.polygon_scene],
            "source": self.source,
            "rank": int(self.rank),
            "raw_feature": dict(self.raw_feature),
        }
