"""Domain data models — pure dataclasses, no algorithm logic."""

from .agent import AgentSpec
from .attack_region import AttackRegion
from .position import Position
from .result import AlgorithmResult
from .route import FormationPlan, Route
from .task import ReconTask, StrikeTask, TaskSpec

__all__ = [
    "AgentSpec",
    "AlgorithmResult",
    "AttackRegion",
    "FormationPlan",
    "Position",
    "ReconTask",
    "Route",
    "StrikeTask",
    "TaskSpec",
]
