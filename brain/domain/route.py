"""Route and plan data models produced by MPPI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Route:
    """One platform's independent path to its assigned attack position."""

    platform: str
    waypoints: list[list[float]]
    role: str = ""
    target_id: str = ""
    position_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FormationPlan:
    """Route-plan result.

    Reconnaissance can still use a formation.  Strike planning additionally
    populates ``routes`` with one explicit route per aircraft/target binding.
    Existing ``team_paths`` fields are retained for compatibility.
    """

    formation_type: str
    team_count: int
    success: bool
    center_path: list[list[float]] = field(default_factory=list)
    team_paths: list[list[list[float]]] = field(default_factory=list)
    formation_roles: list[str] = field(default_factory=list)
    planner_stats: dict = field(default_factory=dict)
    routes: list[Route] = field(default_factory=list)
    assignment_map: list[dict[str, Any]] = field(default_factory=list)
