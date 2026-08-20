"""Generic algorithm-result type used across all adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AlgorithmResult:
    """Carrier for the outcome of any algorithm call (MILP, MPPI, position).

    Attributes:
        success: ``True`` when the algorithm produced a usable result.
        data:    Algorithm-specific payload (e.g. ``AllocationPlan``,
                 ``FormationPlanResult``, list of ``Position``).
        reason:  Human-readable explanation on failure.
    """

    success: bool
    data: Any | None = None
    reason: str = ""

    @classmethod
    def ok(cls, data: Any = None) -> AlgorithmResult:
        """Shorthand for a successful result."""
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, reason: str) -> AlgorithmResult:
        """Shorthand for a failed result."""
        return cls(success=False, reason=reason)
