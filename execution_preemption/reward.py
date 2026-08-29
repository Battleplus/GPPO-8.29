"""Frozen transition reward for Execution-Preemption V1.

This module is intentionally independent from the legacy random-event reward.
All inputs are normalized, algorithm-independent transition signals.  Safety
violations are hard failures rather than scalar penalties that a policy could
trade against task utility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Mapping


REWARD_CONTRACT_ID = "execution-preemption-reward-v1"

REWARD_WEIGHTS: Mapping[str, float] = MappingProxyType({
    "weighted_progress_gain": 4.0,
    "urgent_deadline_miss_rate": -10.0,
    "weighted_vacancy_time": -5.0,
    "progress_loss": -4.0,
    "starvation_exposure": -3.0,
    "switch_time": -1.0,
    "energy_consumed": -1.0,
    "normalized_distance": -0.5,
    "load_gap": -0.5,
})

NORMALIZED_SIGNAL_NAMES = tuple(REWARD_WEIGHTS)
HARD_SAFETY_SIGNAL_NAMES = (
    "resource_conflicts",
    "stale_command_resurrections",
    "energy_safety_violations",
)


class HardSafetyViolation(RuntimeError):
    """Raised when an invalid transition must stop the run."""


def _require_unit_interval(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _require_count(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class TransitionSignals:
    """Normalized evidence for one accepted allocation transition.

    Every soft signal is in ``[0, 1]``.  The caller must normalize using the
    frozen horizons in the training contract before constructing this object.
    Hard safety counts are diagnostic gates and are never reward terms.
    """

    weighted_progress_gain: float = 0.0
    urgent_deadline_miss_rate: float = 0.0
    weighted_vacancy_time: float = 0.0
    progress_loss: float = 0.0
    starvation_exposure: float = 0.0
    switch_time: float = 0.0
    energy_consumed: float = 0.0
    normalized_distance: float = 0.0
    load_gap: float = 0.0
    resource_conflicts: int = 0
    stale_command_resurrections: int = 0
    energy_safety_violations: int = 0

    def __post_init__(self) -> None:
        for name in NORMALIZED_SIGNAL_NAMES:
            object.__setattr__(self, name, _require_unit_interval(name, getattr(self, name)))
        for name in HARD_SAFETY_SIGNAL_NAMES:
            object.__setattr__(self, name, _require_count(name, getattr(self, name)))

    @property
    def hard_violation_count(self) -> int:
        return sum(int(getattr(self, name)) for name in HARD_SAFETY_SIGNAL_NAMES)

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    contract_id: str
    definition: str
    raw_signals: Mapping[str, float]
    weighted_components: Mapping[str, float]
    reward: float
    hard_constraints_in_reward: bool
    eligible_for_learning: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "definition": self.definition,
            "raw_signals": dict(self.raw_signals),
            "weighted_components": dict(self.weighted_components),
            "reward": self.reward,
            "hard_constraints_in_reward": self.hard_constraints_in_reward,
            "eligible_for_learning": self.eligible_for_learning,
        }


def compute_transition_reward(signals: TransitionSignals) -> RewardBreakdown:
    """Return the frozen reward, or stop on any hard safety violation."""

    if not isinstance(signals, TransitionSignals):
        raise TypeError("signals must be TransitionSignals")
    if signals.hard_violation_count:
        details = ", ".join(
            f"{name}={getattr(signals, name)}"
            for name in HARD_SAFETY_SIGNAL_NAMES
            if getattr(signals, name)
        )
        raise HardSafetyViolation(
            f"transition is ineligible for learning because hard safety failed: {details}"
        )
    raw = {name: float(getattr(signals, name)) for name in NORMALIZED_SIGNAL_NAMES}
    weighted = {name: float(raw[name] * REWARD_WEIGHTS[name]) for name in NORMALIZED_SIGNAL_NAMES}
    reward = float(math.fsum(weighted.values()))
    if not math.isfinite(reward):
        raise ValueError("reward must be finite")
    return RewardBreakdown(
        contract_id=REWARD_CONTRACT_ID,
        definition="sum(weight_i * normalized_transition_signal_i)",
        raw_signals=raw,
        weighted_components=weighted,
        reward=reward,
        hard_constraints_in_reward=False,
        eligible_for_learning=True,
    )


__all__ = [
    "HARD_SAFETY_SIGNAL_NAMES",
    "HardSafetyViolation",
    "NORMALIZED_SIGNAL_NAMES",
    "REWARD_CONTRACT_ID",
    "REWARD_WEIGHTS",
    "RewardBreakdown",
    "TransitionSignals",
    "compute_transition_reward",
]

