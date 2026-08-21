"""Frozen state-cost reward for random-event recovery.

The only reward definition used by Phase J is::

    r_t = J(before) - J(after)
    J = alpha*weighted_uncovered_regions + beta*normalized_distance
        + gamma*load_gap + delta*switch_count + eta*recovery_delay

The five weights and the normalization below intentionally mirror
``configs/random_event_protocol.json``.  Constraint violations are reported as
an audit diagnostic, but are *not* a sixth reward term: action legality is a
hard mask and the frozen protocol explicitly excludes a constraint penalty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

try:
    from config import AREA_SIZE, NO_UAV, TaskType
except ImportError:
    from ..config import AREA_SIZE, NO_UAV, TaskType


FROZEN_REWARD_WEIGHTS = {
    "uncovered": 5.0,
    "distance": 1.0,
    "load_gap": 1.0,
    "switches": 0.25,
    "recovery_delay": 0.5,
}
VACANCY_DURATION_WEIGHT = 0.2

# Frozen censoring horizon for an event that never produced a decision (e.g.
# it was never observed because an earlier event caused final-infeasible
# termination).  Mirrors ``random_event_protocol.json``
# ``validation_metrics.fixed_j_unobserved_event_rule.recovery_delay_seconds``.
# It is an explicit censored upper horizon, NEVER an implicit None -> 0.
UNOBSERVED_EVENT_RECOVERY_PENALTY_SECONDS = 200.0


@dataclass(frozen=True)
class CostWeights:
    """Protocol mapping alpha/beta/gamma/delta/eta -> J components."""

    uncovered: float = FROZEN_REWARD_WEIGHTS["uncovered"]
    distance: float = FROZEN_REWARD_WEIGHTS["distance"]
    load_gap: float = FROZEN_REWARD_WEIGHTS["load_gap"]
    switches: float = FROZEN_REWARD_WEIGHTS["switches"]
    recovery_delay: float = FROZEN_REWARD_WEIGHTS["recovery_delay"]
    # Retained only for backwards-compatible diagnostics; never included in J.
    constraint_violation: float = 0.0


@dataclass(frozen=True)
class CostBreakdown:
    uncovered: float
    distance: float
    load_gap: float
    switches: float
    recovery_delay: float
    constraint_violation: float
    weighted: Mapping[str, float]
    total: float

    def to_dict(self) -> dict:
        value = asdict(self)
        value["weighted"] = dict(self.weighted)
        return value


def assignment_map(env) -> dict[int, int]:
    return {int(rid): int(region.assigned_uav) for rid, region in env.regions.items()}


def compute_cost(
    env,
    weights: CostWeights | None = None,
    reference_assignments: Mapping[int, int] | None = None,
) -> CostBreakdown:
    """Compute the frozen five-term J and separately expose violations.

    ``uncovered`` is the protocol's ``weighted_uncovered_regions``: current
    priority*workload demand plus 0.2 times normalized accumulated vacancy
    duration for each uncovered region.  Distance is normalized by the fixed
    scenario diagonal ``AREA_SIZE*sqrt(2)`` and averaged over assigned regions.
    """
    weights = weights or CostWeights()
    uncovered = 0.0
    violation = 0.0
    distance = 0.0
    assigned_count = 0
    diagonal = max(1e-9, AREA_SIZE * np.sqrt(2.0))
    vacancy_duration = getattr(env, "vacancy_duration", {})
    max_time = max(1.0, float(getattr(env, "max_time", 1.0)))

    for rid, region in env.regions.items():
        uid = int(region.assigned_uav)
        priority = float(getattr(region, "priority", 1.0))
        workload = float(getattr(region, "workload", 1.0))
        legal = uid != NO_UAV and env._valid_search_assign(uid, rid)
        if not legal:
            vacancy = float(vacancy_duration.get(int(rid), 0.0)) / max_time
            uncovered += priority * workload + VACANCY_DURATION_WEIGHT * vacancy
            if uid != NO_UAV:
                violation += 1.0
            continue
        uav = env.uavs[uid]
        distance += float(np.hypot(uav.x - region.center_x, uav.y - region.center_y)) / diagonal
        assigned_count += 1
    if assigned_count:
        distance /= assigned_count

    loads = [
        len(u.regions)
        for u in env.uavs.values()
        if u.alive and not u.sensor_failed and u.task != TaskType.TRACK
    ]
    load_gap = float(max(loads) - min(loads)) if len(loads) > 1 else 0.0

    switches = 0.0
    if reference_assignments is not None:
        switches = float(sum(
            int(int(env.regions[rid].assigned_uav) != int(old_uid))
            for rid, old_uid in reference_assignments.items()
        ))

    recovery_delay = float(sum(
        float(vacancy_duration.get(int(rid), 0.0))
        for rid in getattr(env, "pending_regions", ())
    ))
    raw = {
        "uncovered": uncovered,
        "distance": distance,
        "load_gap": load_gap,
        "switches": switches,
        "recovery_delay": recovery_delay,
        "constraint_violation": violation,
    }
    # Exactly the five frozen protocol terms.  Constraint violation remains a
    # raw audit field but cannot silently alter reward.
    weighted = {
        name: float(raw[name] * getattr(weights, name))
        for name in FROZEN_REWARD_WEIGHTS
    }
    return CostBreakdown(
        uncovered=uncovered,
        distance=distance,
        load_gap=load_gap,
        switches=switches,
        recovery_delay=recovery_delay,
        constraint_violation=violation,
        weighted=weighted,
        total=float(sum(weighted.values())),
    )


def compute_fixed_j_from_components(
    *,
    uncovered: float | None,
    distance: float | None,
    load_gap: float | None,
    switches: float | None,
    recovery_delay: float | None,
    weights: CostWeights | None = None,
) -> float:
    """Evaluate frozen J; missing required components are a hard error."""
    values = {
        "uncovered": uncovered,
        "distance": distance,
        "load_gap": load_gap,
        "switches": switches,
        "recovery_delay": recovery_delay,
    }
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ValueError(f"fixed J requires finite components; missing: {', '.join(missing)}")
    weights = weights or CostWeights()
    return float(sum(float(values[name]) * getattr(weights, name) for name in FROZEN_REWARD_WEIGHTS))


def cost_difference_reward(before: CostBreakdown, after: CostBreakdown) -> tuple[float, dict]:
    components = {
        name: float(before.weighted[name] - after.weighted[name])
        for name in before.weighted
    }
    reward = float(before.total - after.total)
    return reward, {
        "definition": "J(before)-J(after)",
        "formula": "alpha*weighted_uncovered_regions+beta*normalized_distance+gamma*load_gap+delta*switch_count+eta*recovery_delay",
        "constraint_term_included": False,
        "before": before.to_dict(),
        "after": after.to_dict(),
        "reward_components": components,
        "reward": reward,
    }
