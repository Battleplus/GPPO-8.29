"""Cost-difference reward for random-event recovery.

Unlike the legacy absolute reward (which contains a large constant coverage
bonus), this reward measures whether one decision improved the current state:

    reward = J(before) - J(after)

Every raw and weighted component is returned for traceable experiment logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

try:
    from config import AREA_SIZE, NO_UAV, TaskType
except ImportError:
    from ..config import AREA_SIZE, NO_UAV, TaskType


@dataclass(frozen=True)
class CostWeights:
    uncovered: float = 20.0
    distance: float = 3.0
    load_gap: float = 2.0
    switches: float = 1.0
    recovery_delay: float = 2.0
    constraint_violation: float = 20.0


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
    weights = weights or CostWeights()
    uncovered = 0.0
    violation = 0.0
    distance = 0.0
    assigned_count = 0
    diagonal = max(1e-9, AREA_SIZE * np.sqrt(2.0))

    for rid, region in env.regions.items():
        uid = int(region.assigned_uav)
        priority = float(getattr(region, "priority", 1.0))
        workload = float(getattr(region, "workload", 1.0))
        legal = uid != NO_UAV and env._valid_search_assign(uid, rid)
        if not legal:
            uncovered += priority * workload
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

    vacancy = getattr(env, "vacancy_duration", {})
    recovery_delay = float(sum(float(vacancy.get(int(rid), 0.0)) for rid in getattr(env, "pending_regions", ())))

    raw = {
        "uncovered": uncovered,
        "distance": distance,
        "load_gap": load_gap,
        "switches": switches,
        "recovery_delay": recovery_delay,
        "constraint_violation": violation,
    }
    weighted = {name: float(value * getattr(weights, name)) for name, value in raw.items()}
    return CostBreakdown(**raw, weighted=weighted, total=float(sum(weighted.values())))


def cost_difference_reward(before: CostBreakdown, after: CostBreakdown) -> tuple[float, dict]:
    components = {
        name: float(before.weighted[name] - after.weighted[name])
        for name in before.weighted
    }
    reward = float(before.total - after.total)
    return reward, {
        "definition": "J(before)-J(after)",
        "before": before.to_dict(),
        "after": after.to_dict(),
        "reward_components": components,
        "reward": reward,
    }
