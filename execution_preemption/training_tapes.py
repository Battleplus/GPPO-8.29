"""Deterministic training-only tape stream for Execution-Preemption V1."""

from __future__ import annotations

import copy
import hashlib
import random
from typing import Any

from .contract import TRAINING_SCALES, TRAINING_SEEDS
from .tapes import SCENARIO_CATALOG, build_development_tape, validate_tape


TRAINING_BANK = "Execution-Preemption-Train"
TRAINING_NAMESPACE = "execution_preemption_v1/train"
TRAINING_TASKS_PER_UAV = (2, 3)


def training_case_seed(policy_seed: int, uav_count: int, episode_index: int) -> int:
    if policy_seed not in TRAINING_SEEDS:
        raise ValueError("policy seed is outside the frozen training contract")
    if uav_count not in TRAINING_SCALES:
        raise ValueError("UAV count is outside the frozen training scales")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    identity = f"{TRAINING_NAMESPACE}/{policy_seed}/{uav_count}/{episode_index}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return 1_000_000_000 + int.from_bytes(digest[:8], "big") % 900_000_000


def _new_uav(uav_id: str, rng: random.Random) -> dict[str, Any]:
    return {
        "uav_id": uav_id,
        "energy_ratio": round(rng.uniform(0.72, 0.98), 6),
        "reserve_energy": 0.1,
        "estimated_rtb_energy": round(rng.uniform(0.08, 0.14), 6),
        "communication_state": "CONNECTED",
        "supported_task_types": ["SEARCH", "STRIKE", "URGENT"],
    }


def _new_task(
    task_id: str,
    rng: random.Random,
    *,
    assigned_uav: str | None = None,
) -> dict[str, Any]:
    running = assigned_uav is not None
    return {
        "task_id": task_id,
        "task_type": "SEARCH",
        "priority": rng.randint(10, 70),
        "deadline": round(rng.uniform(80.0, 160.0), 6),
        "state": "RUNNING" if running else "PENDING",
        "progress": round(rng.uniform(0.05, 0.75), 6) if running else 0.0,
        "assigned_uav": assigned_uav,
        "preemptible": True,
        "resume_policy": "ANY_COMPATIBLE",
    }


def build_training_tape(
    *,
    policy_seed: int,
    uav_count: int,
    episode_index: int,
    tasks_per_uav: int | None = None,
) -> dict[str, Any]:
    """Build one paired, training-only episode without consuming Dev/Hidden seeds."""
    if tasks_per_uav is None:
        tasks_per_uav = TRAINING_TASKS_PER_UAV[episode_index % 2]
    if tasks_per_uav not in TRAINING_TASKS_PER_UAV:
        raise ValueError("tasks_per_uav must be 2 or 3")
    seed = training_case_seed(policy_seed, uav_count, episode_index)
    rng = random.Random(seed)
    scenario = SCENARIO_CATALOG[episode_index % len(SCENARIO_CATALOG)]["id"]
    base = build_development_tape(scenario, episode_index % 20)
    tape = copy.deepcopy(base)
    shift = round(20.0 + rng.uniform(0.0, 30.0), 6)

    for uav in tape["initial_state"]["uavs"]:
        uav["energy_ratio"] = round(
            min(0.99, max(0.16, float(uav["energy_ratio"]) + rng.uniform(-0.025, 0.025))),
            6,
        )
    for task in tape["initial_state"]["tasks"]:
        if task.get("deadline") is not None:
            task["deadline"] = round(float(task["deadline"]) + shift, 6)
    for event_index, event in enumerate(tape["events"]):
        event["event_id"] = f"TR-{episode_index:08d}-{event_index:02d}"
        event["batch_id"] = f"TRB-{episode_index:08d}-{event.get('batch_id', event_index)}"
        event["occurred_at"] = round(float(event["occurred_at"]) + shift, 6)
        event["received_at"] = round(float(event["received_at"]) + shift, 6)
        if event.get("deadline") is not None:
            event["deadline"] = round(float(event["deadline"]) + shift, 6)

    uavs = tape["initial_state"]["uavs"]
    tasks = tape["initial_state"]["tasks"]
    for index in range(len(uavs), uav_count):
        uav_id = f"U{index}"
        uavs.append(_new_uav(uav_id, rng))
        tasks.append(_new_task(f"TRAIN-RUN-{index:02d}", rng, assigned_uav=uav_id))
    target_task_count = uav_count * tasks_per_uav
    while len(tasks) < target_task_count:
        index = len(tasks)
        tasks.append(_new_task(f"TRAIN-PENDING-{index:03d}", rng))
    if len(tasks) > target_task_count:
        raise ValueError("base scenario exceeds the requested training task cardinality")

    tape.update({
        "bank": TRAINING_BANK,
        "classification": "training_only_not_development_or_hidden",
        "case_index": episode_index,
        "case_seed": seed,
        "tape_id": (
            f"train-s{policy_seed}-u{uav_count}-t{tasks_per_uav}-"
            f"e{episode_index:08d}-{seed}"
        ),
        "training_identity": {
            "namespace": TRAINING_NAMESPACE,
            "policy_seed": policy_seed,
            "uav_count": uav_count,
            "tasks_per_uav": tasks_per_uav,
            "episode_index": episode_index,
        },
    })
    validate_tape(tape)
    return tape


__all__ = [
    "TRAINING_BANK",
    "TRAINING_NAMESPACE",
    "TRAINING_TASKS_PER_UAV",
    "build_training_tape",
    "training_case_seed",
]
