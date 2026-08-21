"""Heterogeneous graph observation and edge-action construction.

The legacy environment exposes a flat 165-dimensional vector and a four-way
MultiDiscrete action.  This module leaves that interface untouched and builds a
parallel graph view for GPPO.  The graph contains UAV, Region and Target nodes;
an action chooses one UAV--Region edge, or the final NOOP action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
import torch

try:  # script-style imports used by the original ppo_allocation project
    from config import AREA_SIZE, NUM_REGIONS, NUM_TARGETS, NUM_UAVS, NO_TARGET, NO_UAV, TaskType
except ImportError:  # package-style imports used by the new tests
    from ..config import AREA_SIZE, NUM_REGIONS, NUM_TARGETS, NUM_UAVS, NO_TARGET, NO_UAV, TaskType


Relation = Tuple[str, str, str]


@dataclass(frozen=True)
class HeteroGraphState:
    """Tensor representation consumed by :class:`GraphActorCritic`.

    ``candidate_edges[i] == (u, r)`` maps action ``i`` to assigning Region r
    to UAV u.  The last action is always NOOP and therefore has no edge row.
    """

    nodes: Mapping[str, torch.Tensor]
    edge_index: Mapping[Relation, torch.Tensor]
    edge_attr: Mapping[Relation, torch.Tensor]
    candidate_edges: torch.Tensor
    action_mask: torch.Tensor
    graph_version: int

    @property
    def noop_action(self) -> int:
        return int(self.candidate_edges.shape[0])

    @property
    def num_actions(self) -> int:
        return self.noop_action + 1

    def to(self, device: torch.device | str) -> "HeteroGraphState":
        return HeteroGraphState(
            nodes={k: v.to(device) for k, v in self.nodes.items()},
            edge_index={k: v.to(device) for k, v in self.edge_index.items()},
            edge_attr={k: v.to(device) for k, v in self.edge_attr.items()},
            candidate_edges=self.candidate_edges.to(device),
            action_mask=self.action_mask.to(device),
            graph_version=self.graph_version,
        )


def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def _pending_regions(env) -> set[int]:
    pending = getattr(env, "pending_regions", None)
    if pending is not None:
        return {int(r) for r in pending}
    return {rid for rid, r in env.regions.items() if bool(r.need_reassign)}


def _region_workload(region) -> float:
    return float(getattr(region, "workload", 1.0))


def _vacancy_duration(env, rid: int) -> float:
    durations = getattr(env, "vacancy_duration", {})
    horizon = max(1.0, float(getattr(env, "max_time", getattr(env, "max_decision_steps", 50))))
    return min(1.0, float(durations.get(rid, 0.0)) / horizon)


def _comm_quality(env, u: int, v: int) -> float:
    quality = getattr(env, "communication_quality", None)
    if quality is None:
        return 1.0
    if isinstance(quality, Mapping):
        return float(quality.get((u, v), quality.get(f"{u}-{v}", 1.0)))
    return float(quality[u, v])


def build_graph_state(env) -> HeteroGraphState:
    """Build a deterministic heterogeneous graph from an environment state."""

    pending = _pending_regions(env)
    uav_rows: list[list[float]] = []
    for uid in range(NUM_UAVS):
        u = env.uavs[uid]
        task = _one_hot(int(u.task), 3)
        target = _one_hot(u.target_id if u.target_id != NO_TARGET else NUM_TARGETS, NUM_TARGETS + 1)
        uav_rows.append([
            float(u.alive),
            float(not u.sensor_failed),
            *task,
            float(u.x) / AREA_SIZE,
            float(u.y) / AREA_SIZE,
            len(u.regions) / max(1, NUM_REGIONS),
            *target,
        ])

    region_rows: list[list[float]] = []
    for rid in range(NUM_REGIONS):
        r = env.regions[rid]
        assigned = _one_hot(r.assigned_uav if r.assigned_uav != NO_UAV else NUM_UAVS, NUM_UAVS + 1)
        legal = r.assigned_uav != NO_UAV and env._valid_search_assign(r.assigned_uav, rid)
        region_rows.append([
            float(r.center_x) / AREA_SIZE,
            float(r.center_y) / AREA_SIZE,
            float(getattr(r, "priority", 1.0)),
            _region_workload(r),
            _vacancy_duration(env, rid),
            *assigned,
            float(rid in pending),
            float(legal),
        ])

    target_rows: list[list[float]] = []
    for tid in range(NUM_TARGETS):
        t = env.targets[tid]
        target_type = _one_hot(int(t.target_type), 2)
        region = _one_hot(t.region, NUM_REGIONS)
        tracker = _one_hot(t.tracker_id if t.tracker_id != NO_UAV else NUM_UAVS, NUM_UAVS + 1)
        target_rows.append([
            *target_type,
            float(t.discovered),
            float(t.tracked),
            float(t.destroyed),
            (float(t.x) / AREA_SIZE) if t.discovered else 0.0,
            (float(t.y) / AREA_SIZE) if t.discovered else 0.0,
            *region,
            *tracker,
        ])

    edges: Dict[Relation, list[tuple[int, int]]] = {}
    attrs: Dict[Relation, list[list[float]]] = {}

    def add(rel: Relation, src: int, dst: int, attr: list[float]) -> None:
        edges.setdefault(rel, []).append((src, dst))
        attrs.setdefault(rel, []).append(attr)

    candidate: list[tuple[int, int]] = []
    mask: list[bool] = []
    diagonal = max(1e-9, AREA_SIZE * np.sqrt(2.0))
    for uid in range(NUM_UAVS):
        u = env.uavs[uid]
        for rid in range(NUM_REGIONS):
            r = env.regions[rid]
            distance = float(np.hypot(u.x - r.center_x, u.y - r.center_y)) / diagonal
            capable = bool(u.alive and not u.sensor_failed and u.task != TaskType.TRACK)
            current = r.assigned_uav == uid
            feature = [float(capable), distance, float(current), len(u.regions) / max(1, NUM_REGIONS), 1.0]
            add(("uav", "can_serve", "region"), uid, rid, feature)
            add(("region", "served_by", "uav"), rid, uid, feature)
            candidate.append((uid, rid))
            # Only pending regions may change.  This enforces local recovery and
            # guarantees that assignments unaffected by an event are preserved.
            mask.append(bool(rid in pending and capable))

    adjacency = getattr(env, "region_adjacency", None)
    if adjacency is None:
        try:
            from config import REGION_ADJACENCY
        except ImportError:
            from ..config import REGION_ADJACENCY
        adjacency = REGION_ADJACENCY
    for src, neighbours in adjacency.items():
        for dst in sorted(neighbours):
            add(("region", "adjacent", "region"), int(src), int(dst), [1.0])

    for tid, t in env.targets.items():
        add(("target", "located_in", "region"), tid, int(t.region), [float(t.discovered), float(t.destroyed)])
        add(("region", "contains", "target"), int(t.region), tid, [float(t.discovered), float(t.destroyed)])
        for uid in range(NUM_UAVS):
            tracking = float(t.tracked and t.tracker_id == uid)
            add(("uav", "tracks", "target"), uid, tid, [tracking, float(t.discovered)])
            add(("target", "tracked_by", "uav"), tid, uid, [tracking, float(t.discovered)])

    for src in range(NUM_UAVS):
        for dst in range(NUM_UAVS):
            if src != dst:
                add(("uav", "communicates", "uav"), src, dst, [_comm_quality(env, src, dst)])

    edge_index = {
        rel: torch.tensor(values, dtype=torch.long).t().contiguous()
        for rel, values in edges.items()
    }
    edge_attr = {rel: torch.tensor(values, dtype=torch.float32) for rel, values in attrs.items()}
    action_mask = torch.tensor(mask + [len(pending) == 0], dtype=torch.bool)
    # A pending region with no feasible UAV is temporarily infeasible: NOOP is
    # available so the queue can wait for a later release event.
    if pending and not any(mask):
        action_mask[-1] = True

    return HeteroGraphState(
        nodes={
            "uav": torch.tensor(uav_rows, dtype=torch.float32),
            "region": torch.tensor(region_rows, dtype=torch.float32),
            "target": torch.tensor(target_rows, dtype=torch.float32),
        },
        edge_index=edge_index,
        edge_attr=edge_attr,
        candidate_edges=torch.tensor(candidate, dtype=torch.long),
        action_mask=action_mask,
        graph_version=int(getattr(env, "graph_version", 0)),
    )


def decode_edge_action(graph: HeteroGraphState, action: int) -> tuple[int, int] | None:
    """Return ``(uav_id, region_id)`` or ``None`` for NOOP."""

    action = int(action)
    if action == graph.noop_action:
        return None
    if not 0 <= action < graph.noop_action:
        raise ValueError(f"action {action} outside [0, {graph.noop_action}]")
    uid, rid = graph.candidate_edges[action].tolist()
    return int(uid), int(rid)
