"""Deterministic and stochastic edge-action baselines.

All policies consume the same graph/action mask as GPPO.  This prevents a
baseline from receiving either weaker constraints or extra state information.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .graph import HeteroGraphState, build_graph_state, decode_edge_action
from .reward import CostWeights, assignment_map, compute_cost


class EdgePolicy(Protocol):
    name: str

    def select_action(self, env, graph: HeteroGraphState, deterministic: bool = True) -> int: ...


def legal_edge_actions(graph: HeteroGraphState) -> list[int]:
    return [i for i in range(graph.noop_action) if bool(graph.action_mask[i])]


@dataclass
class MaskedRandomPolicy:
    seed: int = 0
    name: str = "Masked Random"

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def select_action(self, env, graph: HeteroGraphState, deterministic: bool = False) -> int:
        legal = np.flatnonzero(graph.action_mask.cpu().numpy())
        return int(self.rng.choice(legal))


@dataclass
class NearestLegalPolicy:
    name: str = "Nearest Legal"

    def select_action(self, env, graph: HeteroGraphState, deterministic: bool = True) -> int:
        legal = legal_edge_actions(graph)
        if not legal:
            return graph.noop_action
        return min(legal, key=lambda i: float(graph.edge_attr[("uav", "can_serve", "region")][i, 1]))


@dataclass
class MinLoadPolicy:
    name: str = "Min Load"

    def select_action(self, env, graph: HeteroGraphState, deterministic: bool = True) -> int:
        legal = legal_edge_actions(graph)
        if not legal:
            return graph.noop_action
        return min(
            legal,
            key=lambda i: (
                len(env.uavs[int(graph.candidate_edges[i, 0])].regions),
                float(graph.edge_attr[("uav", "can_serve", "region")][i, 1]),
                i,
            ),
        )


def _apply_edge_to_clone(env, graph: HeteroGraphState, action: int):
    clone = copy.deepcopy(env)
    decoded = decode_edge_action(graph, action)
    if decoded is not None:
        uid, rid = decoded
        clone._assign_region_to_uav(rid, uid)
        if hasattr(clone, "pending_regions"):
            clone.pending_regions.discard(rid)
    return clone


@dataclass
class GreedyCostPolicy:
    weights: CostWeights = CostWeights()
    name: str = "Greedy Cost"

    def select_action(self, env, graph: HeteroGraphState, deterministic: bool = True) -> int:
        legal = legal_edge_actions(graph)
        if not legal:
            return graph.noop_action
        old = assignment_map(env)
        return min(
            legal,
            key=lambda action: compute_cost(
                _apply_edge_to_clone(env, graph, action), self.weights, reference_assignments=old
            ).total,
        )


@dataclass
class CurrentPendingExactPlannerPolicy:
    """Enumerate all legal assignments for current pending regions and return the one with minimum cost.
    
    This planner only considers currently pending regions, not future events.
    It is NOT a global upper bound.
    """

    weights: CostWeights = CostWeights()
    name: str = "Current-Pending Exact Planner"

    def select_action(self, env, graph: HeteroGraphState, deterministic: bool = True) -> int:
        pending = sorted(int(r) for r in getattr(env, "pending_regions", ()))
        if not pending:
            return graph.noop_action
        options: list[list[int]] = []
        for rid in pending:
            actions = [
                i for i in legal_edge_actions(graph)
                if int(graph.candidate_edges[i, 1]) == rid
            ]
            if not actions:
                return graph.noop_action
            options.append(actions)

        reference = assignment_map(env)
        best_cost = float("inf")
        best_plan: tuple[int, ...] | None = None
        for plan in itertools.product(*options):
            clone = copy.deepcopy(env)
            for action in plan:
                uid, rid = decode_edge_action(graph, action)  # type: ignore[misc]
                clone._assign_region_to_uav(rid, uid)
                if hasattr(clone, "pending_regions"):
                    clone.pending_regions.discard(rid)
            cost = compute_cost(clone, self.weights, reference_assignments=reference).total
            if cost < best_cost or (cost == best_cost and (best_plan is None or plan < best_plan)):
                best_cost, best_plan = cost, plan
        assert best_plan is not None
        return int(best_plan[0])


@dataclass
class GraphPolicyAdapter:
    model: object
    name: str

    def select_action(self, env, graph: HeteroGraphState, deterministic: bool = True) -> int:
        action, _, _, _ = self.model.act(graph, deterministic=deterministic)
        return int(action)


def default_baselines(seed: int = 0) -> list[EdgePolicy]:
    return [
        MaskedRandomPolicy(seed=seed),
        NearestLegalPolicy(),
        MinLoadPolicy(),
        GreedyCostPolicy(),
        CurrentPendingExactPlannerPolicy(),
    ]
