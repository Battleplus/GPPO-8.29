"""Framework tensor conversion for the frozen policy adapter.

The repository's GPPO implementation is PyTorch-native and does not require
torch-geometric.  This module therefore exposes typed tensors directly while
retaining a lossless path to PyG-style node/edge dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .adapter import (
    ActionSpaceSnapshot,
    FlatPolicyObservation,
    HeteroPolicyObservation,
)
from .graph import NODE_TYPES, RELATIONS


@dataclass(frozen=True, slots=True)
class TorchFlatObservation:
    vector: torch.Tensor
    action_mask: torch.Tensor
    action_space: ActionSpaceSnapshot
    graph_version: int
    graph_sha256: str

    def to(self, device: str | torch.device) -> "TorchFlatObservation":
        return TorchFlatObservation(
            vector=self.vector.to(device),
            action_mask=self.action_mask.to(device),
            action_space=self.action_space,
            graph_version=self.graph_version,
            graph_sha256=self.graph_sha256,
        )


@dataclass(frozen=True, slots=True)
class TorchHeteroObservation:
    nodes: Mapping[str, torch.Tensor]
    edge_index: Mapping[tuple[str, str, str], torch.Tensor]
    node_ids: Mapping[str, tuple[str, ...]]
    rule_context: torch.Tensor
    action_mask: torch.Tensor
    action_space: ActionSpaceSnapshot
    graph_version: int
    graph_sha256: str

    def to(self, device: str | torch.device) -> "TorchHeteroObservation":
        return TorchHeteroObservation(
            nodes={key: value.to(device) for key, value in self.nodes.items()},
            edge_index={key: value.to(device) for key, value in self.edge_index.items()},
            node_ids=self.node_ids,
            rule_context=self.rule_context.to(device),
            action_mask=self.action_mask.to(device),
            action_space=self.action_space,
            graph_version=self.graph_version,
            graph_sha256=self.graph_sha256,
        )


def flat_to_torch(
    observation: FlatPolicyObservation,
    *,
    device: str | torch.device = "cpu",
) -> TorchFlatObservation:
    observation.validate()
    return TorchFlatObservation(
        vector=torch.tensor(observation.vector, dtype=torch.float32, device=device),
        action_mask=torch.tensor(observation.action_space.mask, dtype=torch.bool, device=device),
        action_space=observation.action_space,
        graph_version=observation.graph_version,
        graph_sha256=observation.graph_sha256,
    )


def hetero_to_torch(
    observation: HeteroPolicyObservation,
    *,
    device: str | torch.device = "cpu",
) -> TorchHeteroObservation:
    observation.validate()
    from .graph import FEATURE_DIMENSIONS
    nodes = {}
    for node_type in NODE_TYPES:
        rows = observation.node_features[node_type]
        if rows:
            nodes[node_type] = torch.tensor(rows, dtype=torch.float32, device=device)
        else:
            nodes[node_type] = torch.empty(
                (0, FEATURE_DIMENSIONS[node_type]), dtype=torch.float32, device=device
            )
    edge_index = {}
    for relation in RELATIONS:
        rows = observation.edge_indices[relation]
        if rows:
            edge_index[relation] = torch.tensor(
                rows, dtype=torch.long, device=device
            ).t().contiguous()
        else:
            edge_index[relation] = torch.empty((2, 0), dtype=torch.long, device=device)
    return TorchHeteroObservation(
        nodes=nodes,
        edge_index=edge_index,
        node_ids=observation.node_ids,
        rule_context=torch.tensor(
            observation.rule_context, dtype=torch.float32, device=device
        ),
        action_mask=torch.tensor(observation.action_space.mask, dtype=torch.bool, device=device),
        action_space=observation.action_space,
        graph_version=observation.graph_version,
        graph_sha256=observation.graph_sha256,
    )


__all__ = [
    "TorchFlatObservation",
    "TorchHeteroObservation",
    "flat_to_torch",
    "hetero_to_torch",
]
