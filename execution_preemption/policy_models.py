"""Small V1 PPO/GPPO actor-critics used for framework integration smoke.

These are new-contract models.  They do not load or modify legacy checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.distributions import Categorical

from .adapter import ACTION_CAPACITY, FLAT_OBSERVATION_DIMENSION, RULE_CONTEXT_DIMENSION
from .framework import TorchFlatObservation, TorchHeteroObservation
from .graph import FEATURE_DIMENSIONS, NODE_TYPES, RELATIONS


def _relation_key(relation: tuple[str, str, str]) -> str:
    return "__".join(relation)


def _masked_outputs(raw_logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if raw_logits.shape != mask.shape:
        raise ValueError("logit/mask shape mismatch")
    if not bool(mask.any()):
        raise ValueError("action mask has no legal action")
    raw_probabilities = torch.softmax(raw_logits, dim=-1)
    invalid_mass = raw_probabilities[~mask].sum()
    masked = raw_logits.masked_fill(~mask, torch.finfo(raw_logits.dtype).min)
    return masked, invalid_mass


class ExecutionPPOMLP(nn.Module):
    policy_id = "ppo_mlp_execution_preemption_v1"

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(FLAT_OBSERVATION_DIMENSION, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(self.hidden_dim, ACTION_CAPACITY)
        self.critic = nn.Linear(self.hidden_dim, 1)

    def forward(self, observation: TorchFlatObservation):
        hidden = self.encoder(observation.vector)
        raw_logits = self.actor(hidden)
        logits, invalid_mass = _masked_outputs(raw_logits, observation.action_mask)
        value = self.critic(hidden).squeeze(-1)
        return logits, value, {
            "pre_mask_invalid_probability": invalid_mass,
            "raw_logits": raw_logits,
        }

    @torch.no_grad()
    def act(self, observation: TorchFlatObservation, *, deterministic: bool = True):
        logits, value, diagnostics = self(observation)
        distribution = Categorical(logits=logits)
        action = torch.argmax(logits) if deterministic else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item()), {
            "pre_mask_invalid_probability": float(
                diagnostics["pre_mask_invalid_probability"].item()
            ),
        }


class AdaptiveRelationLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.messages = nn.ModuleDict({
            _relation_key(relation): nn.Linear(hidden_dim, hidden_dim, bias=False)
            for relation in RELATIONS
        })
        self.updates = nn.ModuleDict({
            node_type: nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())
            for node_type in NODE_TYPES
        })
        self.gates = nn.ModuleDict({
            node_type: nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())
            for node_type in NODE_TYPES
        })

    def forward(self, hidden, edge_index):
        totals = {key: torch.zeros_like(value) for key, value in hidden.items()}
        counts = {
            key: torch.zeros((value.shape[0], 1), dtype=value.dtype, device=value.device)
            for key, value in hidden.items()
        }
        for relation in RELATIONS:
            src_type, _, dst_type = relation
            index = edge_index[relation]
            if index.numel() == 0:
                continue
            src, dst = index[0], index[1]
            messages = self.messages[_relation_key(relation)](hidden[src_type][src])
            totals[dst_type].index_add_(0, dst, messages)
            counts[dst_type].index_add_(
                0, dst,
                torch.ones((dst.shape[0], 1), dtype=messages.dtype, device=messages.device),
            )
        output = {}
        gates = {}
        for node_type in NODE_TYPES:
            current = hidden[node_type]
            aggregate = totals[node_type] / counts[node_type].clamp_min(1.0)
            joined = torch.cat([current, aggregate], dim=-1)
            proposal = self.updates[node_type](joined)
            gate = self.gates[node_type](joined)
            output[node_type] = gate * proposal + (1.0 - gate) * current
            gates[node_type] = gate
        return output, gates


class ExecutionGPPOAdaptive(nn.Module):
    policy_id = "gppo_adaptive_execution_preemption_v1"

    def __init__(self, hidden_dim: int = 64, layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.encoders = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(FEATURE_DIMENSIONS[node_type], self.hidden_dim), nn.Tanh()
            )
            for node_type in NODE_TYPES
        })
        self.layers = nn.ModuleList(
            AdaptiveRelationLayer(self.hidden_dim) for _ in range(int(layers))
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(RULE_CONTEXT_DIMENSION, self.hidden_dim), nn.Tanh()
        )
        self.edge_actor = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        pooled_dim = (len(NODE_TYPES) + 1) * self.hidden_dim
        self.noop_actor = nn.Sequential(
            nn.Linear(pooled_dim, self.hidden_dim), nn.Tanh(), nn.Linear(self.hidden_dim, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(pooled_dim, self.hidden_dim), nn.Tanh(), nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, observation: TorchHeteroObservation):
        hidden = {
            node_type: self.encoders[node_type](observation.nodes[node_type])
            for node_type in NODE_TYPES
        }
        all_gates = {}
        for layer_index, layer in enumerate(self.layers):
            hidden, gates = layer(hidden, observation.edge_index)
            all_gates.update({f"layer{layer_index}/{key}": value for key, value in gates.items()})
        context = self.context_encoder(observation.rule_context)
        id_to_index = {
            node_type: {node_id: index for index, node_id in enumerate(observation.node_ids[node_type])}
            for node_type in NODE_TYPES
        }
        action_indices = []
        uav_indices = []
        task_indices = []
        for action_index, binding in enumerate(observation.action_space.bindings[1:], start=1):
            if binding is None:
                continue
            uav_id, task_id = binding
            if uav_id in id_to_index["UAV"] and task_id in id_to_index["Task"]:
                action_indices.append(action_index)
                uav_indices.append(id_to_index["UAV"][uav_id])
                task_indices.append(id_to_index["Task"][task_id])
        device = context.device
        raw_logits = torch.zeros(ACTION_CAPACITY, dtype=context.dtype, device=device)
        if action_indices:
            uav_tensor = torch.tensor(uav_indices, dtype=torch.long, device=device)
            task_tensor = torch.tensor(task_indices, dtype=torch.long, device=device)
            contexts = context.unsqueeze(0).expand(len(action_indices), -1)
            edge_logits = self.edge_actor(torch.cat([
                hidden["UAV"][uav_tensor], hidden["Task"][task_tensor], contexts
            ], dim=-1)).squeeze(-1)
            raw_logits = raw_logits.index_copy(
                0, torch.tensor(action_indices, dtype=torch.long, device=device), edge_logits
            )
        pooled_parts = []
        for node_type in NODE_TYPES:
            if hidden[node_type].shape[0]:
                pooled_parts.append(hidden[node_type].mean(dim=0))
            else:
                pooled_parts.append(torch.zeros(self.hidden_dim, device=device))
        pooled = torch.cat([*pooled_parts, context], dim=-1)
        raw_logits = raw_logits.index_copy(
            0,
            torch.tensor([0], dtype=torch.long, device=device),
            self.noop_actor(pooled).reshape(1),
        )
        logits, invalid_mass = _masked_outputs(raw_logits, observation.action_mask)
        value = self.critic(pooled).squeeze(-1)
        return logits, value, {
            "pre_mask_invalid_probability": invalid_mass,
            "gates": all_gates,
            "raw_logits": raw_logits,
        }

    @torch.no_grad()
    def act(self, observation: TorchHeteroObservation, *, deterministic: bool = True):
        logits, value, diagnostics = self(observation)
        distribution = Categorical(logits=logits)
        action = torch.argmax(logits) if deterministic else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item()), {
            "pre_mask_invalid_probability": float(
                diagnostics["pre_mask_invalid_probability"].item()
            ),
            "gate_mean": {
                key: float(value.mean().item()) if value.numel() else None
                for key, value in diagnostics["gates"].items()
            },
        }


__all__ = [
    "AdaptiveRelationLayer",
    "ExecutionGPPOAdaptive",
    "ExecutionPPOMLP",
]
