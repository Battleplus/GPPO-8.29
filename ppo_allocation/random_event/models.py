"""Paper-inspired graph actor-critic used by the random-event experiment.

This is a compact dependency-free AHGNN implementation (PyTorch only).  It is
purposefully separate from the legacy Stable-Baselines3 MLP policy so both can
be evaluated on the same event tapes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch
from torch import nn
from torch.distributions import Categorical

from .graph import HeteroGraphState, Relation


RELATIONS: tuple[Relation, ...] = (
    ("uav", "can_serve", "region"),
    ("region", "served_by", "uav"),
    ("region", "adjacent", "region"),
    ("target", "located_in", "region"),
    ("region", "contains", "target"),
    ("uav", "tracks", "target"),
    ("target", "tracked_by", "uav"),
    ("uav", "communicates", "uav"),
)

DEFAULT_EDGE_DIMS: Mapping[Relation, int] = {
    ("uav", "can_serve", "region"): 5,
    ("region", "served_by", "uav"): 5,
    ("region", "adjacent", "region"): 1,
    ("target", "located_in", "region"): 2,
    ("region", "contains", "target"): 2,
    ("uav", "tracks", "target"): 2,
    ("target", "tracked_by", "uav"): 2,
    ("uav", "communicates", "uav"): 1,
}


def _rel_key(rel: Relation) -> str:
    return "__".join(rel)


@dataclass(frozen=True)
class GraphModelConfig:
    hidden_dim: int = 64
    layers: int = 2
    adaptive_gate: bool = True


class AHGNNLayer(nn.Module):
    """Relation-aware aggregation followed by optional adaptive fusion gates."""

    def __init__(self, hidden_dim: int, adaptive_gate: bool, edge_dims: Mapping[Relation, int]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adaptive_gate = adaptive_gate
        self.messages = nn.ModuleDict({
            _rel_key(rel): nn.Linear(hidden_dim + int(edge_dims[rel]), hidden_dim, bias=False)
            for rel in RELATIONS
        })
        self.updates = nn.ModuleDict({
            node: nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())
            for node in ("uav", "region", "target")
        })
        self.gates = nn.ModuleDict({
            node: nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())
            for node in ("uav", "region", "target")
        })

    def forward(
        self,
        hidden: Mapping[str, torch.Tensor],
        edge_index: Mapping[Relation, torch.Tensor],
        edge_attr: Mapping[Relation, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        totals = {name: torch.zeros_like(value) for name, value in hidden.items()}
        counts = {
            name: torch.zeros((value.shape[0], 1), dtype=value.dtype, device=value.device)
            for name, value in hidden.items()
        }
        for rel in RELATIONS:
            if rel not in edge_index:
                continue
            src_type, _, dst_type = rel
            index = edge_index[rel]
            if index.numel() == 0:
                continue
            src, dst = index[0], index[1]
            # Edge features are part of AHGNN aggregation rather than being
            # used only by the Actor.  Thus capability, distance, current
            # ownership, load and communication quality can change both the
            # node embeddings and the whole-graph Critic value.
            message_input = torch.cat([hidden[src_type][src], edge_attr[rel]], dim=-1)
            message = self.messages[_rel_key(rel)](message_input)
            totals[dst_type].index_add_(0, dst, message)
            counts[dst_type].index_add_(
                0,
                dst,
                torch.ones((dst.shape[0], 1), dtype=message.dtype, device=message.device),
            )

        output: Dict[str, torch.Tensor] = {}
        gate_values: Dict[str, torch.Tensor] = {}
        for node_type, self_value in hidden.items():
            aggregate = totals[node_type] / counts[node_type].clamp_min(1.0)
            joined = torch.cat([self_value, aggregate], dim=-1)
            proposal = self.updates[node_type](joined)
            if self.adaptive_gate:
                gate = self.gates[node_type](joined)
                output[node_type] = gate * proposal + (1.0 - gate) * self_value
            else:
                gate = torch.ones_like(proposal)
                output[node_type] = proposal
            gate_values[node_type] = gate
        return output, gate_values


class GraphActorCritic(nn.Module):
    """AHGNN encoder, UAV--Region edge actor and whole-graph critic."""

    def __init__(
        self,
        node_dims: Mapping[str, int],
        config: GraphModelConfig | None = None,
        edge_dims: Mapping[Relation, int] | None = None,
    ):
        super().__init__()
        self.config = config or GraphModelConfig()
        self.node_dims = {k: int(v) for k, v in node_dims.items()}
        self.edge_dims = {
            rel: int((edge_dims or DEFAULT_EDGE_DIMS)[rel])
            for rel in RELATIONS
        }
        h = self.config.hidden_dim
        self.encoders = nn.ModuleDict({
            node: nn.Sequential(nn.Linear(dim, h), nn.Tanh())
            for node, dim in self.node_dims.items()
        })
        self.layers = nn.ModuleList([
            AHGNNLayer(h, adaptive_gate=self.config.adaptive_gate, edge_dims=self.edge_dims)
            for _ in range(self.config.layers)
        ])
        # The edge actor uses both endpoint embeddings and the five raw
        # capability/distance/assignment/load/communication edge features.
        self.edge_actor = nn.Sequential(
            nn.Linear(2 * h + 5, h), nn.Tanh(), nn.Linear(h, 1)
        )
        self.noop_actor = nn.Sequential(nn.Linear(3 * h, h), nn.Tanh(), nn.Linear(h, 1))
        self.critic = nn.Sequential(nn.Linear(3 * h, h), nn.Tanh(), nn.Linear(h, 1))

    @classmethod
    def from_graph(cls, graph: HeteroGraphState, config: GraphModelConfig | None = None) -> "GraphActorCritic":
        return cls(
            {name: value.shape[-1] for name, value in graph.nodes.items()},
            config=config,
            edge_dims={rel: value.shape[-1] for rel, value in graph.edge_attr.items()},
        )

    def encode(self, graph: HeteroGraphState) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        hidden = {name: self.encoders[name](value) for name, value in graph.nodes.items()}
        gates: Dict[str, torch.Tensor] = {}
        for layer_index, layer in enumerate(self.layers):
            hidden, layer_gates = layer(hidden, graph.edge_index, graph.edge_attr)
            gates.update({f"layer{layer_index}/{key}": value for key, value in layer_gates.items()})
        return hidden, gates

    def forward(self, graph: HeteroGraphState) -> tuple[torch.Tensor, torch.Tensor, dict]:
        hidden, gates = self.encode(graph)
        pairs = graph.candidate_edges
        u = hidden["uav"][pairs[:, 0]]
        r = hidden["region"][pairs[:, 1]]
        relation = ("uav", "can_serve", "region")
        edge_features = graph.edge_attr[relation]
        raw_edge_logits = self.edge_actor(torch.cat([u, r, edge_features], dim=-1)).squeeze(-1)
        pooled = torch.cat([
            hidden["uav"].mean(dim=0),
            hidden["region"].mean(dim=0),
            hidden["target"].mean(dim=0),
        ], dim=-1)
        noop_logit = self.noop_actor(pooled).reshape(1)
        raw_logits = torch.cat([raw_edge_logits, noop_logit], dim=0)
        raw_probs = torch.softmax(raw_logits, dim=-1)
        invalid_mass = raw_probs[~graph.action_mask].sum()
        masked_logits = raw_logits.masked_fill(~graph.action_mask, torch.finfo(raw_logits.dtype).min)
        value = self.critic(pooled).squeeze(-1)
        diagnostics = {
            "raw_logits": raw_logits,
            "pre_mask_invalid_probability": invalid_mass,
            "gates": gates,
            "graph_embedding": pooled,
        }
        return masked_logits, value, diagnostics

    def distribution(self, graph: HeteroGraphState) -> tuple[Categorical, torch.Tensor, dict]:
        logits, value, diagnostics = self(graph)
        return Categorical(logits=logits), value, diagnostics

    @torch.no_grad()
    def act(self, graph: HeteroGraphState, deterministic: bool = False) -> tuple[int, float, float, dict]:
        distribution, value, diagnostics = self.distribution(graph)
        action = torch.argmax(distribution.logits) if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        clean = {
            "pre_mask_invalid_probability": float(diagnostics["pre_mask_invalid_probability"].cpu()),
            "gate_mean": {
                name: float(gate.mean().cpu()) for name, gate in diagnostics["gates"].items()
            },
        }
        return int(action.item()), float(log_prob.item()), float(value.item()), clean

    def evaluate_action(self, graph: HeteroGraphState, action: torch.Tensor | int):
        distribution, value, diagnostics = self.distribution(graph)
        action_tensor = torch.as_tensor(action, dtype=torch.long, device=value.device)
        return distribution.log_prob(action_tensor), distribution.entropy(), value, diagnostics

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": "random-event-gppo-v1",
            "node_dims": self.node_dims,
            "edge_dims": {_rel_key(rel): dim for rel, dim in self.edge_dims.items()},
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> tuple["GraphActorCritic", dict]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if payload.get("format") != "random-event-gppo-v1":
            raise ValueError("unsupported graph-policy checkpoint format")
        saved_edge_dims = payload.get("edge_dims")
        edge_dims = None
        if saved_edge_dims is not None:
            edge_dims = {tuple(key.split("__")): int(value) for key, value in saved_edge_dims.items()}
        model = cls(payload["node_dims"], GraphModelConfig(**payload["config"]), edge_dims=edge_dims)
        model.load_state_dict(payload["state_dict"])
        return model, dict(payload.get("extra", {}))


def make_no_gate_model(graph: HeteroGraphState, hidden_dim: int = 64, layers: int = 2) -> GraphActorCritic:
    return GraphActorCritic.from_graph(
        graph,
        GraphModelConfig(hidden_dim=hidden_dim, layers=layers, adaptive_gate=False),
    )


def make_adaptive_model(graph: HeteroGraphState, hidden_dim: int = 64, layers: int = 2) -> GraphActorCritic:
    return GraphActorCritic.from_graph(
        graph,
        GraphModelConfig(hidden_dim=hidden_dim, layers=layers, adaptive_gate=True),
    )


class FairPPOMLP(nn.Module):
    """Fair PPO-MLP baseline that uses the same graph inputs as GPPO.
    
    This model reads the canonical flattened graph (nodes, edges, edge features)
    but uses a simple MLP instead of AHGNN for encoding. The only difference
    from GPPO is the network architecture: MLP vs AHGNN vs AHGNN+Gate.
    
    This ensures a fair comparison where the graph information, pending/queue/
    lease/communication/version features, edge+NOOP actions, mask, reward, PPO
    parameters, budget, seed, Validation and checkpoint intervals are identical.
    """

    def __init__(
        self,
        graph: HeteroGraphState,
        hidden_dim: int = 64,
        layers: int = 2,
    ):
        super().__init__()
        self.graph = graph
        self.hidden_dim = hidden_dim
        self.layers = layers
        
        # Calculate input dimensions from graph
        node_input_dim = sum(value.shape[-1] for value in graph.nodes.values())
        edge_input_dim = sum(value.shape[-1] for value in graph.edge_attr.values())
        self.input_dim = node_input_dim + edge_input_dim
        
        # MLP encoder layers
        mlp_layers = []
        current_dim = self.input_dim
        for _ in range(layers):
            mlp_layers.extend([nn.Linear(current_dim, hidden_dim), nn.Tanh()])
            current_dim = hidden_dim
        self.encoder = nn.Sequential(*mlp_layers)
        
        # Actor: edge logits
        self.edge_actor = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        
        # Actor: NOOP logit
        self.noop_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        
        # Critic: value estimate
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, graph: HeteroGraphState) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass through the fair PPO-MLP model."""
        # Flatten all node features
        node_features = []
        for node_type in ["uav", "region", "target"]:
            if node_type in graph.nodes:
                node_features.append(graph.nodes[node_type].reshape(-1))
        
        # Flatten all edge features
        edge_features = []
        for edge_type, edge_attr in graph.edge_attr.items():
            edge_features.append(edge_attr.reshape(-1))
        
        # Concatenate all features
        combined = torch.cat(node_features + edge_features, dim=0)
        
        # Encode through MLP
        encoded = self.encoder(combined)
        
        # Actor: compute edge logits
        pairs = graph.candidate_edges
        edge_logits = []
        for i in range(pairs.shape[0]):
            uav_idx, region_idx = int(pairs[i, 0]), int(pairs[i, 1])
            # Get edge features for this pair
            relation = ("uav", "can_serve", "region")
            if relation in graph.edge_attr:
                edge_feat = graph.edge_attr[relation][i]
            else:
                edge_feat = torch.zeros(5, device=graph.action_mask.device)
            
            # Concatenate encoded state with edge features
            edge_input = torch.cat([encoded, edge_feat], dim=0)
            edge_logit = self.edge_actor(edge_input)
            edge_logits.append(edge_logit)
        
        # NOOP logit
        noop_logit = self.noop_actor(encoded).reshape(1)
        
        # Combine all logits
        raw_logits = torch.cat(edge_logits + [noop_logit], dim=0)
        
        # Apply mask
        masked_logits = raw_logits.masked_fill(~graph.action_mask, torch.finfo(raw_logits.dtype).min)
        
        # Value estimate
        value = self.critic(encoded).squeeze(-1)
        
        # Diagnostics
        raw_probs = torch.softmax(raw_logits, dim=-1)
        invalid_mass = raw_probs[~graph.action_mask].sum()
        diagnostics = {
            "pre_mask_invalid_probability": invalid_mass,
            "gate_mean": {},  # No gates in MLP
        }
        
        return masked_logits, value, diagnostics
    
    def distribution(self, graph: HeteroGraphState) -> tuple[Categorical, torch.Tensor, dict]:
        logits, value, diagnostics = self(graph)
        return Categorical(logits=logits), value, diagnostics
    
    @torch.no_grad()
    def act(self, graph: HeteroGraphState, deterministic: bool = False) -> tuple[int, float, float, dict]:
        distribution, value, diagnostics = self.distribution(graph)
        action = torch.argmax(distribution.logits) if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        clean = {
            "pre_mask_invalid_probability": float(diagnostics["pre_mask_invalid_probability"].cpu()),
            "gate_mean": {},  # No gates in MLP
        }
        return int(action.item()), float(log_prob.item()), float(value.item()), clean
    
    def evaluate_action(self, graph: HeteroGraphState, action: torch.Tensor | int):
        distribution, value, diagnostics = self.distribution(graph)
        action_tensor = torch.as_tensor(action, dtype=torch.long, device=value.device)
        return distribution.log_prob(action_tensor), distribution.entropy(), value, diagnostics
    
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": "fair-ppo-mlp-v1",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }, path)
    
    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> tuple["FairPPOMLP", dict]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if payload.get("format") != "fair-ppo-mlp-v1":
            raise ValueError("unsupported fair-ppo-mlp checkpoint format")
        # Reconstruct the model with a dummy graph
        # In practice, the caller should provide the correct graph
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.input_dim = payload["input_dim"]
        model.hidden_dim = payload["hidden_dim"]
        model.layers = payload["layers"]
        model.load_state_dict(payload["state_dict"])
        return model, dict(payload.get("extra", {}))


def make_fair_ppo_mlp(graph: HeteroGraphState, hidden_dim: int = 64, layers: int = 2) -> FairPPOMLP:
    """Create a fair PPO-MLP model with the same inputs as GPPO."""
    return FairPPOMLP(graph, hidden_dim=hidden_dim, layers=layers)
