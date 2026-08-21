"""Small, dependency-free PPO trainer for the random-event graph policy.

The graph observations in this project have a fixed topology size today, but
they are deliberately kept as individual :class:`HeteroGraphState` objects in
the rollout buffer.  This avoids padding assumptions and lets the same trainer
work when later experiments add/remove Targets or communication edges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .graph import HeteroGraphState, build_graph_state
from .models import (
    FairPPOMLP,
    GraphActorCritic,
    make_adaptive_model,
    make_fair_ppo_mlp,
    make_no_gate_model,
)
from .environment import ActionSubmission, DecisionContext


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for discrete edge-action PPO."""

    rollout_steps: int = 512
    learning_rate: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    target_kl: float | None = None
    seed: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.rollout_steps <= 0 or self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("rollout_steps, update_epochs and minibatch_size must be positive")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must lie in [0, 1]")
        if self.clip_coef < 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("clip_coef must be non-negative and max_grad_norm positive")


@dataclass
class TrajectoryBuffer:
    """On-policy transitions plus GAE targets.

    ``next_values`` are stored per transition.  This is important for a time
    limit: truncation bootstraps from its final observation, while GAE still
    stops at the episode boundary and cannot leak into the next reset.
    """

    graphs: list[HeteroGraphState]
    actions: list[int]
    rewards: list[float]
    terminated: list[bool]
    truncated: list[bool]
    old_log_probs: list[float]
    old_values: list[float]
    next_values: list[float]
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    @classmethod
    def empty(cls) -> "TrajectoryBuffer":
        return cls([], [], [], [], [], [], [], [])

    def __len__(self) -> int:
        return len(self.actions)

    def add(
        self,
        graph: HeteroGraphState,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        log_prob: float,
        value: float,
        next_value: float,
    ) -> None:
        self.graphs.append(_copy_graph_to_cpu(graph))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.old_log_probs.append(float(log_prob))
        self.old_values.append(float(value))
        self.next_values.append(float(next_value))

    def compute_gae(self, gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
        if not self:
            raise ValueError("cannot compute GAE for an empty trajectory")
        advantages = np.zeros(len(self), dtype=np.float32)
        gae = 0.0
        for index in range(len(self) - 1, -1, -1):
            bootstrap = 0.0 if self.terminated[index] else 1.0
            episode_continues = 0.0 if (self.terminated[index] or self.truncated[index]) else 1.0
            delta = (
                self.rewards[index]
                + gamma * self.next_values[index] * bootstrap
                - self.old_values[index]
            )
            gae = delta + gamma * gae_lambda * episode_continues * gae
            advantages[index] = gae
        self.advantages = torch.from_numpy(advantages)
        self.returns = self.advantages + torch.tensor(self.old_values, dtype=torch.float32)
        return self.advantages, self.returns


def _copy_graph_to_cpu(graph: HeteroGraphState) -> HeteroGraphState:
    """Detach rollout data from both the live environment and autograd graph."""

    def copy_tensor(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to("cpu").clone()

    return HeteroGraphState(
        nodes={key: copy_tensor(value) for key, value in graph.nodes.items()},
        edge_index={key: copy_tensor(value) for key, value in graph.edge_index.items()},
        edge_attr={key: copy_tensor(value) for key, value in graph.edge_attr.items()},
        candidate_edges=copy_tensor(graph.candidate_edges),
        action_mask=copy_tensor(graph.action_mask),
        graph_version=int(graph.graph_version),
    )


def _extract_graph(observation: Any, info: Mapping[str, Any] | None, env: Any) -> HeteroGraphState:
    """Accept direct graph observations and the two common wrapper layouts."""

    if isinstance(observation, HeteroGraphState):
        return observation
    info = info or {}
    for source in (info, observation if isinstance(observation, Mapping) else {}):
        for key in ("graph", "graph_state", "hetero_graph"):
            value = source.get(key)
            if isinstance(value, HeteroGraphState):
                return value
    # This fallback makes the trainer usable with a thin wrapper around the
    # legacy environment without requiring that wrapper to duplicate graph.py.
    candidates = [env]
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and unwrapped is not env:
        candidates.append(unwrapped)
    for candidate in candidates:
        try:
            return build_graph_state(candidate)
        except (AttributeError, KeyError, TypeError):
            continue
    raise TypeError("reset/step did not expose a HeteroGraphState and graph construction failed")


def _reset_env(env: Any, seed: int | None = None) -> tuple[HeteroGraphState, dict[str, Any]]:
    try:
        result = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
    else:  # old Gym-style reset
        observation, info = result, {}
    info = dict(info or {})
    return _extract_graph(observation, info, env), info


def _step_env(env: Any, action: int) -> tuple[HeteroGraphState, float, bool, bool, dict[str, Any]]:
    result = env.step(int(action))
    if not isinstance(result, tuple):
        raise TypeError("env.step(action) must return a tuple")
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
    elif len(result) == 4:  # old Gym compatibility
        observation, reward, done, info = result
        info = dict(info or {})
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done) and not truncated
    else:
        raise TypeError("env.step(action) must return 4 or 5 values")
    info = dict(info or {})
    graph = _extract_graph(observation, info, env)
    return graph, float(reward), bool(terminated), bool(truncated), info


def _versioned_step_env(
    env: Any, action: int, ctx: "DecisionContext"
) -> tuple[HeteroGraphState, float, bool, bool, dict[str, Any]]:
    """Step using the versioned submission contract (begin_decision → submit_action).

    If the env supports ``begin_decision``/``submit_action`` (``RandomEventAllocationEnv``),
    this path is used; otherwise falls back to ``env.step``.
    """
    if hasattr(env, "submit_action") and isinstance(ctx, DecisionContext):
        submission = ActionSubmission.from_decision(action, ctx)
        result = env.submit_action(submission)
        if not isinstance(result, tuple):
            raise TypeError("env.submit_action(submission) must return a tuple")
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        elif len(result) == 4:
            observation, reward, done, info = result
            info = dict(info or {})
            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(done) and not truncated
        else:
            raise TypeError("env.submit_action must return 4 or 5 values")
        info = dict(info or {})
        graph = _extract_graph(observation, info, env)
        return graph, float(reward), bool(terminated), bool(truncated), info
    return _step_env(env, action)


def _explained_variance(prediction: np.ndarray, target: np.ndarray) -> float:
    target_variance = float(np.var(target))
    if target_variance <= 1e-12:
        return float("nan")
    return float(1.0 - np.var(target - prediction) / target_variance)


class PPOTrainer:
    """End-to-end PPO for ``PPO-MLP``, ``GPPO-NoGate`` and ``GPPO-Adaptive``.

    The model is initialized lazily from the first graph, so constructing a
    trainer does not reset the environment or consume an event tape.

    All three variants share the same environment, graph observation contract,
    action space, mask, reward, PPO hyperparameters, seeds, Validation/Test
    banks and checkpoint schedule.  Only the network architecture differs:
    - PPO-MLP       = canonical flattened graph -> MLP
    - GPPO-NoGate   = graph -> AHGNN (no adaptive gates)
    - GPPO-Adaptive = graph -> AHGNN + adaptive gates
    """

    def __init__(
        self,
        env: Any,
        variant: str = "GPPO-Adaptive",
        config: PPOConfig | None = None,
        model: GraphActorCritic | FairPPOMLP | None = None,
    ) -> None:
        self.env = env
        self.variant = self._normalise_variant(variant)
        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device) if model is not None else None
        if self.model is not None:
            expected_gate = self.variant == "GPPO-Adaptive"
            if isinstance(self.model, GraphActorCritic) and bool(self.model.config.adaptive_gate) != expected_gate:
                raise ValueError("variant and supplied model adaptive_gate setting disagree")
        self.optimizer: torch.optim.Optimizer | None = None
        self.history: list[dict[str, Any]] = []
        self.total_steps = 0
        self.update_count = 0
        self._current_graph: HeteroGraphState | None = None
        self._episode_return = 0.0
        self._episode_length = 0
        self._seed_everything(self.config.seed)
        if self.model is not None:
            self._make_optimizer()

    @staticmethod
    def _normalise_variant(value: str) -> str:
        compact = value.strip().lower().replace("_", "-")
        aliases = {
            "ppo-mlp": "PPO-MLP",
            "ppomlp": "PPO-MLP",
            "fair-ppo-mlp": "PPO-MLP",
            "gppo-adaptive": "GPPO-Adaptive",
            "adaptive": "GPPO-Adaptive",
            "gppo-nogate": "GPPO-NoGate",
            "gppo-no-gate": "GPPO-NoGate",
            "nogate": "GPPO-NoGate",
            "no-gate": "GPPO-NoGate",
        }
        if compact not in aliases:
            raise ValueError("variant must be PPO-MLP, GPPO-Adaptive or GPPO-NoGate")
        return aliases[compact]

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _make_optimizer(self) -> None:
        assert self.model is not None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def initialize(self, graph: HeteroGraphState) -> GraphActorCritic | FairPPOMLP:
        """Create the requested model variant after observing node dimensions."""

        if self.model is None:
            if self.variant == "PPO-MLP":
                self.model = make_fair_ppo_mlp(graph)
            elif self.variant == "GPPO-Adaptive":
                self.model = make_adaptive_model(graph)
            else:
                self.model = make_no_gate_model(graph)
            self.model.to(self.device)
            self._make_optimizer()
        return self.model

    def _ensure_current_graph(self) -> HeteroGraphState:
        if self._current_graph is None:
            self._current_graph, _ = _reset_env(self.env, seed=self.config.seed)
            self.initialize(self._current_graph)
        return self._current_graph

    def collect_rollout(self, steps: int | None = None) -> tuple[TrajectoryBuffer, dict[str, Any]]:
        """Collect an on-policy rollout and compute its GAE targets."""

        steps = int(steps or self.config.rollout_steps)
        if steps <= 0:
            raise ValueError("rollout size must be positive")
        graph = self._ensure_current_graph()
        model = self.initialize(graph)
        model.eval()
        buffer = TrajectoryBuffer.empty()
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        invalid_probabilities: list[float] = []
        gate_values: dict[str, list[float]] = {}

        for _ in range(steps):
            device_graph = graph.to(self.device)
            if not bool(device_graph.action_mask.any().item()):
                raise RuntimeError("graph action mask contains no legal action")
            action, log_prob, value, diagnostics = model.act(device_graph, deterministic=False)
            # Phase J: use versioned submission contract for RandomEventAllocationEnv
            ctx = None
            if hasattr(self.env, "begin_decision"):
                ctx = self.env.begin_decision()
            next_graph, reward, terminated, truncated, _ = (
                _versioned_step_env(self.env, action, ctx) if ctx is not None
                else _step_env(self.env, action)
            )
            with torch.no_grad():
                if terminated:
                    next_value = 0.0
                else:
                    _, next_value_tensor, _ = model(next_graph.to(self.device))
                    next_value = float(next_value_tensor.item())
            buffer.add(
                graph,
                action,
                reward,
                terminated,
                truncated,
                log_prob,
                value,
                next_value,
            )
            invalid_probabilities.append(float(diagnostics["pre_mask_invalid_probability"]))
            for key, gate_mean in diagnostics.get("gate_mean", {}).items():
                gate_values.setdefault(key, []).append(float(gate_mean))

            self.total_steps += 1
            self._episode_return += reward
            self._episode_length += 1
            if terminated or truncated:
                episode_returns.append(self._episode_return)
                episode_lengths.append(self._episode_length)
                self._episode_return = 0.0
                self._episode_length = 0
                graph, _ = _reset_env(self.env)
            else:
                graph = next_graph

        self._current_graph = graph
        buffer.compute_gae(self.config.gamma, self.config.gae_lambda)
        rollout_stats: dict[str, Any] = {
            "rollout_steps": len(buffer),
            "episodes_completed": len(episode_returns),
            "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else float("nan"),
            "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
            "pre_mask_invalid_probability": float(np.mean(invalid_probabilities)),
            "gate_means": {key: float(np.mean(values)) for key, values in sorted(gate_values.items())},
        }
        rollout_stats["gate_mean"] = (
            float(np.mean(list(rollout_stats["gate_means"].values())))
            if rollout_stats["gate_means"] else float("nan")
        )
        return buffer, rollout_stats

    def update(self, buffer: TrajectoryBuffer) -> dict[str, Any]:
        """Run clipped PPO updates over a completed on-policy buffer."""

        if buffer.advantages is None or buffer.returns is None:
            buffer.compute_gae(self.config.gamma, self.config.gae_lambda)
        if len(buffer) == 0:
            raise ValueError("cannot update from an empty trajectory")
        model = self.initialize(buffer.graphs[0])
        assert self.optimizer is not None
        model.train()

        advantages = buffer.advantages.to(self.device)
        returns = buffer.returns.to(self.device)
        old_log_probs = torch.tensor(buffer.old_log_probs, dtype=torch.float32, device=self.device)
        if self.config.normalize_advantages and len(buffer) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        metric_lists: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "pre_mask_invalid_probability": [],
            "grad_norm": [],
        }
        gate_lists: dict[str, list[float]] = {}
        stop_for_kl = False

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(len(buffer), device=self.device)
            for start in range(0, len(buffer), self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                log_probs: list[torch.Tensor] = []
                entropies: list[torch.Tensor] = []
                values: list[torch.Tensor] = []
                invalid_masses: list[torch.Tensor] = []
                minibatch_gates: dict[str, list[torch.Tensor]] = {}
                for index_tensor in indices:
                    index = int(index_tensor.item())
                    graph = buffer.graphs[index].to(self.device)
                    log_prob, entropy, value, diagnostics = model.evaluate_action(graph, buffer.actions[index])
                    log_probs.append(log_prob)
                    entropies.append(entropy)
                    values.append(value)
                    invalid_masses.append(diagnostics["pre_mask_invalid_probability"])
                    for key, gate in diagnostics.get("gates", {}).items():
                        minibatch_gates.setdefault(key, []).append(gate.mean())

                new_log_probs = torch.stack(log_probs)
                entropy_tensor = torch.stack(entropies)
                new_values = torch.stack(values).reshape(-1)
                log_ratio = new_log_probs - old_log_probs[indices]
                ratio = log_ratio.exp()
                minibatch_advantages = advantages[indices]
                policy_loss = torch.maximum(
                    -minibatch_advantages * ratio,
                    -minibatch_advantages * ratio.clamp(
                        1.0 - self.config.clip_coef,
                        1.0 + self.config.clip_coef,
                    ),
                ).mean()
                value_loss = 0.5 * (new_values - returns[indices]).pow(2).mean()
                entropy = entropy_tensor.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()
                metric_lists["policy_loss"].append(float(policy_loss.detach().cpu()))
                metric_lists["value_loss"].append(float(value_loss.detach().cpu()))
                metric_lists["entropy"].append(float(entropy.detach().cpu()))
                metric_lists["approx_kl"].append(float(approx_kl.cpu()))
                metric_lists["clip_fraction"].append(float(clip_fraction.cpu()))
                metric_lists["pre_mask_invalid_probability"].append(
                    float(torch.stack(invalid_masses).mean().detach().cpu())
                )
                metric_lists["grad_norm"].append(float(torch.as_tensor(grad_norm).detach().cpu()))
                for key, values_for_key in minibatch_gates.items():
                    gate_lists.setdefault(key, []).append(
                        float(torch.stack(values_for_key).mean().detach().cpu())
                    )

                if self.config.target_kl is not None and float(approx_kl) > self.config.target_kl:
                    stop_for_kl = True
                    break
            if stop_for_kl:
                break

        # Explained variance uses the final critic, not stale pre-update values.
        with torch.no_grad():
            final_values = np.asarray([
                float(model(graph.to(self.device))[1].cpu()) for graph in buffer.graphs
            ], dtype=np.float32)
        target_values = buffer.returns.numpy()
        stats: dict[str, Any] = {
            key: float(np.mean(values)) for key, values in metric_lists.items()
        }
        stats["explained_variance"] = _explained_variance(final_values, target_values)
        stats["gate_means"] = {
            key: float(np.mean(values)) for key, values in sorted(gate_lists.items())
        }
        stats["gate_mean"] = (
            float(np.mean(list(stats["gate_means"].values())))
            if stats["gate_means"] else float("nan")
        )
        stats["early_stop_kl"] = stop_for_kl
        return stats

    def train(self, total_timesteps: int) -> list[dict[str, Any]]:
        """Train until at least ``total_timesteps`` have been collected."""

        target = int(total_timesteps)
        if target <= 0:
            raise ValueError("total_timesteps must be positive")
        starting_steps = self.total_steps
        while self.total_steps - starting_steps < target:
            remaining = target - (self.total_steps - starting_steps)
            buffer, rollout_stats = self.collect_rollout(min(self.config.rollout_steps, remaining))
            update_stats = self.update(buffer)
            self.update_count += 1
            record = {
                "update": self.update_count,
                "total_steps": self.total_steps,
                "variant": self.variant,
                **rollout_stats,
                **update_stats,
            }
            # The update diagnostics are authoritative for these two fields;
            # retain collection-time values under explicit rollout names.
            record["rollout_pre_mask_invalid_probability"] = rollout_stats[
                "pre_mask_invalid_probability"
            ]
            record["rollout_gate_means"] = rollout_stats["gate_means"]
            self.history.append(record)
        return self.history

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        """Delegate checkpoint serialization to :meth:`GraphActorCritic.save`."""

        if self.model is None:
            raise RuntimeError("cannot save before the model has been initialized")
        metadata = {
            "variant": self.variant,
            "ppo_config": asdict(self.config),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
            "history": self.history,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
        }
        if extra:
            metadata.update(extra)
        self.model.save(path, extra=metadata)

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: Any,
        device: str | None = None,
    ) -> tuple["PPOTrainer", dict[str, Any]]:
        """Restore model and trainer metadata from a model-owned checkpoint."""

        import torch as _torch
        payload = _torch.load(Path(path), map_location=device or "cpu", weights_only=False)
        fmt = payload.get("format")
        if fmt == "fair-ppo-mlp-v1" or fmt == "fair-ppo-mlp-v2":
            model, metadata = FairPPOMLP.load(path, map_location=device or "cpu")
        elif fmt == "random-event-gppo-v1":
            model, metadata = GraphActorCritic.load(path, map_location=device or "cpu")
        else:
            raise ValueError(f"unsupported checkpoint format: {fmt}")
        config_values = dict(metadata.get("ppo_config", {}))
        if device is not None:
            config_values["device"] = device
        config = PPOConfig(**config_values) if config_values else PPOConfig(device=device or "cpu")
        variant = metadata.get("variant", cls._normalise_variant_from_model(model))
        trainer = cls(env=env, variant=variant, config=config, model=model)
        trainer.total_steps = int(metadata.get("total_steps", 0))
        trainer.update_count = int(metadata.get("update_count", 0))
        trainer.history = list(metadata.get("history", []))
        optimizer_state = metadata.get("optimizer_state")
        if optimizer_state is not None and trainer.optimizer is not None:
            trainer.optimizer.load_state_dict(optimizer_state)
        return trainer, metadata

    @staticmethod
    def _normalise_variant_from_model(model: Any) -> str:
        if isinstance(model, FairPPOMLP):
            return "PPO-MLP"
        if isinstance(model, GraphActorCritic):
            return "GPPO-Adaptive" if model.config.adaptive_gate else "GPPO-NoGate"
        return "GPPO-Adaptive"


__all__ = ["PPOConfig", "PPOTrainer", "TrajectoryBuffer"]
