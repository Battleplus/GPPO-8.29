"""PPO training runner for the frozen Execution-Preemption V1 contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .contract import (
    CHECKPOINT_STEPS,
    LEARNED_METHODS,
    TRAINING_SCALES,
    TRAINING_SEEDS,
    TRAINING_STEPS_PER_RUN,
    load_training_contract,
)
from .framework import TorchFlatObservation, TorchHeteroObservation, flat_to_torch, hetero_to_torch
from .gate import _check_execution_launch_gate
from .gym_env import ExecutionPreemptionGymEnv
from .policy_models import ExecutionGPPOAdaptive, ExecutionPPOMLP
from .training_tapes import build_training_tape


PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_VALUE_COEFFICIENT = 0.5
PPO_ENTROPY_COEFFICIENT = 0.01
PPO_MAX_GRAD_NORM = 0.5
PPO_ROLLOUT_STEPS = 64
PPO_UPDATE_EPOCHS = 4
MODEL_HIDDEN_DIM = 64
GPPO_RELATION_LAYERS = 2


@dataclass(frozen=True, slots=True)
class MethodSpec:
    method_id: str
    model_family: str
    expose_rule_context: bool


METHOD_SPECS: dict[str, MethodSpec] = {
    "ppo_mlp_reactive_v1": MethodSpec(
        "ppo_mlp_reactive_v1", "ppo_mlp", False
    ),
    "gppo_adaptive_reactive_v1": MethodSpec(
        "gppo_adaptive_reactive_v1", "gppo_adaptive", False
    ),
    "ppo_mlp_rule_arbiter_v1": MethodSpec(
        "ppo_mlp_rule_arbiter_v1", "ppo_mlp", True
    ),
    "gppo_adaptive_rule_arbiter_v1": MethodSpec(
        "gppo_adaptive_rule_arbiter_v1", "gppo_adaptive", True
    ),
}


@dataclass(frozen=True, slots=True)
class TrainingRunConfig:
    method_id: str
    policy_seed: int
    uav_count: int
    accepted_decision_steps: int = TRAINING_STEPS_PER_RUN
    checkpoint_steps: tuple[int, ...] = CHECKPOINT_STEPS
    rollout_steps: int = PPO_ROLLOUT_STEPS
    update_epochs: int = PPO_UPDATE_EPOCHS
    learning_rate: float = PPO_LEARNING_RATE
    hidden_dim: int = MODEL_HIDDEN_DIM
    relation_layers: int = GPPO_RELATION_LAYERS
    formal: bool = True

    @property
    def method(self) -> MethodSpec:
        try:
            return METHOD_SPECS[self.method_id]
        except KeyError as exc:
            raise ValueError(f"unknown learned method {self.method_id}") from exc

    def validate(self) -> None:
        _ = self.method
        if self.policy_seed not in TRAINING_SEEDS:
            raise ValueError("policy seed drift")
        if self.uav_count not in TRAINING_SCALES:
            raise ValueError("training scale drift")
        if self.accepted_decision_steps <= 0:
            raise ValueError("accepted decision budget must be positive")
        if not self.checkpoint_steps:
            raise ValueError("checkpoint grid must not be empty")
        if tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps:
            raise ValueError("checkpoint grid must be sorted and unique")
        if self.checkpoint_steps[-1] != self.accepted_decision_steps:
            raise ValueError("final checkpoint must equal the accepted decision budget")
        if self.rollout_steps <= 0 or self.update_epochs <= 0:
            raise ValueError("PPO rollout and update counts must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning rate must be finite and positive")
        if self.hidden_dim <= 0 or self.relation_layers <= 0:
            raise ValueError("model dimensions must be positive")
        if self.formal:
            if self.accepted_decision_steps != TRAINING_STEPS_PER_RUN:
                raise ValueError("formal training budget drift")
            if self.checkpoint_steps != CHECKPOINT_STEPS:
                raise ValueError("formal checkpoint grid drift")
            if self.rollout_steps != PPO_ROLLOUT_STEPS:
                raise ValueError("formal rollout size drift")
            if self.update_epochs != PPO_UPDATE_EPOCHS:
                raise ValueError("formal update epoch drift")
            if self.learning_rate != PPO_LEARNING_RATE:
                raise ValueError("formal learning rate drift")
            if self.hidden_dim != MODEL_HIDDEN_DIM:
                raise ValueError("formal hidden dimension drift")
            if self.relation_layers != GPPO_RELATION_LAYERS:
                raise ValueError("formal GPPO layer drift")


PolicyObservation = TorchFlatObservation | TorchHeteroObservation


@dataclass(slots=True)
class _Transition:
    observation: PolicyObservation
    action: int
    old_log_probability: float
    reward: float
    value: float
    done: bool
    tape_id: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rng_state_sha256(state: Mapping[str, Any]) -> str:
    python_state = state["python"]
    numpy_state = state["numpy"]
    torch_state = state["torch"].detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(repr(python_state).encode("utf-8"))
    digest.update(str(numpy_state[0]).encode("ascii"))
    digest.update(np.asarray(numpy_state[1], dtype=np.uint32).tobytes())
    digest.update(str(numpy_state[2:]).encode("ascii"))
    digest.update(torch_state.numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _append_json_line(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fresh_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"training output must be new and empty: {path}")
    else:
        path.mkdir(parents=True)


def build_model(config: TrainingRunConfig) -> nn.Module:
    if config.method.model_family == "ppo_mlp":
        return ExecutionPPOMLP(hidden_dim=config.hidden_dim)
    return ExecutionGPPOAdaptive(
        hidden_dim=config.hidden_dim,
        layers=config.relation_layers,
    )


def formal_run_relative_path(method_id: str, policy_seed: int, uav_count: int) -> Path:
    config = TrainingRunConfig(
        method_id=method_id,
        policy_seed=policy_seed,
        uav_count=uav_count,
    )
    config.validate()
    return Path(method_id) / f"uav_{uav_count:02d}" / f"seed_{policy_seed}"


def planned_formal_runs() -> tuple[TrainingRunConfig, ...]:
    runs = tuple(
        TrainingRunConfig(method_id=method, policy_seed=seed, uav_count=scale)
        for scale in TRAINING_SCALES
        for seed in TRAINING_SEEDS
        for method in LEARNED_METHODS
    )
    paths = {
        formal_run_relative_path(item.method_id, item.policy_seed, item.uav_count).as_posix()
        for item in runs
    }
    if len(runs) != 36 or len(paths) != len(runs):
        raise RuntimeError("formal training cardinality or worker path collision")
    return runs


def _observation(info: Mapping[str, Any], family: str) -> PolicyObservation:
    if family == "ppo_mlp":
        return flat_to_torch(info["flat_observation"])
    return hetero_to_torch(info["hetero_observation"])


def _forward(policy: nn.Module, observation: PolicyObservation):
    return policy(observation)


def _sample_action(policy: nn.Module, observation: PolicyObservation) -> tuple[int, float, float]:
    with torch.no_grad():
        logits, value, _ = _forward(policy, observation)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return (
            int(action.item()),
            float(distribution.log_prob(action).item()),
            float(value.item()),
        )


def _state_value(policy: nn.Module, observation: PolicyObservation) -> float:
    with torch.no_grad():
        _, value, _ = _forward(policy, observation)
    return float(value.item())


def _returns_and_advantages(
    transitions: Sequence[_Transition],
    bootstrap_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = [0.0] * len(transitions)
    next_advantage = 0.0
    next_value = float(bootstrap_value)
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        continuation = 0.0 if transition.done else 1.0
        delta = transition.reward + PPO_GAMMA * next_value * continuation - transition.value
        next_advantage = delta + PPO_GAMMA * PPO_GAE_LAMBDA * continuation * next_advantage
        advantages[index] = next_advantage
        next_value = transition.value
    advantage_tensor = torch.tensor(advantages, dtype=torch.float32)
    returns = advantage_tensor + torch.tensor(
        [item.value for item in transitions], dtype=torch.float32
    )
    if advantage_tensor.numel() > 1:
        standard_deviation = advantage_tensor.std(unbiased=False)
        if float(standard_deviation.item()) > 1e-8:
            advantage_tensor = (
                advantage_tensor - advantage_tensor.mean()
            ) / (standard_deviation + 1e-8)
    return returns, advantage_tensor


def _ppo_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[_Transition],
    *,
    bootstrap_value: float,
    epochs: int,
) -> dict[str, float]:
    returns, advantages = _returns_and_advantages(transitions, bootstrap_value)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_values = {key: 0.0 for key in totals}
        for index, transition in enumerate(transitions):
            logits, value, _ = _forward(policy, transition.observation)
            distribution = Categorical(logits=logits)
            action = torch.tensor(transition.action, dtype=torch.long, device=logits.device)
            log_probability = distribution.log_prob(action)
            ratio = torch.exp(log_probability - transition.old_log_probability)
            advantage = advantages[index].to(logits.device)
            unclipped = ratio * advantage
            clipped = torch.clamp(
                ratio, 1.0 - PPO_CLIP_RANGE, 1.0 + PPO_CLIP_RANGE
            ) * advantage
            policy_loss = -torch.minimum(unclipped, clipped)
            value_loss = (value - returns[index].to(value.device)).pow(2)
            entropy = distribution.entropy()
            loss = (
                policy_loss
                + PPO_VALUE_COEFFICIENT * value_loss
                - PPO_ENTROPY_COEFFICIENT * entropy
            ) / len(transitions)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite PPO loss")
            loss.backward()
            epoch_values["policy_loss"] += float(policy_loss.detach().item())
            epoch_values["value_loss"] += float(value_loss.detach().item())
            epoch_values["entropy"] += float(entropy.detach().item())
        gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), PPO_MAX_GRAD_NORM)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite PPO gradient")
        optimizer.step()
        for key in totals:
            totals[key] += epoch_values[key] / len(transitions)
    return {key: value / epochs for key, value in totals.items()}


def _rng_record() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def _save_checkpoint(
    path: Path,
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainingRunConfig,
    step: int,
    episode_index: int,
    optimizer_step_count: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    rng_state = _rng_record()
    model_hash = model_state_sha256(policy.state_dict())
    rng_hash = rng_state_sha256(rng_state)
    payload = {
        "schema_version": 1,
        "contract_id": "execution-preemption-training-v1",
        "method_id": config.method_id,
        "policy_seed": config.policy_seed,
        "uav_count": config.uav_count,
        "accepted_decision_step": step,
        "episode_index": episode_index,
        "optimizer_step_count": optimizer_step_count,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": rng_state,
        "model_state_sha256": model_hash,
        "rng_state_sha256": rng_hash,
        "provenance": dict(provenance),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary, _use_new_zipfile_serialization=False)
    os.replace(temporary, path)
    return {
        "step": step,
        "path": path.name,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "model_state_sha256": model_hash,
        "rng_state_sha256": rng_hash,
    }


def verify_checkpoint(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
    expected_method_id: str | None = None,
    expected_policy_seed: int | None = None,
    expected_uav_count: int | None = None,
    expected_step: int | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    file_hash = _sha256_file(path)
    if expected_file_sha256 is not None and file_hash != expected_file_sha256:
        raise ValueError("checkpoint file SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    expected = {
        "method_id": expected_method_id,
        "policy_seed": expected_policy_seed,
        "uav_count": expected_uav_count,
        "accepted_decision_step": expected_step,
    }
    for key, value in expected.items():
        if value is not None and payload.get(key) != value:
            raise ValueError(f"checkpoint {key} mismatch")
    model_hash = model_state_sha256(payload["model_state_dict"])
    rng_hash = rng_state_sha256(payload["rng_state"])
    if model_hash != payload.get("model_state_sha256"):
        raise ValueError("checkpoint model-state SHA-256 mismatch")
    if rng_hash != payload.get("rng_state_sha256"):
        raise ValueError("checkpoint RNG-state SHA-256 mismatch")
    if not isinstance(payload.get("optimizer_state_dict"), Mapping):
        raise ValueError("checkpoint optimizer state missing")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint provenance missing")
    return {
        "status": "PASS",
        "file_sha256": file_hash,
        "model_state_sha256": model_hash,
        "rng_state_sha256": rng_hash,
        "method_id": payload.get("method_id"),
        "policy_seed": payload.get("policy_seed"),
        "uav_count": payload.get("uav_count"),
        "accepted_decision_step": payload.get("accepted_decision_step"),
        "optimizer_step_count": payload.get("optimizer_step_count"),
        "provenance": dict(provenance),
    }


def _build_file_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "sha256_inventory.json"
    ]


def verify_training_run(
    output_dir: Path,
    *,
    expected_config: TrainingRunConfig,
) -> dict[str, Any]:
    expected_config.validate()
    output = output_dir.resolve()
    manifest_path = output / "run_manifest.json"
    progress_path = output / "progress.json"
    log_path = output / "updates.jsonl"
    inventory_path = output / "sha256_inventory.json"
    for path in (manifest_path, progress_path, log_path, inventory_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or progress.get("status") != "COMPLETE":
        raise ValueError("training run is not complete")
    expected_values = {
        "method_id": expected_config.method_id,
        "policy_seed": expected_config.policy_seed,
        "uav_count": expected_config.uav_count,
        "accepted_decision_steps": expected_config.accepted_decision_steps,
        "checkpoint_steps": list(expected_config.checkpoint_steps),
        "formal": expected_config.formal,
    }
    for key, value in expected_values.items():
        if manifest.get(key) != value:
            raise ValueError(f"run manifest {key} mismatch")
    if manifest.get("checkpoint_selection") is not False:
        raise ValueError("checkpoint selection must be false")
    if any(manifest.get(key) is not False for key in (
        "validation_started", "freeze_started", "test_started", "hidden_evaluation_started"
    )):
        raise ValueError("post-training evaluation phase was started")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("run provenance missing")
    if provenance.get("legacy_checkpoint_loaded") is not False:
        raise ValueError("legacy checkpoint reuse detected")
    if provenance.get("old_campaign_reused") is not False:
        raise ValueError("old campaign reuse detected")

    actual_files = _build_file_inventory(output)
    declared_files = inventory.get("files")
    if inventory.get("status") != "PASS" or declared_files != actual_files:
        raise ValueError("training file SHA-256 inventory mismatch")
    updates = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not updates:
        raise ValueError("training update log is empty")
    steps = [int(item["accepted_decision_steps"]) for item in updates]
    if steps != sorted(set(steps)) or steps[-1] != expected_config.accepted_decision_steps:
        raise ValueError("training update steps are not strictly monotonic and exact")
    if progress.get("accepted_decision_steps") != expected_config.accepted_decision_steps:
        raise ValueError("progress step mismatch")
    if progress.get("checkpoint_steps_written") != list(expected_config.checkpoint_steps):
        raise ValueError("progress checkpoint grid mismatch")

    checkpoint_records = manifest.get("checkpoints")
    if not isinstance(checkpoint_records, list) or len(checkpoint_records) != len(
        expected_config.checkpoint_steps
    ):
        raise ValueError("checkpoint cardinality mismatch")
    verified = []
    for record, step in zip(checkpoint_records, expected_config.checkpoint_steps):
        if record.get("step") != step:
            raise ValueError("checkpoint step order mismatch")
        verified.append(verify_checkpoint(
            output / "checkpoints" / str(record["path"]),
            expected_file_sha256=str(record["sha256"]),
            expected_method_id=expected_config.method_id,
            expected_policy_seed=expected_config.policy_seed,
            expected_uav_count=expected_config.uav_count,
            expected_step=step,
        ))
    return {
        "status": "PASS",
        "method_id": expected_config.method_id,
        "policy_seed": expected_config.policy_seed,
        "uav_count": expected_config.uav_count,
        "accepted_decision_steps": expected_config.accepted_decision_steps,
        "optimizer_step_count": manifest.get("optimizer_step_count"),
        "checkpoint_count": len(verified),
        "checkpoint_steps": list(expected_config.checkpoint_steps),
        "checkpoint_sha256_verified": True,
        "inventory_file_count": len(actual_files),
        "provenance": dict(provenance),
    }


def train_run(config: TrainingRunConfig, output_dir: Path) -> dict[str, Any]:
    """Train one fresh run; formal mode checks the Gate before any write/optimizer."""
    config.validate()
    root = _repo_root()
    gate_record: Mapping[str, Any] | None = None
    if config.formal:
        gate_record = _check_execution_launch_gate(require_fully_clean=True)
    output = output_dir.resolve()
    _fresh_output(output)

    random.seed(config.policy_seed)
    np.random.seed(config.policy_seed)
    torch.manual_seed(config.policy_seed)
    torch.use_deterministic_algorithms(True)
    policy = build_model(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    contract = load_training_contract(root / "configs" / "execution_training_contract_v1.json")
    provenance = {
        "runtime_head_sha": _git(root, "rev-parse", "HEAD"),
        "runtime_tree_sha": _git(root, "rev-parse", "HEAD^{tree}"),
        "training_contract_sha256": contract.canonical_sha256,
        "training_namespace": "execution_preemption_v1/train",
        "gate": gate_record,
        "legacy_checkpoint_loaded": False,
        "old_campaign_reused": False,
        "checkpoint_selection": False,
    }

    progress_path = output / "progress.json"
    log_path = output / "updates.jsonl"
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    accepted_steps = 0
    episode_index = 0
    optimizer_step_count = 0
    checkpoints: list[dict[str, Any]] = []
    checkpoint_targets = set(config.checkpoint_steps)
    next_info: Mapping[str, Any] | None = None
    env: ExecutionPreemptionGymEnv | None = None

    _write_json(progress_path, {
        "status": "RUNNING",
        "accepted_decision_steps": 0,
        "target_steps": config.accepted_decision_steps,
        "optimizer_step_count": 0,
        "formal": config.formal,
    })
    while accepted_steps < config.accepted_decision_steps:
        remaining_to_checkpoint = min(
            target - accepted_steps
            for target in config.checkpoint_steps
            if target > accepted_steps
        )
        rollout_target = min(
            config.rollout_steps,
            remaining_to_checkpoint,
            config.accepted_decision_steps - accepted_steps,
        )
        transitions: list[_Transition] = []
        while len(transitions) < rollout_target:
            if next_info is None or bool(next_info.get("episode_terminated", False)):
                tape = build_training_tape(
                    policy_seed=config.policy_seed,
                    uav_count=config.uav_count,
                    episode_index=episode_index,
                )
                episode_index += 1
                env = ExecutionPreemptionGymEnv(
                    tape,
                    allocator_id=config.method_id,
                    expose_rule_context=config.method.expose_rule_context,
                )
                _, next_info = env.reset(seed=tape["case_seed"])
                if bool(next_info["episode_terminated"]):
                    continue
            assert env is not None and next_info is not None
            observation = _observation(next_info, config.method.model_family)
            action, old_log_probability, value = _sample_action(policy, observation)
            if not bool(env.action_masks()[action]):
                raise RuntimeError("policy selected a masked action")
            _, reward, terminated, truncated, after_info = env.step(action)
            if truncated:
                raise RuntimeError("training tape unexpectedly truncated")
            transitions.append(_Transition(
                observation=observation,
                action=action,
                old_log_probability=old_log_probability,
                reward=float(reward),
                value=value,
                done=bool(terminated),
                tape_id=str(next_info["tape_id"]),
            ))
            next_info = after_info

        bootstrap = 0.0
        if next_info is not None and not bool(next_info.get("episode_terminated", False)):
            bootstrap = _state_value(
                policy, _observation(next_info, config.method.model_family)
            )
        losses = _ppo_update(
            policy,
            optimizer,
            transitions,
            bootstrap_value=bootstrap,
            epochs=config.update_epochs,
        )
        optimizer_step_count += config.update_epochs
        accepted_steps += len(transitions)
        reward_sum = float(sum(item.reward for item in transitions))
        update_record = {
            "accepted_decision_steps": accepted_steps,
            "rollout_decision_count": len(transitions),
            "episode_index": episode_index,
            "optimizer_step_count": optimizer_step_count,
            "reward_sum": reward_sum,
            **losses,
        }
        _append_json_line(log_path, update_record)
        if accepted_steps in checkpoint_targets:
            checkpoint_path = checkpoint_dir / f"step_{accepted_steps:06d}.pt"
            checkpoints.append(_save_checkpoint(
                checkpoint_path,
                policy=policy,
                optimizer=optimizer,
                config=config,
                step=accepted_steps,
                episode_index=episode_index,
                optimizer_step_count=optimizer_step_count,
                provenance=provenance,
            ))
        _write_json(progress_path, {
            "status": "COMPLETE" if accepted_steps == config.accepted_decision_steps else "RUNNING",
            "accepted_decision_steps": accepted_steps,
            "target_steps": config.accepted_decision_steps,
            "episode_index": episode_index,
            "optimizer_step_count": optimizer_step_count,
            "checkpoint_steps_written": [item["step"] for item in checkpoints],
            "formal": config.formal,
        })

    checkpoint_steps = tuple(int(item["step"]) for item in checkpoints)
    if checkpoint_steps != config.checkpoint_steps:
        raise RuntimeError("checkpoint cardinality/grid mismatch")
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "classification": (
            "formal_execution_preemption_training_run"
            if config.formal else "tiny_training_runner_smoke_not_model_evidence"
        ),
        "method_id": config.method_id,
        "model_family": config.method.model_family,
        "rule_context_exposed": config.method.expose_rule_context,
        "policy_seed": config.policy_seed,
        "uav_count": config.uav_count,
        "accepted_decision_steps": accepted_steps,
        "episode_count": episode_index,
        "optimizer_step_count": optimizer_step_count,
        "checkpoint_steps": list(checkpoint_steps),
        "checkpoints": checkpoints,
        "provenance": provenance,
        "formal": config.formal,
        "validation_started": False,
        "freeze_started": False,
        "test_started": False,
        "hidden_evaluation_started": False,
        "checkpoint_selection": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    _write_json(output / "sha256_inventory.json", {
        "schema_version": 1,
        "status": "PASS",
        "files": _build_file_inventory(output),
    })
    return manifest


__all__ = [
    "GPPO_RELATION_LAYERS",
    "METHOD_SPECS",
    "MODEL_HIDDEN_DIM",
    "PPO_CLIP_RANGE",
    "PPO_ENTROPY_COEFFICIENT",
    "PPO_GAE_LAMBDA",
    "PPO_GAMMA",
    "PPO_LEARNING_RATE",
    "PPO_MAX_GRAD_NORM",
    "PPO_ROLLOUT_STEPS",
    "PPO_UPDATE_EPOCHS",
    "PPO_VALUE_COEFFICIENT",
    "MethodSpec",
    "TrainingRunConfig",
    "build_model",
    "formal_run_relative_path",
    "model_state_sha256",
    "rng_state_sha256",
    "planned_formal_runs",
    "train_run",
    "verify_checkpoint",
    "verify_training_run",
]
