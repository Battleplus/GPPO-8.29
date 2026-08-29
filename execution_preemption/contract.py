"""Load and validate the frozen Execution-Preemption V1 training contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .metrics import METRICS_SCHEMA_ID
from .reward import REWARD_CONTRACT_ID, REWARD_WEIGHTS


TRAINING_CONTRACT_ID = "execution-preemption-training-v1"
TRAINING_SEEDS = (1101, 2202, 3303)
TRAINING_STEPS_PER_RUN = 50_000
CHECKPOINT_STEPS = (25_000, 50_000)
FIXED_EVALUATION_CHECKPOINT = 50_000
TRAINING_SCALES = (4, 8, 16)
ZERO_SHOT_SCALE = 32

EXPECTED_COMPARISON_METHODS = (
    "senior_legacy_method_v1",
    "greedy_priority_v1",
    "ppo_mlp_reactive_v1",
    "gppo_adaptive_reactive_v1",
    "beam_mpc_v1",
    "ppo_mlp_rule_arbiter_v1",
    "gppo_adaptive_rule_arbiter_v1",
)

LEARNED_METHODS = (
    "ppo_mlp_reactive_v1",
    "gppo_adaptive_reactive_v1",
    "ppo_mlp_rule_arbiter_v1",
    "gppo_adaptive_rule_arbiter_v1",
)


class TrainingContractError(ValueError):
    """Raised when a configuration drifts from the frozen V1 contract."""


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ValidatedTrainingContract:
    path: Path
    canonical_sha256: str
    contract_id: str
    status: str
    training_allowed: bool
    learned_run_count: int
    checkpoint_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "canonical_sha256": self.canonical_sha256,
            "contract_id": self.contract_id,
            "status": self.status,
            "training_allowed": self.training_allowed,
            "learned_run_count": self.learned_run_count,
            "checkpoint_count": self.checkpoint_count,
        }


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingContractError(message)


def validate_training_contract(
    value: Mapping[str, Any],
    *,
    path: str | Path = "<memory>",
) -> ValidatedTrainingContract:
    _expect(value.get("contract_id") == TRAINING_CONTRACT_ID, "contract_id drift")
    _expect(value.get("schema_version") == 1, "schema_version must equal 1")
    _expect(value.get("status") == "FROZEN_FOR_ADAPTER_IMPLEMENTATION", "status drift")

    compatibility = value.get("compatibility", {})
    _expect(compatibility.get("legacy_checkpoint_compatible") is False,
            "legacy checkpoints must be incompatible")
    _expect(compatibility.get("legacy_evidence_reusable") is False,
            "legacy evidence must not be reusable")
    _expect(compatibility.get("modifies_legacy_reward") is False,
            "the new contract must not modify legacy reward")

    reward = value.get("reward", {})
    _expect(reward.get("contract_id") == REWARD_CONTRACT_ID, "reward contract drift")
    _expect(reward.get("hard_constraints_in_reward") is False,
            "hard constraints must stay outside scalar reward")
    configured_weights = reward.get("weights", {})
    _expect(configured_weights == dict(REWARD_WEIGHTS), "reward weights drift")

    metrics = value.get("metrics", {})
    _expect(metrics.get("schema_id") == METRICS_SCHEMA_ID, "metrics schema drift")
    required_metrics = set(metrics.get("required", ()))
    _expect({
        "urgent_deadline_miss_rate",
        "mean_preemption_response_latency",
        "cumulative_weighted_vacancy",
        "normal_task_recovery_rate",
        "cumulative_progress_loss",
        "energy_safety_violations",
        "resource_conflicts",
        "stale_command_resurrections",
        "task_starvation_rate",
        "cumulative_normalized_distance",
        "mean_load_gap",
        "inference_latency_p95_ms",
        "inference_latency_p99_ms",
    }.issubset(required_metrics), "required metrics are incomplete")

    comparison = value.get("comparison", {})
    methods = tuple(comparison.get("methods", ()))
    _expect(methods == EXPECTED_COMPARISON_METHODS, "comparison method order/content drift")
    _expect(tuple(comparison.get("learned_methods", ())) == LEARNED_METHODS,
            "learned method set drift")
    _expect(comparison.get("safety_shell_mandatory_for_all_methods") is True,
            "safety shell must be mandatory")

    training = value.get("training", {})
    _expect(tuple(training.get("seeds", ())) == TRAINING_SEEDS, "training seeds drift")
    _expect(training.get("accepted_decision_steps_per_run") == TRAINING_STEPS_PER_RUN,
            "training budget drift")
    _expect(tuple(training.get("checkpoint_steps", ())) == CHECKPOINT_STEPS,
            "checkpoint grid drift")
    _expect(training.get("fixed_evaluation_checkpoint") == FIXED_EVALUATION_CHECKPOINT,
            "fixed evaluation checkpoint drift")
    _expect(training.get("checkpoint_selection") is False,
            "checkpoint selection must be disabled")
    _expect(tuple(training.get("training_uav_scales", ())) == TRAINING_SCALES,
            "training scale sequence drift")
    _expect(training.get("zero_shot_scalability_uav_count") == ZERO_SHOT_SCALE,
            "zero-shot scale drift")
    _expect(training.get("zero_shot_checkpoint_source_uav_count") == TRAINING_SCALES[-1],
            "zero-shot checkpoint source scale drift")
    _expect(training.get("reuse_old_campaign") is False, "old campaign reuse is forbidden")
    _expect(training.get("resume_old_checkpoint") is False, "old checkpoint resume is forbidden")

    seed_partition = value.get("seed_partition", {})
    _expect(seed_partition.get("identity_rule") == "contract_id/namespace/integer",
            "seed identity rule drift")
    _expect(seed_partition.get("development_hidden_overlap_allowed") is False,
            "development/hidden overlap must be forbidden")
    hidden = seed_partition.get("hidden", {})
    _expect(hidden.get("status") == "NOT_GENERATED", "hidden bank must not exist yet")
    _expect(hidden.get("seeds") == [], "hidden seeds must remain empty before freeze")

    scale_gate = value.get("scale_gate", {})
    _expect(tuple(scale_gate.get("order", ())) == (4, 8, 16, 32), "scale gate order drift")
    _expect(scale_gate.get("advance_only_after_previous_pass") is True,
            "scale advancement must be gated")

    launch_gate = value.get("launch_gate", {})
    _expect(launch_gate.get("training_allowed") is False,
            "training must remain disabled during contract freeze")
    _expect(launch_gate.get("requires_source_bound_gate") is True,
            "source-bound launch gate is required")
    _expect(launch_gate.get("requires_clean_training_worktree") is True,
            "clean training worktree is required")

    learned_run_count = len(LEARNED_METHODS) * len(TRAINING_SEEDS) * len(TRAINING_SCALES)
    checkpoint_count = learned_run_count * len(CHECKPOINT_STEPS)
    declared = value.get("planned_cardinality", {})
    _expect(declared.get("learned_run_count") == learned_run_count,
            "planned learned run count drift")
    _expect(declared.get("checkpoint_count") == checkpoint_count,
            "planned checkpoint count drift")

    return ValidatedTrainingContract(
        path=Path(path),
        canonical_sha256=canonical_json_sha256(value),
        contract_id=TRAINING_CONTRACT_ID,
        status=str(value["status"]),
        training_allowed=False,
        learned_run_count=learned_run_count,
        checkpoint_count=checkpoint_count,
    )


def load_training_contract(path: str | Path) -> ValidatedTrainingContract:
    resolved = Path(path).resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TrainingContractError("training contract root must be an object")
    return validate_training_contract(value, path=resolved)


__all__ = [
    "CHECKPOINT_STEPS",
    "EXPECTED_COMPARISON_METHODS",
    "FIXED_EVALUATION_CHECKPOINT",
    "LEARNED_METHODS",
    "TRAINING_CONTRACT_ID",
    "TRAINING_SCALES",
    "TRAINING_SEEDS",
    "TRAINING_STEPS_PER_RUN",
    "TrainingContractError",
    "ValidatedTrainingContract",
    "ZERO_SHOT_SCALE",
    "canonical_json_sha256",
    "load_training_contract",
    "validate_training_contract",
]
