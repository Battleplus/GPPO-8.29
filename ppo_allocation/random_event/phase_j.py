"""Phase J: Preliminary Training Orchestrator.

This module implements the full Phase J workflow:
1. preliminary-train: 300k steps with periodic checkpoints
2. preliminary-validate: lexicographic checkpoint selection PER (variant, seed)
3. preliminary-freeze: freeze 9 selected checkpoints with SHA attestation
4. preliminary-test: run Test bank on all 9 frozen checkpoints (once each)

All operations respect the frozen protocol and refuse to proceed if
the P0 gate is not green or if source/protocol hashes have drifted.

Resume is NOT supported for formal Preliminary training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .environment import RandomEventAllocationEnv
from .experiment import (
    _json_file,
    _relative_label,
    _relative_path,
    _sha256_bytes,
    _sha256_file,
    _select_action,
    environment_metadata,
    generate_protocol_bank,
    generate_tape_bank,
    load_tape_bank,
    run_episode,
)
from .models import GraphActorCritic
from .reward import CostWeights, assignment_map, compute_cost, compute_fixed_j_from_components
from .metrics import EpisodeMetrics
from .trainer import PPOConfig, PPOTrainer

MODES = ("single", "sequential", "overlap", "burst")

# ---------------------------------------------------------------------------
# Frozen protocol defaults for Preliminary
# ---------------------------------------------------------------------------
VARIANTS = ("PPO-MLP", "GPPO-NoGate", "GPPO-Adaptive")
TRAINING_SEEDS = (1101, 2202, 3303)
DEFAULT_BUDGET = 300_000
CHECKPOINT_INTERVAL = 25_000


@dataclass(frozen=True)
class PreliminaryProtocol:
    """Frozen protocol parameters for Preliminary training."""

    variants: tuple[str, ...] = VARIANTS
    training_seeds: tuple[int, ...] = TRAINING_SEEDS
    budget: int = DEFAULT_BUDGET
    checkpoint_interval: int = CHECKPOINT_INTERVAL
    events_per_tape: int = 5

    @property
    def num_checkpoints(self) -> int:
        return self.budget // self.checkpoint_interval

    @property
    def num_runs(self) -> int:
        return len(self.variants) * len(self.training_seeds)


@dataclass
class CheckpointRecord:
    """Metadata for a single training checkpoint."""

    variant: str
    training_seed: int
    decision_steps: int
    checkpoint_path: str
    checkpoint_sha256: str
    source_tree_hash: str
    attested_source_commit_sha: str
    protocol_sha256: str
    seed_manifest_sha256: str
    ppo_config: dict[str, Any]
    rng_state: dict[str, Any]
    created_at: str


@dataclass
class ValidationMetrics:
    """Metrics for a single checkpoint on the Validation bank."""

    checkpoint_path: str
    checkpoint_sha256: str
    variant: str
    training_seed: int
    decision_steps: int
    final_infeasible_count: int
    final_infeasible_rate: float
    cumulative_weighted_vacancy: float
    recovery_latency: float
    fixed_j: float


@dataclass
class FreezeManifest:
    """Frozen manifest after validation selection."""

    variant: str
    training_seed: int
    selected_step: int
    checkpoint_path: str
    checkpoint_sha256: str
    source_sha: str
    protocol_sha: str
    seed_manifest_sha: str
    validation_manifest_sha: str
    attested_source_commit_sha: str
    selected_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_checkpoint_steps(budget: int, interval: int) -> list[int]:
    """Return the sorted list of checkpoint decision-step counts."""
    steps = list(range(interval, budget + 1, interval))
    if steps[-1] != budget:
        steps.append(budget)
    return steps


def _get_rng_state() -> dict[str, Any]:
    return {
        "python_random": None,
        "numpy": np.random.get_state()[1][:4].tolist(),
        "torch": torch.random.get_rng_state()[:4].tolist() if torch.random.get_rng_state().numel() > 0 else [],
    }


def _check_p0_gate_strict() -> dict[str, Any]:
    """Run P0 gate check and return the gate dict. Raises SystemExit if RED."""
    from .experiment import _check_p0_gate
    _check_p0_gate()
    gate_path = Path(__file__).resolve().parents[2] / "handoff" / "P0_GATE.json"
    return json.loads(gate_path.read_text(encoding="utf-8"))


def _validate_hashes_match(gate: dict[str, Any], stage: str) -> None:
    """Verify source/protocol/seed_manifest hashes haven't drifted."""
    gate_path = Path(__file__).resolve().parents[2] / "handoff" / "P0_GATE.json"
    root = Path(__file__).resolve().parents[2]

    # Recompute source hashes
    from scripts.build_p0_gate import SOURCE_FILES, sha256_file
    current_sources = {}
    for rel in SOURCE_FILES:
        path = root / rel
        current_sources[rel] = sha256_file(path) if path.exists() else "MISSING"
    current_tree = hashlib.sha256(
        "".join(f"{k}:{v}\n" for k, v in sorted(current_sources.items())).encode()
    ).hexdigest()

    if current_tree != gate.get("source_tree_hash"):
        raise SystemExit(f"{stage}: source_tree_hash drifted — rerun gate")

    # Check protocol
    from scripts.build_p0_gate import PROTOCOL_PATH, SEED_MANIFEST_PATH
    current_protocol = sha256_file(PROTOCOL_PATH)
    current_seed_manifest = sha256_file(SEED_MANIFEST_PATH)
    if current_protocol != gate.get("protocol_sha256"):
        raise SystemExit(f"{stage}: protocol hash drifted — rerun gate")
    if current_seed_manifest != gate.get("seed_manifest_sha256"):
        raise SystemExit(f"{stage}: seed_manifest hash drifted — rerun gate")


def _require_finite_metric(value: Any, name: str) -> float:
    """Return a metric only when it is present and finite; never default to 0."""
    if value is None:
        raise ValueError(f"required validation metric is missing: {name}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"required validation metric is not numeric: {name}") from exc
    if not np.isfinite(number):
        raise ValueError(f"required validation metric is not finite: {name}")
    return number


def compute_fixed_j_from_episode(episode: EpisodeMetrics, trace: dict[str, Any]) -> float:
    """Compute the frozen fixed-J metric from real event metric rows.

    The function intentionally does not read a guessed/fallback field.  Every
    event contributes the five frozen cost components; a missing recovery
    latency is a failed validation evaluation rather than an artificial zero.
    """
    rows = trace.get("events")
    if not isinstance(rows, list) or len(rows) != int(episode.event_count):
        raise ValueError("trace.events must contain one metric row per episode event")
    total = 0.0
    for index, row in enumerate(rows):
        total += compute_fixed_j_from_components(
            uncovered=row.get("weighted_uncovered"),
            distance=row.get("normalized_distance"),
            load_gap=row.get("load_gap"),
            switches=row.get("switch_count"),
            recovery_delay=row.get("recovery_delay"),
        )
    return float(total)


def extract_validation_metrics(
    episode: EpisodeMetrics,
    trace: dict[str, Any],
    checkpoint: CheckpointRecord,
) -> ValidationMetrics:
    """Extract the five selection metrics from the authoritative EpisodeMetrics."""
    if not isinstance(episode, EpisodeMetrics):
        raise TypeError("validation extraction requires EpisodeMetrics")
    event_count = int(episode.event_count)
    if event_count <= 0:
        raise ValueError("validation episode must contain at least one event")
    final_count = int(episode.final_infeasible_count)
    final_rate = _require_finite_metric(episode.final_infeasible_rate, "final_infeasible_rate")
    expected_rate = final_count / event_count
    if abs(final_rate - expected_rate) > 1e-9:
        raise ValueError("final_infeasible_rate is inconsistent with final_infeasible_count")
    vacancy = _require_finite_metric(
        episode.cumulative_uncovered_time, "cumulative_weighted_vacancy"
    )
    observed_recoveries = int(episode.recovery_delay_observed_count)
    if observed_recoveries != event_count:
        raise ValueError(
            "recovery_delay is incomplete for validation selection; "
            f"observed={observed_recoveries}, events={event_count}"
        )
    recovery = _require_finite_metric(episode.recovery_delay, "recovery_latency")
    fixed_j = compute_fixed_j_from_episode(episode, trace)
    return ValidationMetrics(
        checkpoint_path=checkpoint.checkpoint_path,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        variant=checkpoint.variant,
        training_seed=checkpoint.training_seed,
        decision_steps=int(checkpoint.decision_steps),
        final_infeasible_count=final_count,
        final_infeasible_rate=final_rate,
        cumulative_weighted_vacancy=vacancy,
        recovery_latency=recovery,
        fixed_j=fixed_j,
    )


# ---------------------------------------------------------------------------
# Preliminary training
# ---------------------------------------------------------------------------

def preliminary_train(
    output_dir: Path,
    *,
    protocol: PreliminaryProtocol | None = None,
    ppo_config: PPOConfig | None = None,
    formal: bool = True,
) -> dict[str, Any]:
    """Run preliminary training: 3 variants × 3 seeds, periodic checkpoints.

    The P0 gate must be green before training starts.
    Resume is NOT supported — training always starts fresh.
    """
    from .experiment import CyclingTrainingEnv

    gate = _check_p0_gate_strict()
    _validate_hashes_match(gate, "preliminary-train")

    protocol = protocol or PreliminaryProtocol()
    if formal and protocol != PreliminaryProtocol():
        raise SystemExit(
            "formal Preliminary requires the frozen 300000/25000/3-variant/3-seed protocol; "
            "use developer mode for a dry-run"
        )
    ppo_config = ppo_config or PPOConfig(seed=1, device="cpu")
    model_dir = output_dir / "preliminary" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    source_tree_hash = gate.get("source_tree_hash", "UNKNOWN")
    attested_commit = gate.get("attested_source_commit_sha", "UNKNOWN")
    protocol_hash = gate.get("protocol_sha256", "UNKNOWN")
    seed_manifest_hash = gate.get("seed_manifest_sha256", "UNKNOWN")

    checkpoint_steps = compute_checkpoint_steps(protocol.budget, protocol.checkpoint_interval)
    all_checkpoints: list[CheckpointRecord] = []
    training_summary: dict[str, Any] = {
        "protocol": asdict(protocol),
        "ppo_config": asdict(ppo_config),
        "checkpoints": [],
        "created_at": _utc_now(),
        "resume_supported": False,
    }

    for variant in protocol.variants:
        for seed in protocol.training_seeds:
            env = CyclingTrainingEnv(
                seed=seed,
                modes=MODES,
                events_per_episode=protocol.events_per_tape,
            )
            # Copy every frozen PPO hyperparameter; only the run seed differs.
            config_values = asdict(ppo_config)
            config_values["seed"] = seed
            config_values["device"] = "cpu"
            config_values["rollout_steps"] = min(
                int(config_values["rollout_steps"]), max(1, protocol.budget)
            )
            config = PPOConfig(**config_values)
            trainer = PPOTrainer(env=env, variant=variant, config=config)
            started = time.perf_counter()

            for target_step in checkpoint_steps:
                remaining = target_step - trainer.total_steps
                if remaining > 0:
                    trainer.train(remaining)

                safe_variant = variant.lower().replace("-", "_")
                ckpt_name = f"{safe_variant}_seed{seed}_step{target_step}.pt"
                ckpt_path = model_dir / ckpt_name
                trainer.save(ckpt_path, extra={
                    "variant": variant,
                    "training_seed": seed,
                    "decision_steps": target_step,
                    "budget": protocol.budget,
                })
                # Re-hash after save (Phase J requirement 9)
                ckpt_sha = _sha256_file(ckpt_path)

                record = CheckpointRecord(
                    variant=variant,
                    training_seed=seed,
                    decision_steps=target_step,
                    checkpoint_path=_relative_label(ckpt_path),
                    checkpoint_sha256=ckpt_sha,
                    source_tree_hash=source_tree_hash,
                    attested_source_commit_sha=attested_commit,
                    protocol_sha256=protocol_hash,
                    seed_manifest_sha256=seed_manifest_hash,
                    ppo_config=asdict(config),
                    rng_state=_get_rng_state(),
                    created_at=_utc_now(),
                )
                all_checkpoints.append(record)

            elapsed = time.perf_counter() - started
            training_summary["checkpoints"].append({
                "variant": variant,
                "seed": seed,
                "elapsed_seconds": elapsed,
                "checkpoints_created": len([
                    c for c in all_checkpoints
                    if c.variant == variant and c.training_seed == seed
                ]),
            })
            env.close()

    _json_file(output_dir / "preliminary" / "training_summary.json", training_summary)
    _json_file(output_dir / "preliminary" / "checkpoint_index.json", [asdict(c) for c in all_checkpoints])
    return {"checkpoints": [asdict(c) for c in all_checkpoints], "summary": training_summary}


# ---------------------------------------------------------------------------
# Validation bank generation
# ---------------------------------------------------------------------------

def generate_validation_bank(output_dir: Path) -> dict[str, Any]:
    """Generate the frozen Validation bank (100 tapes: 25×4 modes, no Unseen)."""
    gate = _check_p0_gate_strict()
    _validate_hashes_match(gate, "generate-validation-bank")
    return generate_protocol_bank(
        output_dir / "preliminary",
        tier="preliminary",
        split="validation",
        events_per_tape=5,
    )


# ---------------------------------------------------------------------------
# Lexicographic checkpoint selection (per variant×seed group)
# ---------------------------------------------------------------------------

def _evaluate_checkpoint_on_bank(
    checkpoint: CheckpointRecord,
    bank_manifest_path: Path,
    max_decisions: int = 100,
) -> ValidationMetrics:
    """Evaluate a single checkpoint on all tapes in a validation bank."""
    manifest, tapes = load_tape_bank(bank_manifest_path)

    ckpt_path = _relative_path(checkpoint.checkpoint_path)
    fmt = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("format", "")
    if fmt in ("fair-ppo-mlp-v1", "fair-ppo-mlp-v2"):
        from .models import FairPPOMLP
        model, _ = FairPPOMLP.load(ckpt_path, map_location="cpu")
    else:
        model, _ = GraphActorCritic.load(ckpt_path, map_location="cpu")
    model.eval()

    from .baselines import GraphPolicyAdapter
    policy = GraphPolicyAdapter(
        model=model, name=f"{checkpoint.variant} step={checkpoint.decision_steps}"
    )

    extracted: list[ValidationMetrics] = []
    for tape_id, tape in tapes:
        episode, trace = run_episode(
            policy, tape_id=tape_id, tape=tape,
            algorithm=checkpoint.variant, max_decisions=max_decisions,
        )
        extracted.append(extract_validation_metrics(episode, trace, checkpoint))

    if not extracted:
        raise ValueError("validation bank produced no episodes")
    # Aggregate from the real per-tape metrics.  The rate denominator is the
    # total number of event opportunities, not the number of tapes.
    tape_event_counts = []
    for tape_id, tape in tapes:
        tape_event_counts.append(len(tape.events))
    event_denominator = sum(tape_event_counts)
    final_count = sum(metric.final_infeasible_count for metric in extracted)
    return ValidationMetrics(
        checkpoint_path=checkpoint.checkpoint_path,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        variant=checkpoint.variant,
        training_seed=checkpoint.training_seed,
        decision_steps=checkpoint.decision_steps,
        final_infeasible_count=final_count,
        final_infeasible_rate=final_count / event_denominator,
        cumulative_weighted_vacancy=float(sum(metric.cumulative_weighted_vacancy for metric in extracted)),
        recovery_latency=float(sum(metric.recovery_latency for metric in extracted)),
        fixed_j=float(sum(metric.fixed_j for metric in extracted)),
    )


def _lexicographic_select(
    metrics_list: list[ValidationMetrics],
) -> tuple[ValidationMetrics, list[ValidationMetrics]]:
    """Lexicographic sort: infeasible → vacancy → recovery → J → earlier step."""
    sorted_metrics = sorted(
        metrics_list,
        key=lambda m: (
            m.final_infeasible_rate,
            m.cumulative_weighted_vacancy,
            m.recovery_latency,
            m.fixed_j,
            m.decision_steps,
        ),
    )
    return sorted_metrics[0], sorted_metrics


def _validate_validation_manifest(
    manifest_path: Path,
    gate: dict[str, Any],
    *,
    formal: bool,
) -> dict[str, Any]:
    """Validate the frozen Validation bank contract before selection."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Validation bank not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if formal:
        required = {
            "split": "validation",
            "tier": "preliminary",
            "tape_count": 100,
            "expected_tape_count": 100,
            "complete_frozen_bank": True,
            "checkpoint_selection": True,
            "reward_tuning": False,
            "seed_manifest_sha256": gate.get("seed_manifest_sha256"),
            "protocol_sha256": gate.get("protocol_sha256"),
        }
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise SystemExit(
                    f"Validation manifest contract mismatch for {key}: "
                    f"{manifest.get(key)!r} != {expected!r}"
                )
        counts = {mode: 0 for mode in MODES}
        for entry in manifest.get("entries", []):
            mode = entry.get("mode")
            if mode in counts:
                counts[mode] += 1
        if counts != {mode: 25 for mode in MODES}:
            raise SystemExit(f"Validation mode counts mismatch: {counts}")
        if any(entry.get("distribution_profile") == "unseen_shift_v1" for entry in manifest["entries"]):
            raise SystemExit("formal Validation must not contain Unseen tapes")
    return manifest


def _validate_formal_checkpoints(
    checkpoints: list[CheckpointRecord],
    gate: dict[str, Any],
) -> None:
    """Require exactly 9 groups × 12 exact checkpoints for formal validation."""
    expected_keys = {(variant, seed) for variant in VARIANTS for seed in TRAINING_SEEDS}
    groups: dict[tuple[str, int], list[CheckpointRecord]] = {}
    for record in checkpoints:
        key = (record.variant, int(record.training_seed))
        groups.setdefault(key, []).append(record)
    if set(groups) != expected_keys:
        raise SystemExit(f"formal checkpoint groups mismatch: {sorted(groups)}")
    expected_steps = compute_checkpoint_steps(DEFAULT_BUDGET, CHECKPOINT_INTERVAL)
    reference_config = None
    for key, records in groups.items():
        steps = [int(record.decision_steps) for record in records]
        if len(records) != len(expected_steps) or sorted(steps) != expected_steps:
            raise SystemExit(f"formal checkpoints for {key} must be exactly {expected_steps}, got {steps}")
        if len({record.checkpoint_path for record in records}) != len(records):
            raise SystemExit(f"duplicate checkpoint path in group {key}")
        for record in records:
            path = _relative_path(record.checkpoint_path)
            if not path.exists() or _sha256_file(path) != record.checkpoint_sha256:
                raise SystemExit(f"checkpoint SHA mismatch: {record.checkpoint_path}")
            if record.source_tree_hash != gate.get("source_tree_hash"):
                raise SystemExit(f"checkpoint source hash mismatch: {record.checkpoint_path}")
            if record.attested_source_commit_sha != gate.get("attested_source_commit_sha"):
                raise SystemExit(f"checkpoint source commit mismatch: {record.checkpoint_path}")
            if record.protocol_sha256 != gate.get("protocol_sha256"):
                raise SystemExit(f"checkpoint protocol hash mismatch: {record.checkpoint_path}")
            if record.seed_manifest_sha256 != gate.get("seed_manifest_sha256"):
                raise SystemExit(f"checkpoint seed manifest hash mismatch: {record.checkpoint_path}")
            config_without_seed = dict(record.ppo_config)
            config_without_seed.pop("seed", None)
            if reference_config is None:
                reference_config = config_without_seed
            elif config_without_seed != reference_config:
                raise SystemExit(
                    f"PPO hyperparameters differ across formal runs: {record.checkpoint_path}"
                )


def preliminary_validate(
    output_dir: Path,
    checkpoints: list[CheckpointRecord] | None = None,
    *,
    formal: bool = True,
    validation_manifest: Path | None = None,
) -> dict[str, Any]:
    """Evaluate checkpoints on Validation bank and select best PER (variant, seed).

    Returns 9 selected checkpoints (3 variants × 3 seeds), each independently
    chosen from its own 12-checkpoint group.
    """
    gate = _check_p0_gate_strict()
    _validate_hashes_match(gate, "preliminary-validate")

    if checkpoints is None:
        ckpt_index_path = output_dir / "preliminary" / "checkpoint_index.json"
        if not ckpt_index_path.exists():
            raise FileNotFoundError(f"No checkpoint index: {ckpt_index_path}")
        raw = json.loads(ckpt_index_path.read_text(encoding="utf-8"))
        checkpoints = [CheckpointRecord(**c) for c in raw]

    bank_manifest = validation_manifest or (
        output_dir / "preliminary" / "tapes" / "preliminary_validation_protocol" / "manifest.json"
    )
    manifest = _validate_validation_manifest(bank_manifest, gate, formal=formal)
    if formal:
        _validate_formal_checkpoints(checkpoints, gate)
    else:
        if not checkpoints:
            raise ValueError("developer validation requires checkpoints")
        # Developer dry-runs still require exactly the three variants × three
        # seeds, but permit a smaller common checkpoint schedule.
        expected_keys = {(variant, seed) for variant in VARIANTS for seed in TRAINING_SEEDS}
        actual_keys = {(c.variant, int(c.training_seed)) for c in checkpoints}
        if actual_keys != expected_keys:
            raise SystemExit("developer validation must contain all 9 variant/seed groups")

    # Group by (variant, training_seed) → select 1 per group
    groups: dict[tuple[str, int], list[CheckpointRecord]] = {}
    for ckpt in checkpoints:
        key = (ckpt.variant, ckpt.training_seed)
        groups.setdefault(key, []).append(ckpt)

    all_selections = []
    all_metrics_by_group = {}
    for (variant, seed), group_ckpts in sorted(groups.items()):
        selected, all_group_metrics = _lexicographic_select([
            _evaluate_checkpoint_on_bank(ckpt, bank_manifest)
            for ckpt in group_ckpts
        ])
        all_selections.append(selected)
        all_metrics_by_group[f"{variant}_seed{seed}"] = [asdict(m) for m in all_group_metrics]

    validation_manifest_sha = _sha256_file(bank_manifest)

    selection = {
        "created_at": _utc_now(),
        "num_groups": len(groups),
        "selected_count": len(all_selections),
        "selected_checkpoints": [asdict(m) for m in all_selections],
        "all_metrics_by_group": all_metrics_by_group,
        "validation_manifest_sha": validation_manifest_sha,
        "validation_manifest_sha256": validation_manifest_sha,
        "validation_manifest_path": _relative_label(bank_manifest),
        "protocol_sha256": gate.get("protocol_sha256"),
        "seed_manifest_sha256": gate.get("seed_manifest_sha256"),
        "source_tree_hash": gate.get("source_tree_hash"),
        "attested_source_commit_sha": gate.get("attested_source_commit_sha"),
        "formal": formal,
        "selection_criteria": [
            "lowest_final_infeasible_rate",
            "lowest_cumulative_weighted_vacancy",
            "lowest_recovery_latency",
            "lowest_fixed_j",
            "earlier_checkpoint_tiebreak",
        ],
    }
    _json_file(output_dir / "preliminary" / "validation_selection.json", selection)
    return selection


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------

def preliminary_freeze(
    output_dir: Path,
    *,
    formal: bool = True,
    validation_manifest: Path | None = None,
) -> dict[str, Any]:
    """Freeze selected checkpoints after revalidating hashes and selection cardinality."""
    gate = _check_p0_gate_strict()
    _validate_hashes_match(gate, "preliminary-freeze")

    selection_path = output_dir / "preliminary" / "validation_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("Run preliminary-validate first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected_keys = {(variant, seed) for variant in VARIANTS for seed in TRAINING_SEEDS}
    selected = selection.get("selected_checkpoints", [])
    selected_keys = {(item.get("variant"), int(item.get("training_seed"))) for item in selected}
    if selected_keys != expected_keys or len(selected) != len(expected_keys):
        raise SystemExit(
            f"freeze requires exactly 9 unique variant/seed selections; got {len(selected)}"
        )

    manifest_path = validation_manifest or _relative_path(
        selection.get("validation_manifest_path", "results/random_event/tapes/preliminary_validation_protocol/manifest.json")
    )
    validation_sha = _sha256_file(manifest_path)

    # Verify validation manifest SHA matches what was recorded at selection time
    if validation_sha != selection.get("validation_manifest_sha"):
        raise SystemExit(
            "Validation manifest SHA mismatch between selection and freeze — "
            "bank may have been modified"
        )

    freezes = []
    for sel in selection["selected_checkpoints"]:
        # Re-hash checkpoint before freeze (Phase J requirement 9)
        ckpt_path = _relative_path(sel["checkpoint_path"])
        current_ckpt_sha = _sha256_file(ckpt_path)
        if current_ckpt_sha != sel["checkpoint_sha256"]:
            raise SystemExit(
                f"Checkpoint SHA mismatch: {sel['checkpoint_path']} — "
                "checkpoint may have been modified after selection"
            )

        freeze = FreezeManifest(
            variant=sel["variant"],
            training_seed=sel["training_seed"],
            selected_step=sel["decision_steps"],
            checkpoint_path=sel["checkpoint_path"],
            checkpoint_sha256=current_ckpt_sha,
            source_sha=gate.get("source_tree_hash", "UNKNOWN"),
            protocol_sha=gate.get("protocol_sha256", "UNKNOWN"),
            seed_manifest_sha=gate.get("seed_manifest_sha256", "UNKNOWN"),
            validation_manifest_sha=validation_sha,
            attested_source_commit_sha=gate.get("attested_source_commit_sha", "UNKNOWN"),
            selected_at=_utc_now(),
        )
        freezes.append(asdict(freeze))

    _json_file(output_dir / "preliminary" / "frozen_manifests.json", {
        "created_at": _utc_now(),
        "freeze_count": len(freezes),
        "freezes": freezes,
        "formal": formal,
        "validation_manifest_sha256": validation_sha,
        "source_tree_hash": gate.get("source_tree_hash"),
        "attested_source_commit_sha": gate.get("attested_source_commit_sha"),
        "protocol_sha256": gate.get("protocol_sha256"),
        "seed_manifest_sha256": gate.get("seed_manifest_sha256"),
    })

    # Also write individual per-(variant, seed) manifests
    freeze_dir = output_dir / "preliminary" / "freezes"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    for f in freezes:
        safe = f"{f['variant'].lower().replace('-', '_')}_seed{f['training_seed']}"
        _json_file(freeze_dir / f"{safe}.json", f)

    return {"freeze_count": len(freezes), "freezes": freezes}


# ---------------------------------------------------------------------------
# Test bank generation & execution
# ---------------------------------------------------------------------------

def generate_test_bank(output_dir: Path) -> dict[str, Any]:
    """Generate the frozen Test bank (200 tapes: 40×5 sets).

    Requires frozen selection to exist before generating.
    """
    freeze_path = output_dir / "preliminary" / "frozen_manifests.json"
    if not freeze_path.exists():
        raise SystemExit(
            "Cannot generate Test bank: run preliminary-freeze first. "
            "Test bank must not be generated before checkpoint selection is frozen."
        )
    freeze_payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if int(freeze_payload.get("freeze_count", 0)) != 9:
        raise SystemExit("Cannot generate formal Test bank until exactly 9 checkpoints are frozen")

    gate = _check_p0_gate_strict()
    _validate_hashes_match(gate, "generate-test-bank")

    return generate_protocol_bank(
        output_dir / "preliminary",
        tier="preliminary",
        split="test",
        events_per_tape=5,
    )


def preliminary_test(output_dir: Path) -> dict[str, Any]:
    """Run Test bank on all 9 frozen checkpoints.

    Each checkpoint gets its own test run.  The test ledger records
    consumption status to prevent re-testing.
    """
    freeze_path = output_dir / "preliminary" / "frozen_manifests.json"
    if not freeze_path.exists():
        raise FileNotFoundError("Run preliminary-freeze first")

    gate = _check_p0_gate_strict()
    _validate_hashes_match(gate, "preliminary-test")

    freeze_payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    freezes = freeze_payload.get("freezes", [])
    expected_keys = {(variant, seed) for variant in VARIANTS for seed in TRAINING_SEEDS}
    freeze_keys = {(item.get("variant"), int(item.get("training_seed"))) for item in freezes}
    if len(freezes) != 9 or freeze_keys != expected_keys:
        raise SystemExit("formal Test requires exactly 9 frozen variant/seed checkpoints")

    # Load test bank
    test_bank_dir = output_dir / "preliminary" / "tapes" / "preliminary_test_protocol"
    test_manifest = test_bank_dir / "manifest.json"
    if not test_manifest.exists():
        raise FileNotFoundError("Run generate_test_bank first")

    test_manifest_sha = _sha256_file(test_manifest)
    test_manifest_payload = json.loads(test_manifest.read_text(encoding="utf-8"))
    if test_manifest_payload.get("split") != "test" or test_manifest_payload.get("tier") != "preliminary":
        raise SystemExit("formal Test requires a preliminary/test manifest")
    if test_manifest_payload.get("tape_count") != 200:
        raise SystemExit("formal Test requires exactly 200 tapes")
    mode_counts = {mode: 0 for mode in MODES}
    for item in test_manifest_payload.get("entries", []):
        mode = item.get("mode")
        if mode in mode_counts:
            mode_counts[mode] += 1
    # Unseen is represented by Test-Unseen entries whose mode cycles over the
    # four timing modes; the set count is checked separately.
    set_counts = {}
    for item in test_manifest_payload.get("entries", []):
        set_counts[item.get("set_name")] = set_counts.get(item.get("set_name"), 0) + 1
    if set_counts.get("Test-Unseen") != 40 or any(
        set_counts.get(f"Test-{mode.title()}") != 40 for mode in MODES
    ):
        raise SystemExit(f"formal Test mode/set counts mismatch: {set_counts}")
    if test_manifest_payload.get("seed_manifest_sha256") != gate.get("seed_manifest_sha256"):
        raise SystemExit("Test seed manifest hash mismatch")
    if test_manifest_payload.get("protocol_sha256") != gate.get("protocol_sha256"):
        raise SystemExit("Test protocol hash mismatch")

    # Test ledger tracks per-(variant, seed, checkpoint_sha, test_manifest_sha)
    ledger_path = output_dir / "preliminary" / "test_ledger.json"
    ledger = {}
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    all_results = []
    for freeze in freezes:
        key = (
            f"{freeze['variant']}_seed{freeze['training_seed']}_"
            f"{freeze['checkpoint_sha256'][:12]}_{test_manifest_sha}"
        )
        if ledger.get(key, {}).get("consumed"):
            raise SystemExit(
                f"formal Test combination already consumed: {key}; "
                "use an explicit developer retest path"
            )

        # Verify checkpoint SHA matches freeze
        ckpt_path = _relative_path(freeze["checkpoint_path"])
        current_sha = _sha256_file(ckpt_path)
        if current_sha != freeze["checkpoint_sha256"]:
            raise SystemExit(
                f"Checkpoint SHA mismatch at test time: {freeze['checkpoint_path']}"
            )

        # Load model by variant
        fmt = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("format", "")
        if fmt in ("fair-ppo-mlp-v1", "fair-ppo-mlp-v2"):
            from .models import FairPPOMLP
            model, _ = FairPPOMLP.load(ckpt_path, map_location="cpu")
        else:
            model, _ = GraphActorCritic.load(ckpt_path, map_location="cpu")
        model.eval()

        from .baselines import GraphPolicyAdapter
        policy = GraphPolicyAdapter(
            model=model,
            name=f"{freeze['variant']} step={freeze['selected_step']}",
        )

        manifest, tapes = load_tape_bank(test_manifest)

        results = []
        for tape_id, tape in tapes:
            episode, trace = run_episode(
                policy, tape_id=tape_id, tape=tape,
                algorithm=freeze["variant"], max_decisions=100,
            )
            results.append({
                "tape_id": tape_id,
                "mode": tape.mode,
                "episode": episode.to_dict(),
            })

        result_path = (
            output_dir / "preliminary" / "test_results"
            / f"{freeze['variant'].lower().replace('-', '_')}_seed{freeze['training_seed']}.json"
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        _json_file(result_path, {
            "created_at": _utc_now(),
            "freeze": freeze,
            "test_manifest_sha256": test_manifest_sha,
            "tape_count": len(results),
            "results": results,
        })

        ledger[key] = {
            "consumed": True,
            "variant": freeze["variant"],
            "training_seed": freeze["training_seed"],
            "checkpoint_sha": freeze["checkpoint_sha256"],
            "test_manifest_sha": test_manifest_sha,
            "at": _utc_now(),
            "result_path": _relative_label(result_path),
        }
        all_results.append({"variant": freeze["variant"], "seed": freeze["training_seed"], "tape_count": len(results)})

    _json_file(ledger_path, ledger)

    return {
        "consumed_count": len(all_results),
        "results": all_results,
        "test_manifest_sha": test_manifest_sha,
    }


def generate_developer_validation_bank(output_dir: Path) -> dict[str, Any]:
    """Create a small validation-only developer bank in a separate namespace."""
    gate = _check_p0_gate_strict()
    _validate_hashes_match(gate, "generate-developer-validation-bank")
    manifest = generate_tape_bank(
        output_dir / "preliminary",
        modes=MODES,
        tapes_per_mode=2,
        events_per_tape=2,
        master_seed=990001,
        bank_name="dev_validation",
    )
    payload = json.loads((output_dir / "preliminary" / "tapes" / "dev_validation" / "manifest.json").read_text())
    payload.update({
        "tier": "developer",
        "split": "validation",
        "intended_use": "developer_orchestrator_smoke_only",
        "formal": False,
        "reward_tuning": False,
        "seed_manifest_sha256": gate.get("seed_manifest_sha256"),
        "protocol_sha256": gate.get("protocol_sha256"),
    })
    _json_file(output_dir / "preliminary" / "tapes" / "dev_validation" / "manifest.json", payload)
    payload["manifest_path"] = _relative_label(output_dir / "preliminary" / "tapes" / "dev_validation" / "manifest.json")
    return payload


# ---------------------------------------------------------------------------
# DRY RUN orchestrator
# ---------------------------------------------------------------------------

def dry_run(output_dir: Path) -> dict[str, Any]:
    """Small-budget DRY RUN to verify the orchestrator pipeline.

    3 variants × 3 seeds × 2 checkpoints (64, 128 steps).
    Validates: checkpoint creation, save/load, validation selection, freeze,
    test isolation guard.

    Does NOT use formal Test seed namespace.
    Does NOT produce Preliminary results.
    """
    protocol = PreliminaryProtocol(
        budget=128,
        checkpoint_interval=64,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test isolation guard must reject formal Test generation before freeze.
    test_before_freeze_rejected = False
    try:
        generate_test_bank(output_dir)
    except (SystemExit, FileNotFoundError):
        test_before_freeze_rejected = True
    if not test_before_freeze_rejected:
        raise AssertionError("dry-run generated formal Test before freeze")

    # Step 1: Train, explicitly developer mode (formal budget is untouched).
    train_result = preliminary_train(output_dir, protocol=protocol, formal=False)
    num_ckpt = len(train_result["checkpoints"])

    # Step 2: Use a separate developer Validation namespace.
    dev_manifest = generate_developer_validation_bank(output_dir)
    val_result = preliminary_validate(
        output_dir,
        checkpoints=[CheckpointRecord(**item) for item in train_result["checkpoints"]],
        formal=False,
        validation_manifest=output_dir / "preliminary" / "tapes" / "dev_validation" / "manifest.json",
    )
    num_selected = val_result["selected_count"]

    # Step 3: Freeze exactly nine developer selections.
    freeze_result = preliminary_freeze(
        output_dir,
        formal=False,
        validation_manifest=output_dir / "preliminary" / "tapes" / "dev_validation" / "manifest.json",
    )
    num_freeze = freeze_result["freeze_count"]

    summary = {
        "formal": False,
        "dry_run": True,
        "run_label": "DRY_RUN / ORCHESTRATOR_SMOKE",
        "note": "This is a DRY RUN — not a Preliminary result",
        "variants": list(VARIANTS),
        "seeds": list(TRAINING_SEEDS),
        "checkpoints_per_group": 2,
        "train_checkpoints": num_ckpt,
        "selected_per_group": num_selected,
        "frozen_count": num_freeze,
        "validation_manifest": dev_manifest.get("manifest_path"),
        "official_test_namespace_touched": False,
        "test_before_freeze_rejected": test_before_freeze_rejected,
        "source_tree_hash": val_result.get("source_tree_hash"),
        "attested_source_commit_sha": val_result.get("attested_source_commit_sha"),
        "protocol_sha256": val_result.get("protocol_sha256"),
        "seed_manifest_sha256": val_result.get("seed_manifest_sha256"),
        "created_at": _utc_now(),
    }
    _json_file(output_dir / "phase_j_dry_run_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def add_phase_j_args(parser: argparse.ArgumentParser) -> None:
    """Add Phase J subcommands to an argument parser."""
    sub = parser.add_subparsers(dest="phase_j_cmd")

    train_p = sub.add_parser("preliminary-train", help="Run 300k preliminary training")
    train_p.add_argument("--output-dir", default="results/random_event")
    train_p.add_argument(
        "--developer-mode", action="store_true",
        help="allow non-frozen budget/interval for developer experiments only",
    )
    train_p.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help=argparse.SUPPRESS)
    train_p.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL, help=argparse.SUPPRESS)

    val_p = sub.add_parser("preliminary-validate", help="Run validation selection")
    val_p.add_argument("--output-dir", default="results/random_event")

    freeze_p = sub.add_parser("preliminary-freeze", help="Freeze selected checkpoints")
    freeze_p.add_argument("--output-dir", default="results/random_event")

    test_p = sub.add_parser("preliminary-test", help="Run test on frozen checkpoints")
    test_p.add_argument("--output-dir", default="results/random_event")

    dry_p = sub.add_parser("phase-j-dry-run", help="Small-budget orchestrator smoke test")
    dry_p.add_argument("--output-dir", default="results/random_event/phase_j_dry_run")


def run_phase_j_command(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch a Phase J subcommand."""
    output_dir = _relative_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase_j_cmd == "preliminary-train":
        if not args.developer_mode and (
            args.budget != DEFAULT_BUDGET or args.checkpoint_interval != CHECKPOINT_INTERVAL
        ):
            raise SystemExit(
                "formal preliminary-train is frozen at 300000/25000; "
                "use --developer-mode for a developer run"
            )
        protocol = PreliminaryProtocol(
            budget=args.budget,
            checkpoint_interval=args.checkpoint_interval,
        )
        result = preliminary_train(
            output_dir, protocol=protocol, formal=not args.developer_mode
        )
        print(f"Training complete: {len(result['checkpoints'])} checkpoints")
    elif args.phase_j_cmd == "preliminary-validate":
        result = preliminary_validate(output_dir)
        print(f"Selected {result['selected_count']} checkpoints across {result['num_groups']} groups")
    elif args.phase_j_cmd == "preliminary-freeze":
        result = preliminary_freeze(output_dir)
        print(f"Frozen {result['freeze_count']} checkpoints")
    elif args.phase_j_cmd == "preliminary-test":
        result = preliminary_test(output_dir)
        print(f"Test complete: {result['consumed_count']} checkpoints tested")
    elif args.phase_j_cmd == "phase-j-dry-run":
        result = dry_run(output_dir)
        print(f"Dry-run: {result['train_checkpoints']} ckpts → {result['selected_per_group']} selected → {result['frozen_count']} frozen")
    else:
        raise SystemExit(f"Unknown Phase J command: {args.phase_j_cmd}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase J Preliminary workflow")
    add_phase_j_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "phase_j_cmd", None):
        build_parser().print_help()
        return 2
    run_phase_j_command(args)
    return 0


__all__ = [
    "CheckpointRecord", "FreezeManifest", "PreliminaryProtocol", "ValidationMetrics",
    "compute_checkpoint_steps", "compute_fixed_j_from_episode", "extract_validation_metrics",
    "generate_developer_validation_bank", "generate_test_bank", "generate_validation_bank",
    "preliminary_train", "preliminary_validate", "preliminary_freeze", "preliminary_test",
    "dry_run", "build_parser", "main", "_lexicographic_select",
]


if __name__ == "__main__":
    raise SystemExit(main())
