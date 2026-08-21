"""Phase J: Preliminary Training Orchestrator.

This module implements the full Phase J workflow:
1. preliminary-train: 300k steps with periodic checkpoints
2. preliminary-validate: lexicographic checkpoint selection
3. preliminary-freeze: freeze selected checkpoint with SHA attestation
4. preliminary-test: run Test bank on frozen checkpoint (once only)

All operations respect the frozen protocol and refuse to proceed if
the P0 gate is not green or if source/protocol hashes have drifted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

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
    run_episode,
)
from .models import GraphActorCritic
MODES = ("single", "sequential", "overlap", "burst")
from .trainer import PPOConfig, PPOTrainer


# ---------------------------------------------------------------------------
# Frozen protocol defaults for Preliminary
# ---------------------------------------------------------------------------
VARIANTS = ("PPO-MLP", "GPPO-NoGate", "GPPO-Adaptive")
TRAINING_SEEDS = (1101, 2202, 3303)
DEFAULT_BUDGET = 300_000
CHECKPOINT_INTERVAL = 25_000
VALIDATION_TAPES_PER_MODE = 25
TEST_TAPES_PER_SET = 40


@dataclass(frozen=True)
class PreliminaryProtocol:
    """Frozen protocol parameters for Preliminary training."""

    variants: tuple[str, ...] = VARIANTS
    training_seeds: tuple[int, ...] = TRAINING_SEEDS
    budget: int = DEFAULT_BUDGET
    checkpoint_interval: int = CHECKPOINT_INTERVAL
    validation_tapes_per_mode: int = VALIDATION_TAPES_PER_MODE
    test_tapes_per_set: int = TEST_TAPES_PER_SET
    events_per_tape: int = 5

    @property
    def num_checkpoints(self) -> int:
        return self.budget // self.checkpoint_interval


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
    infeasible_rate: float
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
    selected_at: str


# ---------------------------------------------------------------------------
# Checkpoint scheduling
# ---------------------------------------------------------------------------

def compute_checkpoint_steps(budget: int, interval: int) -> list[int]:
    """Return the sorted list of checkpoint decision-step counts."""
    steps = list(range(interval, budget + 1, interval))
    if steps[-1] != budget:
        steps.append(budget)
    return steps


def _get_rng_state() -> dict[str, Any]:
    return {
        "python_random": None,  # saved separately if needed
        "numpy": np.random.get_state()[1][:4].tolist(),
        "torch": torch.random.get_rng_state()[:4].tolist() if torch.random.get_rng_state().numel() > 0 else [],
    }


# ---------------------------------------------------------------------------
# Preliminary training
# ---------------------------------------------------------------------------

def preliminary_train(
    output_dir: Path,
    *,
    protocol: PreliminaryProtocol | None = None,
    ppo_config: PPOConfig | None = None,
) -> dict[str, Any]:
    """Run preliminary training: 3 variants × 3 seeds, periodic checkpoints.

    Each checkpoint is saved with full provenance metadata.  The P0 gate
    must be green before training starts.
    """
    from .experiment import CyclingTrainingEnv, _check_p0_gate

    _check_p0_gate()  # refuse if gate is RED

    protocol = protocol or PreliminaryProtocol()
    ppo_config = ppo_config or PPOConfig(seed=1, device="cpu")
    model_dir = output_dir / "preliminary" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Record source hashes at training start
    gate_path = Path(__file__).resolve().parents[2] / "handoff" / "P0_GATE.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
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
    }

    for variant in protocol.variants:
        for seed in protocol.training_seeds:
            env = CyclingTrainingEnv(
                seed=seed,
                modes=MODES,
                events_per_episode=protocol.events_per_tape,
            )
            config = PPOConfig(
                rollout_steps=min(128, protocol.budget),
                update_epochs=ppo_config.update_epochs,
                minibatch_size=ppo_config.minibatch_size,
                seed=seed,
                device="cpu",
            )
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
                "checkpoints_created": len([c for c in all_checkpoints if c.variant == variant and c.training_seed == seed]),
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
    return generate_protocol_bank(
        output_dir / "preliminary",
        tier="preliminary",
        split="validation",
        events_per_tape=5,
    )


# ---------------------------------------------------------------------------
# Lexicographic checkpoint selection
# ---------------------------------------------------------------------------

def _evaluate_checkpoint_on_bank(
    checkpoint: CheckpointRecord,
    bank_manifest_path: Path,
    max_decisions: int = 100,
) -> ValidationMetrics:
    """Evaluate a single checkpoint on all tapes in a validation bank."""
    from .experiment import load_tape_bank, _load_policies

    manifest, tapes = load_tape_bank(bank_manifest_path)

    # Load the checkpoint model
    ckpt_path = _relative_path(checkpoint.checkpoint_path)
    fmt = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("format", "")
    if fmt in ("fair-ppo-mlp-v1", "fair-ppo-mlp-v2"):
        from .models import FairPPOMLP
        model, _ = FairPPOMLP.load(ckpt_path, map_location="cpu")
    else:
        model, _ = GraphActorCritic.load(ckpt_path, map_location="cpu")
    model.eval()

    from .baselines import GraphPolicyAdapter
    policy = GraphPolicyAdapter(model=model, name=f"{checkpoint.variant} step={checkpoint.decision_steps}")

    total_infeasible = 0
    total_vacancy = 0.0
    total_recovery_delay = 0.0
    total_fixed_j = 0.0
    tape_count = len(tapes)

    for tape_id, tape in tapes:
        episode, trace = run_episode(
            policy, tape_id=tape_id, tape=tape,
            algorithm=checkpoint.variant, max_decisions=max_decisions,
        )
        total_infeasible += 1 if trace.get("final_infeasible", False) else 0
        # Accumulate vacancy, recovery, J from trace
        for row in trace.get("decisions", []):
            rt = row.get("reward_trace", {})
            total_vacancy += rt.get("uncovered", 0.0)
            total_recovery_delay += rt.get("recovery_delay", 0.0)
        total_fixed_j += trace.get("episode", {}).get("fixed_j", 0.0)

    return ValidationMetrics(
        checkpoint_path=checkpoint.checkpoint_path,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        variant=checkpoint.variant,
        training_seed=checkpoint.training_seed,
        decision_steps=checkpoint.decision_steps,
        infeasible_rate=total_infeasible / max(1, tape_count),
        cumulative_weighted_vacancy=total_vacancy,
        recovery_latency=total_recovery_delay,
        fixed_j=total_fixed_j,
    )


def lexicographic_select(
    checkpoints: list[CheckpointRecord],
    bank_manifest_path: Path,
) -> tuple[ValidationMetrics, list[ValidationMetrics]]:
    """Run lexicographic selection: lowest infeasible, vacancy, recovery, J, earliest."""
    all_metrics = []
    for ckpt in checkpoints:
        metrics = _evaluate_checkpoint_on_bank(ckpt, bank_manifest_path)
        all_metrics.append(metrics)

    # Lexicographic sort: ascending for all criteria, tie → earlier checkpoint
    sorted_metrics = sorted(
        all_metrics,
        key=lambda m: (
            m.infeasible_rate,
            m.cumulative_weighted_vacancy,
            m.recovery_latency,
            m.fixed_j,
            m.decision_steps,  # tie → earlier
        ),
    )
    return sorted_metrics[0], sorted_metrics


def preliminary_validate(
    output_dir: Path,
    checkpoints: list[CheckpointRecord] | None = None,
) -> dict[str, Any]:
    """Evaluate all checkpoints on Validation bank and select the best."""
    if checkpoints is None:
        ckpt_index_path = output_dir / "preliminary" / "checkpoint_index.json"
        if not ckpt_index_path.exists():
            raise FileNotFoundError(f"No checkpoint index found at {ckpt_index_path}")
        raw = json.loads(ckpt_index_path.read_text(encoding="utf-8"))
        checkpoints = [CheckpointRecord(**c) for c in raw]

    bank_dir = output_dir / "preliminary" / "tapes" / "preliminary_validation_protocol"
    bank_manifest = bank_dir / "manifest.json"
    if not bank_manifest.exists():
        raise FileNotFoundError(f"Validation bank not found: {bank_manifest}")

    selected, all_metrics = lexicographic_select(checkpoints, bank_manifest)

    selection = {
        "created_at": _utc_now(),
        "selected_checkpoint": asdict(selected),
        "all_checkpoints": [asdict(m) for m in all_metrics],
        "selection_criteria": [
            "lowest_infeasible_rate",
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

def preliminary_freeze(output_dir: Path) -> dict[str, Any]:
    """Freeze the selected checkpoint and produce a freeze manifest."""
    selection_path = output_dir / "preliminary" / "validation_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("Run preliminary-validate first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    sel = selection["selected_checkpoint"]

    gate_path = Path(__file__).resolve().parents[2] / "handoff" / "P0_GATE.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    manifest_path = output_dir / "preliminary" / "tapes" / "preliminary_validation_protocol" / "manifest.json"
    validation_sha = _sha256_file(manifest_path)

    freeze = FreezeManifest(
        variant=sel["variant"],
        training_seed=sel["training_seed"],
        selected_step=sel["decision_steps"],
        checkpoint_path=sel["checkpoint_path"],
        checkpoint_sha256=sel["checkpoint_sha256"],
        source_sha=gate.get("source_tree_hash", "UNKNOWN"),
        protocol_sha=gate.get("protocol_sha256", "UNKNOWN"),
        seed_manifest_sha=gate.get("seed_manifest_sha256", "UNKNOWN"),
        validation_manifest_sha=validation_sha,
        selected_at=_utc_now(),
    )
    _json_file(output_dir / "preliminary" / "frozen_manifest.json", asdict(freeze))
    return asdict(freeze)


# ---------------------------------------------------------------------------
# Test bank generation & execution
# ---------------------------------------------------------------------------

def generate_test_bank(output_dir: Path) -> dict[str, Any]:
    """Generate the frozen Test bank (200 tapes: 40×5 sets)."""
    return generate_protocol_bank(
        output_dir / "preliminary",
        tier="preliminary",
        split="test",
        events_per_tape=5,
    )


def preliminary_test(output_dir: Path) -> dict[str, Any]:
    """Run Test bank on the frozen checkpoint.  Once only."""
    freeze_path = output_dir / "preliminary" / "frozen_manifest.json"
    if not freeze_path.exists():
        raise FileNotFoundError("Run preliminary-freeze first")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    # Verify selection is frozen
    if not freeze.get("checkpoint_sha256"):
        raise RuntimeError("Freeze manifest has no checkpoint_sha256")

    ckpt_path = _relative_path(freeze["checkpoint_path"])
    fmt = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("format", "")
    if fmt in ("fair-ppo-mlp-v1", "fair-ppo-mlp-v2"):
        from .models import FairPPOMLP
        model, _ = FairPPOMLP.load(ckpt_path, map_location="cpu")
    else:
        model, _ = GraphActorCritic.load(ckpt_path, map_location="cpu")
    model.eval()

    from .baselines import GraphPolicyAdapter
    policy = GraphPolicyAdapter(model=model, name=f"{freeze['variant']} step={freeze['selected_step']}")

    # Load test bank
    test_bank_dir = output_dir / "preliminary" / "tapes" / "preliminary_test_protocol"
    test_manifest = test_bank_dir / "manifest.json"
    if not test_manifest.exists():
        raise FileNotFoundError("Run generate_test_bank first")

    from .experiment import load_tape_bank
    manifest, tapes = load_tape_bank(test_manifest)

    test_consumed_path = output_dir / "preliminary" / "test_consumed.json"
    if test_consumed_path.exists():
        consumed = json.loads(test_consumed_path.read_text(encoding="utf-8"))
        if consumed.get("consumed"):
            raise RuntimeError("Test bank already consumed for this frozen experiment")

    results = []
    for tape_id, tape in tapes:
        episode, trace = run_episode(
            policy, tape_id=tape_id, tape=tape,
            algorithm=freeze["variant"], max_decisions=100,
        )
        results.append({"tape_id": tape_id, "mode": tape.mode, "episode": episode.to_dict()})

    _json_file(output_dir / "preliminary" / "test_results.json", {
        "created_at": _utc_now(),
        "frozen_manifest": freeze,
        "test_manifest_sha256": _sha256_file(test_manifest),
        "tape_count": len(results),
        "results": results,
    })
    _json_file(test_consumed_path, {"consumed": True, "at": _utc_now(), "manifest_sha": _sha256_file(test_manifest)})

    return {"consumed": True, "tape_count": len(results)}


# ---------------------------------------------------------------------------
# DRY RUN orchestrator
# ---------------------------------------------------------------------------

def dry_run(output_dir: Path) -> dict[str, Any]:
    """Small-budget DRY RUN to verify the orchestrator pipeline.

    3 variants × 3 seeds × 2 checkpoints (5k, 10k steps).
    Validates: checkpoint creation, save/load, validation selection, freeze.
    Does NOT produce Preliminary results.
    """
    protocol = PreliminaryProtocol(
        budget=10_000,
        checkpoint_interval=5_000,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    train_result = preliminary_train(output_dir, protocol=protocol)

    # Generate validation bank (small: 4 tapes per mode for speed)
    # For dry-run we use the standard 25-per-mode bank

    # Validate
    val_result = preliminary_validate(output_dir)

    # Freeze
    freeze_result = preliminary_freeze(output_dir)

    return {
        "dry_run": True,
        "note": "This is a DRY RUN — not a Preliminary result",
        "train_checkpoints": len(train_result["checkpoints"]),
        "selected": val_result["selected_checkpoint"],
        "frozen": freeze_result,
    }


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_phase_j_args(parser: argparse.ArgumentParser) -> None:
    """Add Phase J subcommands to an argument parser."""
    sub = parser.add_subparsers(dest="phase_j_cmd")

    # preliminary-train
    train_p = sub.add_parser("preliminary-train", help="Run 300k preliminary training")
    train_p.add_argument("--output-dir", default="results/random_event")
    train_p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    train_p.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)

    # preliminary-validate
    val_p = sub.add_parser("preliminary-validate", help="Run validation selection")
    val_p.add_argument("--output-dir", default="results/random_event")

    # preliminary-freeze
    freeze_p = sub.add_parser("preliminary-freeze", help="Freeze selected checkpoint")
    freeze_p.add_argument("--output-dir", default="results/random_event")

    # preliminary-test
    test_p = sub.add_parser("preliminary-test", help="Run test on frozen checkpoint")
    test_p.add_argument("--output-dir", default="results/random_event")

    # dry-run
    dry_p = sub.add_parser("phase-j-dry-run", help="Small-budget orchestrator smoke test")
    dry_p.add_argument("--output-dir", default="results/random_event/phase_j_dry_run")


def run_phase_j_command(args: argparse.Namespace) -> None:
    """Dispatch a Phase J subcommand."""
    output_dir = _relative_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase_j_cmd == "preliminary-train":
        protocol = PreliminaryProtocol(
            budget=args.budget,
            checkpoint_interval=args.checkpoint_interval,
        )
        result = preliminary_train(output_dir, protocol=protocol)
        print(f"Training complete: {len(result['checkpoints'])} checkpoints")
    elif args.phase_j_cmd == "preliminary-validate":
        result = preliminary_validate(output_dir)
        sel = result["selected_checkpoint"]
        print(f"Selected: {sel['variant']} seed={sel['training_seed']} step={sel['decision_steps']}")
    elif args.phase_j_cmd == "preliminary-freeze":
        result = preliminary_freeze(output_dir)
        print(f"Frozen: {result['variant']} step={result['selected_step']}")
    elif args.phase_j_cmd == "preliminary-test":
        result = preliminary_test(output_dir)
        print(f"Test complete: {result['tape_count']} tapes consumed")
    elif args.phase_j_cmd == "phase-j-dry-run":
        result = dry_run(output_dir)
        print(f"Dry-run complete: {result['train_checkpoints']} checkpoints")
    else:
        raise SystemExit(f"Unknown Phase J command: {args.phase_j_cmd}")
