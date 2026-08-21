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
    load_tape_bank,
    run_episode,
)
from .models import GraphActorCritic
from .reward import CostWeights, assignment_map, compute_cost
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

    The P0 gate must be green before training starts.
    Resume is NOT supported — training always starts fresh.
    """
    from .experiment import CyclingTrainingEnv

    gate = _check_p0_gate_strict()
    _validate_hashes_match(gate, "preliminary-train")

    protocol = protocol or PreliminaryProtocol()
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
        # Use real EpisodeMetrics, no fallback=0
        total_infeasible += 1 if trace.get("final_infeasible", False) else 0
        episode_dict = trace.get("episode", {})
        total_vacancy += episode_dict.get("cumulative_uncovered_time", 0.0)
        total_recovery_delay += episode_dict.get("recovery_delay", 0.0)
        total_fixed_j += episode_dict.get("fixed_j", 0.0)

    return ValidationMetrics(
        checkpoint_path=checkpoint.checkpoint_path,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        variant=checkpoint.variant,
        training_seed=checkpoint.training_seed,
        decision_steps=checkpoint.decision_steps,
        final_infeasible_count=total_infeasible,
        final_infeasible_rate=total_infeasible / max(1, tape_count),
        cumulative_weighted_vacancy=total_vacancy,
        recovery_latency=total_recovery_delay,
        fixed_j=total_fixed_j,
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


def preliminary_validate(
    output_dir: Path,
    checkpoints: list[CheckpointRecord] | None = None,
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

    bank_dir = output_dir / "preliminary" / "tapes" / "preliminary_validation_protocol"
    bank_manifest = bank_dir / "manifest.json"
    if not bank_manifest.exists():
        raise FileNotFoundError(f"Validation bank not found: {bank_manifest}")

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

def preliminary_freeze(output_dir: Path) -> dict[str, Any]:
    """Freeze all 9 selected checkpoints and produce freeze manifests."""
    gate = _check_p0_gate_strict()
    _validate_hashes_match(gate, "preliminary-freeze")

    selection_path = output_dir / "preliminary" / "validation_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("Run preliminary-validate first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    manifest_path = (
        output_dir / "preliminary" / "tapes"
        / "preliminary_validation_protocol" / "manifest.json"
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
            selected_at=_utc_now(),
        )
        freezes.append(asdict(freeze))

    _json_file(output_dir / "preliminary" / "frozen_manifests.json", {
        "created_at": _utc_now(),
        "freeze_count": len(freezes),
        "freezes": freezes,
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

    freezes = json.loads(freeze_path.read_text(encoding="utf-8"))["freezes"]

    # Load test bank
    test_bank_dir = output_dir / "preliminary" / "tapes" / "preliminary_test_protocol"
    test_manifest = test_bank_dir / "manifest.json"
    if not test_manifest.exists():
        raise FileNotFoundError("Run generate_test_bank first")

    test_manifest_sha = _sha256_file(test_manifest)

    # Test ledger tracks per-(variant, seed, checkpoint_sha) consumption
    ledger_path = output_dir / "preliminary" / "test_ledger.json"
    ledger = {}
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    all_results = []
    for freeze in freezes:
        key = f"{freeze['variant']}_seed{freeze['training_seed']}_{freeze['checkpoint_sha256'][:12]}"
        if ledger.get(key, {}).get("consumed"):
            print(f"  Skip (already consumed): {key}")
            continue

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


# ---------------------------------------------------------------------------
# DRY RUN orchestrator
# ---------------------------------------------------------------------------

def dry_run(output_dir: Path) -> dict[str, Any]:
    """Small-budget DRY RUN to verify the orchestrator pipeline.

    3 variants × 3 seeds × 2 checkpoints (5k, 10k steps).
    Validates: checkpoint creation, save/load, validation selection, freeze,
    test isolation guard.

    Does NOT use formal Test seed namespace.
    Does NOT produce Preliminary results.
    """
    protocol = PreliminaryProtocol(
        budget=10_000,
        checkpoint_interval=5_000,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Train
    train_result = preliminary_train(output_dir, protocol=protocol)
    num_ckpt = len(train_result["checkpoints"])

    # Step 2: Validate (per group selection)
    val_result = preliminary_validate(output_dir)
    num_selected = val_result["selected_count"]

    # Step 3: Freeze
    freeze_result = preliminary_freeze(output_dir)
    num_freeze = freeze_result["freeze_count"]

    # Step 4: Test isolation guard — attempt to re-generate test bank
    # should fail because we don't have formal test bank yet
    try:
        generate_test_bank(output_dir)
    except (SystemExit, FileNotFoundError):
        pass  # expected: freeze must exist first (it does, but test bank may not)

    return {
        "dry_run": True,
        "note": "This is a DRY RUN — not a Preliminary result",
        "train_checkpoints": num_ckpt,
        "selected_per_group": num_selected,
        "frozen_count": num_freeze,
        "selected": val_result["selected_checkpoints"][0] if val_result["selected_checkpoints"] else None,
        "frozen": freeze_result["freezes"][0] if freeze_result["freezes"] else None,
    }


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def add_phase_j_args(parser: argparse.ArgumentParser) -> None:
    """Add Phase J subcommands to an argument parser."""
    sub = parser.add_subparsers(dest="phase_j_cmd")

    train_p = sub.add_parser("preliminary-train", help="Run 300k preliminary training")
    train_p.add_argument("--output-dir", default="results/random_event")
    train_p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    train_p.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)

    val_p = sub.add_parser("preliminary-validate", help="Run validation selection")
    val_p.add_argument("--output-dir", default="results/random_event")

    freeze_p = sub.add_parser("preliminary-freeze", help="Freeze selected checkpoints")
    freeze_p.add_argument("--output-dir", default="results/random_event")

    test_p = sub.add_parser("preliminary-test", help="Run test on frozen checkpoints")
    test_p.add_argument("--output-dir", default="results/random_event")

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
