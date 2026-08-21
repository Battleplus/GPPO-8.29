"""Reproducible command-line experiment orchestration for random-event GPPO.

The implementation intentionally keeps orchestration separate from the model,
environment and metric definitions.  Every evaluated algorithm is replayed on
the exact same serialized :class:`EventTape`; raw decision/event traces and the
aggregate JSON therefore remain independently auditable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .baselines import (
    CurrentPendingExactPlannerPolicy,
    GraphPolicyAdapter,
    GreedyCostPolicy,
    MaskedRandomPolicy,
    MinLoadPolicy,
    NearestLegalPolicy,
)
from .environment import ActionSubmission, DecisionContext, RandomEventAllocationEnv
from .events import EventTape
from .legacy_adapter import LegacyMLPPPOPolicy
from .metrics import (
    EventMetricAccumulator,
    _mean,
    aggregate_episode,
    aggregate_tapes,
    paired_metric_report,
    stable_json_dumps,
    write_metrics_json,
)
from .models import GraphActorCritic
from .reward import compute_cost, compute_fixed_j_from_components
from .scheduler import (
    RandomEventScheduler,
    TimingProfile,
    UNSEEN_EVENT_WEIGHTS,
    UNSEEN_TIMING,
)
from .trainer import PPOConfig, PPOTrainer


PPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("results/random_event")
MODES = ("single", "sequential", "overlap", "burst")
PAIRED_METRICS = (
    "event_success_rate",
    "legal_coverage_rate",
    "recovery_delay",
    "cumulative_uncovered_time",
    "normalized_distance",
    "load_gap",
    "switch_count",
    "episode_return",
    "communication_bytes",
    "inference_latency_ms",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_path(value: str | Path) -> Path:
    """Resolve a CLI path relative to ``ppo_allocation`` rather than cwd."""

    path = Path(value)
    return path if path.is_absolute() else PPO_DIR / path


def _relative_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(PPO_DIR.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_file(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv(value: str) -> list[int]:
    result = [int(item) for item in _parse_csv(value)]
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PPO_DIR.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_metadata(command: Sequence[str] | None = None) -> dict[str, Any]:
    packages = {}
    for name in ("gymnasium", "numpy", "torch", "stable-baselines3", "sb3-contrib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_at_utc": _utc_now(),
        "command": list(command or sys.argv),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "torch_device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "git_commit": _git_commit(),
        "ppo_directory": str(PPO_DIR),
    }


def _make_tape(
    initial_seed: int,
    event_seed: int,
    mode: str,
    event_count: int,
    *,
    weights: Mapping[Any, float] | None = None,
    timing: Mapping[str, TimingProfile] | None = None,
) -> EventTape:
    env = RandomEventAllocationEnv(
        initial_seed=initial_seed,
        event_seed=event_seed,
        mode=mode,
        events_per_episode=event_count,
    )
    try:
        if weights is not None or timing is not None:
            env.scheduler = RandomEventScheduler(
                event_count=event_count,
                weights=weights,
                timing=timing,
            )
        env.reset(seed=initial_seed)
        assert env.event_tape is not None
        return env.event_tape
    finally:
        env.close()


def generate_tape_bank(
    output_dir: Path,
    *,
    modes: Sequence[str] = MODES,
    tapes_per_mode: int = 20,
    events_per_tape: int = 3,
    master_seed: int = 20260820,
    bank_name: str = "smoke",
) -> dict[str, Any]:
    """Generate a stable, replayable bank and return its manifest."""

    if tapes_per_mode <= 0 or events_per_tape <= 0:
        raise ValueError("tapes_per_mode and events_per_tape must be positive")
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise ValueError(f"unknown modes: {unknown}")
    bank_dir = output_dir / "tapes" / bank_name
    rng = random.Random(int(master_seed))
    entries: list[dict[str, Any]] = []
    for mode in modes:
        for index in range(tapes_per_mode):
            initial_seed = rng.getrandbits(31)
            event_seed = rng.getrandbits(63)
            tape = _make_tape(initial_seed, event_seed, mode, events_per_tape)
            tape_id = f"{mode}-{index:04d}-{_sha256_bytes(tape.to_bytes())[:12]}"
            path = bank_dir / mode / f"{tape_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tape.to_json(indent=2).encode("utf-8") + b"\n")
            entries.append(
                {
                    "tape_id": tape_id,
                    "mode": mode,
                    "initial_seed": initial_seed,
                    "event_seed": event_seed,
                    "event_count": len(tape.events),
                    "event_types": [event.event_type.value for event in tape.events],
                    "path": _relative_label(path),
                    "sha256": _sha256_file(path),
                    "canonical_tape_sha256": _sha256_bytes(tape.to_bytes()),
                }
            )
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "bank_name": bank_name,
        "master_seed": int(master_seed),
        "modes": list(modes),
        "tapes_per_mode": int(tapes_per_mode),
        "events_per_tape": int(events_per_tape),
        "tape_count": len(entries),
        "event_count": sum(item["event_count"] for item in entries),
        "entries": entries,
    }
    manifest_path = bank_dir / "manifest.json"
    _json_file(manifest_path, manifest)
    manifest["manifest_path"] = _relative_label(manifest_path)
    return manifest


def _expand_seed_range(spec: Mapping[str, Any]) -> list[int]:
    """Expand the frozen inclusive-start/count/stride seed notation."""

    start = int(spec["start"])
    count = int(spec["count"])
    stride = int(spec.get("stride", 1))
    if count <= 0 or stride <= 0:
        raise ValueError("seed range count and stride must be positive")
    return [start + index * stride for index in range(count)]


def _preliminary_seed_sets(seed_manifest: Mapping[str, Any]) -> dict[str, set[int]]:
    """Return train/validation/test seed sets and reject namespace leakage."""

    preliminary = seed_manifest["preliminary"]
    result: dict[str, set[int]] = {"train": set(), "validation": set(), "test": set()}
    train = preliminary["train"]
    for spec in train["instance_seeds_by_training_seed"].values():
        result["train"].update(_expand_seed_range(spec))
    for spec in train["event_seeds_by_training_seed"].values():
        result["train"].update(_expand_seed_range(spec))
    for item in preliminary["validation"]["modes"].values():
        result["validation"].update(_expand_seed_range(item["instance_seeds"]))
        result["validation"].update(_expand_seed_range(item["event_seeds"]))
    for item in preliminary["test"]["sets"].values():
        result["test"].update(_expand_seed_range(item["instance_seeds"]))
        result["test"].update(_expand_seed_range(item["event_seeds"]))
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = result[left] & result[right]
        if overlap:
            raise ValueError(f"seed namespace leakage between {left} and {right}: {min(overlap)}")
    return result


def _protocol_rows(
    seed_manifest: Mapping[str, Any], *, tier: str, split: str
) -> list[dict[str, Any]]:
    """Normalize preliminary/formal manifest layouts into generation rows."""

    section = seed_manifest[tier][split]
    rows: list[dict[str, Any]] = []
    if tier == "preliminary":
        groups = section["modes"] if split == "validation" else section["sets"]
        for label, item in groups.items():
            initial_seeds = _expand_seed_range(item["instance_seeds"])
            event_seeds = _expand_seed_range(item["event_seeds"])
            if len(initial_seeds) != len(event_seeds):
                raise ValueError(f"unpaired seed counts for {label}")
            rows.append({
                "set_name": label,
                "protocol": item.get("protocol", "P2_continuous_exogenous_tape"),
                "initial_seeds": initial_seeds,
                "event_seeds": event_seeds,
            })
        return rows

    groups = section["set_seed_ranges"]
    for label, item in groups.items():
        count = int(item["count"])
        stride = int(item.get("stride", 1))
        rows.append({
            "set_name": label,
            "protocol": item.get("protocol", "P2_continuous_exogenous_tape"),
            "initial_seeds": [int(item["instance_start"]) + i * stride for i in range(count)],
            "event_seeds": [int(item["event_start"]) + i * stride for i in range(count)],
        })
    return rows


def generate_protocol_bank(
    output_dir: Path,
    *,
    tier: str,
    split: str,
    seed_manifest_path: str | Path = "../configs/seed_manifest.json",
    protocol_path: str | Path = "../configs/random_event_protocol.json",
    events_per_tape: int = 5,
    limit_per_set: int | None = None,
) -> dict[str, Any]:
    """Generate a frozen Validation/Test bank from the committed seed manifest.

    ``Test-Unseen`` is a genuine held-out distribution profile: the four event
    meanings are unchanged, but their mixture and observation-delay ranges are
    shifted.  The underlying occurrence mode cycles over the same four timing
    contracts and is recorded per tape; no test data are used by training.
    """

    tier = str(tier).lower()
    split = str(split).lower()
    if tier not in {"preliminary", "formal"} or split not in {"validation", "test"}:
        raise ValueError("tier must be preliminary/formal and split validation/test")
    if events_per_tape <= 0:
        raise ValueError("events_per_tape must be positive")
    manifest_source = _relative_path(seed_manifest_path)
    protocol_source = _relative_path(protocol_path)
    seed_manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_source.read_text(encoding="utf-8"))
    configured_unseen_weights = protocol["event_scheduler"]["unseen_base_probabilities"]
    code_unseen_weights = {kind.value: value for kind, value in UNSEEN_EVENT_WEIGHTS.items()}
    if configured_unseen_weights != code_unseen_weights:
        raise ValueError("protocol unseen probabilities do not match scheduler constants")
    configured_unseen_delays = protocol["event_scheduler"][
        "unseen_detection_delay_seconds_by_mode"
    ]
    code_unseen_delays = {
        mode: [profile.observation_delay_min, profile.observation_delay_max]
        for mode, profile in UNSEEN_TIMING.items()
    }
    if configured_unseen_delays != code_unseen_delays:
        raise ValueError("protocol unseen detection delays do not match scheduler constants")
    if tier == "preliminary":
        _preliminary_seed_sets(seed_manifest)
    rows = _protocol_rows(seed_manifest, tier=tier, split=split)
    bank_name = f"{tier}_{split}_protocol"
    bank_dir = output_dir / "tapes" / bank_name
    mode_cycle = tuple(protocol["event_scheduler"].get(
        "unseen_mode_cycle", ["single", "sequential", "overlap", "burst"]
    ))
    if not mode_cycle or any(mode not in MODES for mode in mode_cycle):
        raise ValueError("unseen_mode_cycle must contain only supported modes")

    entries: list[dict[str, Any]] = []
    for row in rows:
        label = str(row["set_name"])
        is_unseen = label.lower().endswith("unseen")
        pairs = list(zip(row["initial_seeds"], row["event_seeds"]))
        if limit_per_set is not None:
            pairs = pairs[: int(limit_per_set)]
        for index, (initial_seed, event_seed) in enumerate(pairs):
            if is_unseen:
                mode = mode_cycle[index % len(mode_cycle)]
                weights = UNSEEN_EVENT_WEIGHTS
                timing = UNSEEN_TIMING
                distribution_profile = "unseen_shift_v1"
            else:
                normalized = label.lower().removeprefix("test-")
                mode = normalized if normalized in MODES else "sequential"
                weights = None
                timing = None
                distribution_profile = "in_distribution_v1"
            tape = _make_tape(
                initial_seed,
                event_seed,
                mode,
                events_per_tape,
                weights=weights,
                timing=timing,
            )
            tape_id = f"{label.lower()}-{index:04d}-{_sha256_bytes(tape.to_bytes())[:12]}"
            path = bank_dir / label / f"{tape_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tape.to_json(indent=2).encode("utf-8") + b"\n")
            entries.append({
                "tape_id": tape_id,
                "set_name": label,
                "mode": mode,
                "fairness_protocol": row["protocol"],
                "distribution_profile": distribution_profile,
                "initial_seed": initial_seed,
                "event_seed": event_seed,
                "event_count": len(tape.events),
                "event_types": [event.event_type.value for event in tape.events],
                "path": _relative_label(path),
                "sha256": _sha256_file(path),
                "canonical_tape_sha256": _sha256_bytes(tape.to_bytes()),
            })

    expected_count = sum(len(row["initial_seeds"]) for row in rows)
    complete = limit_per_set is None and len(entries) == expected_count
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "bank_name": bank_name,
        "tier": tier,
        "split": split,
        "intended_use": "checkpoint_selection" if split == "validation" else "final_evaluation_only",
        "checkpoint_selection": split == "validation",
        "reward_tuning": False,
        "complete_frozen_bank": complete,
        "expected_tape_count": expected_count,
        "tape_count": len(entries),
        "events_per_tape": int(events_per_tape),
        "seed_manifest": _relative_label(manifest_source),
        "seed_manifest_sha256": _sha256_file(manifest_source),
        "protocol": _relative_label(protocol_source),
        "protocol_sha256": _sha256_file(protocol_source),
        "unseen_definition": {
            "profile": "unseen_shift_v1",
            "same_event_semantics": True,
            "event_weights": {kind.value: value for kind, value in UNSEEN_EVENT_WEIGHTS.items()},
            "observation_delay_by_mode": {
                mode: [profile.observation_delay_min, profile.observation_delay_max]
                for mode, profile in UNSEEN_TIMING.items()
            },
            "mode_cycle": list(mode_cycle),
        },
        "entries": entries,
    }
    manifest_path = bank_dir / "manifest.json"
    _json_file(manifest_path, manifest)
    manifest["manifest_path"] = _relative_label(manifest_path)
    return manifest


def load_tape_bank(manifest_path: str | Path) -> tuple[dict[str, Any], list[tuple[str, EventTape]]]:
    path = _relative_path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tapes: list[tuple[str, EventTape]] = []
    for entry in manifest.get("entries", []):
        tape_path = _relative_path(entry["path"])
        observed_hash = _sha256_file(tape_path)
        if observed_hash != entry["sha256"]:
            raise ValueError(f"tape hash mismatch: {tape_path}")
        tape = EventTape.from_json(tape_path.read_bytes())
        canonical = _sha256_bytes(tape.to_bytes())
        if canonical != entry["canonical_tape_sha256"]:
            raise ValueError(f"canonical tape hash mismatch: {tape_path}")
        tapes.append((str(entry["tape_id"]), tape))
    if len(tapes) != int(manifest.get("tape_count", len(tapes))):
        raise ValueError("manifest tape_count does not match entries")
    return manifest, tapes


class CyclingTrainingEnv(RandomEventAllocationEnv):
    """Training env that deterministically changes seed and mode every reset.

    ``modes`` is the frozen preliminary train mode cycle (sequential, overlap,
    burst) loaded from ``seed_manifest.json`` by ``preliminary_train``; the
    generic evaluation ``MODES`` constant must NOT be passed here.

    ``max_resets`` enforces the frozen train-seed namespace: the manifest
    reserves ``episodes_per_training_seed`` episode indices per training seed.
    Once exhausted the env hard-FAILs instead of silently producing an
    unregistered training seed.
    """

    def __init__(
        self,
        *,
        seed: int,
        modes: Sequence[str],
        events_per_episode: int,
        max_resets: int = 300_000,
    ):
        self._training_seed = int(seed)
        self._training_modes = tuple(modes)
        self._max_resets = int(max_resets)
        self._reset_index = 0
        super().__init__(
            initial_seed=seed,
            event_seed=seed * 100_003 + 17,
            mode=self._training_modes[0],
            events_per_episode=events_per_episode,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        index = self._reset_index
        if index >= self._max_resets:
            raise RuntimeError(
                f"CyclingTrainingEnv seed {self._training_seed}: reset_index {index} "
                f"reached frozen reserved cap {self._max_resets}; refusing to leave "
                f"the frozen train-seed namespace"
            )
        self._reset_index += 1
        self.initial_seed = self._training_seed * 1_000_003 + index
        self.event_seed = self._training_seed * 10_000_019 + index
        self.mode = self._training_modes[index % len(self._training_modes)]
        self.supplied_tape = None
        return super().reset(seed=self.initial_seed, options=options)


def _load_frozen_train_modes() -> tuple[str, ...]:
    """Return the frozen preliminary train mode cycle from seed_manifest.json.

    The manifest is the single source of truth for the training mode sequence;
    the generic evaluation MODES constant must never be used for training.
    """
    manifest_path = Path(__file__).resolve().parents[2] / "configs" / "seed_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cycle = tuple(manifest["preliminary"]["train"]["mode_cycle"])
    if not cycle or any(mode not in MODES for mode in cycle):
        raise ValueError("frozen preliminary train mode_cycle is invalid")
    return cycle


def _load_frozen_train_episode_cap() -> int:
    """Return the frozen per-training-seed episode reservation from the manifest."""
    manifest_path = Path(__file__).resolve().parents[2] / "configs" / "seed_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return int(manifest["preliminary"]["train"]["episodes_per_training_seed"])


def train_variants(
    output_dir: Path,
    *,
    variants: Sequence[str],
    seeds: Sequence[int],
    timesteps: int,
    events_per_episode: int = 5,
    rollout_steps: int = 128,
    update_epochs: int = 4,
    minibatch_size: int = 64,
) -> dict[str, Any]:
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    model_dir = output_dir / "models"
    records: list[dict[str, Any]] = []
    train_modes = _load_frozen_train_modes()
    train_cap = _load_frozen_train_episode_cap()
    for variant in variants:
        for seed in seeds:
            env = CyclingTrainingEnv(
                seed=seed, modes=train_modes, events_per_episode=events_per_episode,
                max_resets=train_cap,
            )
            config = PPOConfig(
                rollout_steps=min(int(rollout_steps), int(timesteps)),
                update_epochs=int(update_epochs),
                minibatch_size=int(minibatch_size),
                seed=int(seed),
                device="cpu",
            )
            trainer = PPOTrainer(env=env, variant=variant, config=config)
            started = time.perf_counter()
            history = trainer.train(int(timesteps))
            elapsed = time.perf_counter() - started
            safe_variant = trainer.variant.lower().replace("-", "_")
            checkpoint = model_dir / f"{safe_variant}_seed{seed}_steps{timesteps}.pt"
            trainer.save(
                checkpoint,
                extra={
                    "experiment": "random_event_gppo",
                    "training_modes": list(train_modes),
                    "events_per_episode": events_per_episode,
                    "elapsed_seconds": elapsed,
                },
            )
            history_path = model_dir / f"{safe_variant}_seed{seed}_history.json"
            _json_file(history_path, history)
            records.append(
                {
                    "variant": trainer.variant,
                    "seed": int(seed),
                    "timesteps": int(timesteps),
                    "elapsed_seconds": elapsed,
                    "checkpoint": _relative_label(checkpoint),
                    "checkpoint_sha256": _sha256_file(checkpoint),
                    "history": _relative_label(history_path),
                    "updates": len(history),
                    "final_update": history[-1] if history else None,
                }
            )
            env.close()
    summary = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "device": "cpu",
        "variants": list(variants),
        "seeds": [int(seed) for seed in seeds],
        "timesteps": int(timesteps),
        "records": records,
    }
    _json_file(output_dir / "training_summary.json", summary)
    return summary


def _find_legacy_checkpoint() -> Path | None:
    candidates = sorted((PPO_DIR / "results" / "models").glob("**/maskable_ppo_uav_task_allocation.zip"))
    return candidates[-1] if candidates else None


def _load_policies(
    *,
    output_dir: Path,
    legacy_checkpoint: str | Path | None,
    gppo_checkpoints: Sequence[str | Path],
) -> tuple[list[Any], list[dict[str, Any]]]:
    policies: list[Any] = [
        ("Masked Random", "masked_random"),
        NearestLegalPolicy(),
        MinLoadPolicy(),
        GreedyCostPolicy(),
        CurrentPendingExactPlannerPolicy(),
    ]
    provenance: list[dict[str, Any]] = []
    legacy_path = _relative_path(legacy_checkpoint) if legacy_checkpoint else _find_legacy_checkpoint()
    if legacy_path is not None and legacy_path.exists():
        policies.insert(0, LegacyMLPPPOPolicy(legacy_path))
        provenance.append(
            {"algorithm": "MLP-PPO", "checkpoint": _relative_label(legacy_path), "sha256": _sha256_file(legacy_path)}
        )
    else:
        provenance.append({"algorithm": "MLP-PPO", "available": False, "reason": "checkpoint not found"})

    checkpoint_paths: list[Path] = []
    for value in gppo_checkpoints:
        path = _relative_path(value)
        if path.is_dir():
            checkpoint_paths.extend(sorted(path.glob("*.pt")))
        elif path.exists():
            checkpoint_paths.append(path)
    if not checkpoint_paths:
        checkpoint_paths = sorted((output_dir / "models").glob("*.pt"))
    for path in dict.fromkeys(item.resolve() for item in checkpoint_paths):
        model, metadata = GraphActorCritic.load(path, map_location="cpu")
        model.eval()
        variant = str(metadata.get("variant", "GPPO-Adaptive" if model.config.adaptive_gate else "GPPO-NoGate"))
        seed = metadata.get("ppo_config", {}).get("seed", "unknown")
        name = f"{variant} seed={seed}"
        policies.append(GraphPolicyAdapter(model=model, name=name))
        # A trainer checkpoint contains an optimizer state with tensors.  That
        # state is already covered by the checkpoint hash and should not be
        # duplicated into the human-readable evaluation summary.
        metadata_summary = {
            key: value for key, value in metadata.items()
            if key not in {"optimizer_state", "history"}
        }
        metadata_summary["history_records"] = len(metadata.get("history", []))
        provenance.append(
            {
                "algorithm": name,
                "checkpoint": _relative_label(path),
                "sha256": _sha256_file(path),
                "metadata": metadata_summary,
            }
        )
    return policies, provenance


def _policy_for_tape(policy: Any, tape_seed: int) -> Any:
    if isinstance(policy, tuple) and policy == ("Masked Random", "masked_random"):
        return MaskedRandomPolicy(seed=int(tape_seed))
    return policy


def _select_action(policy: Any, env: RandomEventAllocationEnv, graph) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter_ns()
    diagnostics: dict[str, Any] = {}
    if isinstance(policy, GraphPolicyAdapter):
        action, log_probability, value, model_diagnostics = policy.model.act(graph, deterministic=True)
        diagnostics = {
            "log_probability": log_probability,
            "predicted_value": value,
            **model_diagnostics,
        }
    else:
        action = policy.select_action(env, graph, deterministic=True)
    diagnostics["inference_latency_ms"] = (time.perf_counter_ns() - started) / 1_000_000.0
    return int(action), diagnostics


def _coverage(env: RandomEventAllocationEnv) -> float:
    legal = sum(
        region.assigned_uav >= 0 and env._valid_search_assign(region.assigned_uav, rid)
        for rid, region in env.regions.items()
    )
    return float(legal / max(1, len(env.regions)))


def run_episode(
    policy: Any,
    *,
    tape_id: str,
    tape: EventTape,
    algorithm: str,
    max_decisions: int = 100,
) -> tuple[Any, dict[str, Any]]:
    """Replay one policy on one immutable tape and capture every decision."""

    env = RandomEventAllocationEnv(
        initial_seed=tape.initial_seed,
        event_seed=tape.event_seed,
        mode=tape.mode,
        events_per_episode=len(tape.events),
        event_tape=tape,
        max_decisions=max_decisions,
    )
    graph, info = env.reset(seed=tape.initial_seed)
    episode_id = f"{algorithm}/{tape_id}"
    accumulators: dict[str, EventMetricAccumulator] = {}
    first_action_pending: set[str] = set()
    decision_rows: list[dict[str, Any]] = []
    affected_event_ids_this_decision: list[str] = []

    def register_events() -> None:
        for index, event in enumerate(tape.events):
            if event.event_id in env.event_records and event.event_id not in accumulators:
                accumulators[event.event_id] = EventMetricAccumulator(
                    tape_id=tape_id,
                    episode_id=episode_id,
                    event_id=event.event_id,
                    event_type=event.event_type.value,
                    event_index=index,
                )
                first_action_pending.add(event.event_id)

    register_events()
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        active_before = tuple(env.event_queue)
        pre_time = float(env.current_time)
        previous_repairs = int(env.repair_count)
        previous_comm_bytes = int(env.communication_bytes)
        previous_comm_count = int(env.communication_trigger_count)
        # Phase J: begin_decision → infer → versioned submit.  A stale
        # submission is not a decision: no trace row, reward, event metric or
        # budget step is consumed; the policy receives a fresh context.
        stale_attempts = 0
        while True:
            ctx = env.begin_decision() if hasattr(env, "begin_decision") else None
            if ctx is not None:
                graph = ctx.graph
            mask = graph.action_mask.cpu().numpy().astype(bool)
            action, diagnostics = _select_action(policy, env, graph)
            if ctx is not None:
                submission = ActionSubmission.from_decision(action, ctx)
                graph_after, reward, terminated, truncated, info = env.submit_action(submission)
            else:
                graph_after, reward, terminated, truncated, info = env.step(action)
            if info.get("stale_decision", False):
                stale_attempts += 1
                if stale_attempts > 100:
                    raise RuntimeError("more than 100 consecutive stale decisions in evaluation")
                continue
            break

        total_reward += float(reward)
        register_events()

        reward_trace = info.get("reward_trace", {})
        after = reward_trace.get("after", {})
        if not after:
            after = compute_cost(env).to_dict()
        repaired_delta = int(env.repair_count) - previous_repairs
        comm_bytes_delta = int(env.communication_bytes) - previous_comm_bytes
        comm_count_delta = int(env.communication_trigger_count) - previous_comm_count
        gate_values = diagnostics.get("gate_mean", {})
        invalid_probability = diagnostics.get("pre_mask_invalid_probability")
        decision_row = {
            "decision_index": int(env.decision_step),
            "algorithm": algorithm,
            "tape_id": tape_id,
            "graph_version_before": int(graph.graph_version),
            "graph_version_after": int(graph_after.graph_version),
            "simulation_time_before": pre_time,
            "simulation_time_after": float(env.current_time),
            "active_events_before": list(active_before),
            "new_events_after": list(info.get("new_events", [])),
            "raw_action": int(action),
            "repaired_action": info.get("repaired_action"),
            "invalid_action": bool(info.get("invalid_action", False)),
            "reward": float(reward),
            "reward_trace": reward_trace,
            "pending_regions_after": list(info.get("pending_regions", [])),
            "temporary_infeasible": bool(info.get("temporary_infeasible", False)),
            "final_infeasible": bool(info.get("final_infeasible", False)),
            "mask_rate": float(1.0 - np.mean(mask)),
            "diagnostics": diagnostics,
            "stale_submission_retries": stale_attempts,
            "affected_event_ids": list(active_before),
        }
        decision_rows.append(decision_row)

        # Attribute a recovery action to every event that was pending when the
        # action was selected.  Newly observed events start at the next action.
        # IMPORTANT: reward is recorded only ONCE per decision to avoid
        # double-counting when one action serves multiple events.
        affected_event_ids_this_decision.clear()
        for event_id in active_before:
            affected_event_ids_this_decision.append(event_id)

        reward_already_recorded = False
        for event_id in active_before:
            accumulator = accumulators[event_id]
            runtime = env.event_records[event_id]
            first = event_id in first_action_pending
            event_bytes = len(json.dumps(runtime.event.to_dict(), sort_keys=True).encode("utf-8")) if first else 0
            # Record reward only for the first affected event to avoid
            # double-counting.  Subsequent events get reward=None.
            event_reward = reward if not reward_already_recorded else None
            reward_already_recorded = True
            accumulator.record_decision(
                reward=event_reward,
                legal_coverage_rate=_coverage(env),
                weighted_uncovered=after.get("uncovered"),
                delta_time=env.decision_duration,
                normalized_distance=after.get("distance"),
                load_gap=after.get("load_gap"),
                switch_count=int(after.get("switches", 0)),
                repair_count=repaired_delta,
                temporary_infeasible=bool(info.get("temporary_infeasible", False)),
                final_infeasible=bool(info.get("final_infeasible", False)),
                inference_latency_ms=diagnostics.get("inference_latency_ms"),
                event_to_action_latency_ms=(max(0.0, pre_time - runtime.event.observed_at) * 1000.0 if first else None),
                communication_triggered=first,
                communication_bytes=event_bytes,
                communication_opportunities=1,
                pre_mask_invalid_probability=invalid_probability,
                mask_rate=1.0 - float(np.mean(mask)),
                gate_values=gate_values,
                predicted_value=diagnostics.get("predicted_value"),
                value_target=reward,
            )
            first_action_pending.discard(event_id)
        graph = graph_after

    final_cost = compute_cost(env)
    event_metrics = []
    for event in tape.events:
        runtime = env.event_records.get(event.event_id)
        accumulator = accumulators.get(event.event_id)
        if accumulator is None:
            accumulator = EventMetricAccumulator(
                tape_id=tape_id,
                episode_id=episode_id,
                event_id=event.event_id,
                event_type=event.event_type.value,
                event_index=len(event_metrics),
            )
        success = runtime is not None and runtime.status == "resolved"
        recovery_delay = None
        if runtime is not None and runtime.resolved_at is not None:
            recovery_delay = max(0.0, float(runtime.resolved_at - event.observed_at))
        # Frozen censoring rule: an event that never produced a decision (e.g.
        # it was never observed because an earlier event terminated the episode)
        # derives its fixed-J components from the final environment snapshot.
        # recovery_delay uses the explicit frozen 200s horizon penalty, never
        # an implicit None->0 coercion.  This keeps fixed J finite and the
        # checkpoint lexicographically rankable even with unresolved events.
        censored = accumulator._decision_count == 0
        if censored:
            fixed_j = compute_fixed_j_from_components(
                uncovered=final_cost.uncovered,
                distance=final_cost.distance,
                load_gap=final_cost.load_gap,
                switches=final_cost.switches,
                recovery_delay=200.0,
            )
        else:
            fixed_j = compute_fixed_j_from_components(
                uncovered=accumulator._uncovered[-1] if accumulator._uncovered else final_cost.uncovered,
                distance=_mean(accumulator._distance),
                load_gap=_mean(accumulator._load_gap),
                switches=float(accumulator._switch_count),
                recovery_delay=recovery_delay if recovery_delay is not None else 200.0,
            )
        event_metrics.append(
            accumulator.finalize(
                success=success,
                recovery_delay=recovery_delay,
                final_infeasible=(not success and bool(info.get("final_infeasible", False))),
                final_legal_coverage_rate=_coverage(env),
                final_weighted_uncovered=final_cost.uncovered,
                fixed_j=fixed_j,
            )
        )
    episode = aggregate_episode(event_metrics, algorithm=algorithm, tape_id=tape_id, episode_id=episode_id)
    trace = {
        "schema_version": 1,
        "algorithm": algorithm,
        "tape_id": tape_id,
        "tape_sha256": _sha256_bytes(tape.to_bytes()),
        "initial_seed": tape.initial_seed,
        "event_seed": tape.event_seed,
        "mode": tape.mode,
        "event_tape": tape.to_dict(),
        "decisions": decision_rows,
        "events": [item.to_dict() for item in event_metrics],
        "event_runtime": {key: value.to_dict() for key, value in env.event_records.items()},
        "episode": episode.to_dict(),
        "total_reward_check": total_reward,
        "episode_return_check": sum(row["reward"] for row in decision_rows),
        "reward_invariant": abs(total_reward - sum(row["reward"] for row in decision_rows)) < 1e-9,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_snapshot": env.snapshot(),
        "stale_submission_retry_count": sum(
            int(row.get("stale_submission_retries", 0)) for row in decision_rows
        ),
        "communication_counter_check": {
            "trigger_count": env.communication_trigger_count,
            "bytes": env.communication_bytes,
            "last_step_trigger_delta": comm_count_delta if decision_rows else 0,
            "last_step_bytes_delta": comm_bytes_delta if decision_rows else 0,
        },
    }
    env.close()
    return episode, trace


def evaluate_tape_bank(
    output_dir: Path,
    *,
    manifest_path: str | Path,
    legacy_checkpoint: str | Path | None = None,
    gppo_checkpoints: Sequence[str | Path] = (),
    limit: int | None = None,
    max_decisions: int = 100,
    bootstrap_resamples: int = 2_000,
) -> dict[str, Any]:
    manifest, tapes = load_tape_bank(manifest_path)
    if limit is not None:
        tapes = tapes[: int(limit)]
    policies, provenance = _load_policies(
        output_dir=output_dir,
        legacy_checkpoint=legacy_checkpoint,
        gppo_checkpoints=gppo_checkpoints,
    )
    raw_dir = output_dir / "logs" / str(manifest.get("bank_name", "bank"))
    episodes_by_algorithm: dict[str, list[Any]] = {}
    trace_index: list[dict[str, Any]] = []
    for policy_template in policies:
        template_name = policy_template[0] if isinstance(policy_template, tuple) else policy_template.name
        for tape_id, tape in tapes:
            policy = _policy_for_tape(policy_template, tape.event_seed)
            algorithm = policy.name
            episode, trace = run_episode(
                policy,
                tape_id=tape_id,
                tape=tape,
                algorithm=algorithm,
                max_decisions=max_decisions,
            )
            episodes_by_algorithm.setdefault(algorithm, []).append(episode)
            safe_name = algorithm.lower().replace(" ", "_").replace("=", "_").replace("-", "_")
            trace_path = raw_dir / safe_name / f"{tape_id}.json"
            _json_file(trace_path, trace)
            trace_index.append(
                {
                    "algorithm": algorithm,
                    "tape_id": tape_id,
                    "path": _relative_label(trace_path),
                    "sha256": _sha256_file(trace_path),
                }
            )

    algorithms = sorted(episodes_by_algorithm)
    summaries = {
        name: aggregate_tapes(records)
        for name, records in sorted(episodes_by_algorithm.items())
    }
    paired: dict[str, Any] = {}
    for left_index, left in enumerate(algorithms):
        for right in algorithms[left_index + 1 :]:
            key = f"{left} vs {right}"
            paired[key] = {
                metric: paired_metric_report(
                    episodes_by_algorithm[left],
                    episodes_by_algorithm[right],
                    metric,
                    n_resamples=bootstrap_resamples,
                    seed=20260820,
                )
                for metric in PAIRED_METRICS
            }
    result = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "paired_design": True,
        "manifest": _relative_label(_relative_path(manifest_path)),
        "manifest_sha256": _sha256_file(_relative_path(manifest_path)),
        "tape_count": len(tapes),
        "algorithms": algorithms,
        "algorithm_provenance": provenance,
        "summaries": summaries,
        "paired_statistics": paired,
        "episode_records": {
            name: [record.to_dict() for record in records]
            for name, records in sorted(episodes_by_algorithm.items())
        },
        "raw_trace_index": trace_index,
    }
    _json_file(output_dir / "evaluation_summary.json", result)
    _json_file(output_dir / "raw_trace_index.json", trace_index)
    return result


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_small_smoke:
        if args.tapes_per_mode < 20:
            raise ValueError("smoke requires at least 20 tapes per mode (use --allow-small-smoke only for developer checks)")
        if args.events_per_tape < 3:
            raise ValueError("smoke requires at least 3 events per tape (use --allow-small-smoke only for developer checks)")
    output_dir = _relative_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = generate_tape_bank(
        output_dir,
        modes=MODES,
        tapes_per_mode=args.tapes_per_mode,
        events_per_tape=args.events_per_tape,
        master_seed=args.master_seed,
        bank_name=args.bank_name,
    )
    # Replay every generated case with a deterministic legal baseline.  This
    # turns tape generation into an environment smoke test, not a file-only test.
    _, tapes = load_tape_bank(manifest["manifest_path"])
    replay_rows = []
    for tape_id, tape in tapes:
        episode, _ = run_episode(
            NearestLegalPolicy(), tape_id=tape_id, tape=tape,
            algorithm="Nearest Legal", max_decisions=args.max_decisions,
        )
        replay_rows.append(episode.to_dict())
    smoke = {
        "manifest": manifest,
        "replay": aggregate_tapes(replay_rows),
        "replayed_tape_count": len(replay_rows),
    }
    _json_file(output_dir / "smoke_summary.json", smoke)
    _json_file(output_dir / "environment_metadata.json", environment_metadata())
    return smoke


def _check_p0_gate() -> None:
    """Refuse training unless the machine gate and current tree are consistent.

    The gate records ``attested_source_commit_sha`` — the commit where all
    protected source/config/test files were verified.  Training is allowed if
    and only if:

    1. ``training_allowed`` is true in the gate JSON.
    2. ``current HEAD`` is an ancestor-or-equal of ``attested_source_commit_sha``
       OR ``attested_source_commit_sha`` is an ancestor-or-equal of
       ``current HEAD`` (i.e.  the attested commit is reachable).
    3. All protected source/config/test file hashes match the attested values.
    4. ``source_tree_hash``, ``protocol_sha256``, ``seed_manifest_sha256`` match.

    Evidence-only commits (gate JSON, smoke evidence, handoff reports) are
    allowed on top of the attested source commit without invalidating it.
    """

    gate_path = PPO_DIR.parent / "handoff" / "P0_GATE.json"
    root = PPO_DIR.parent
    if not gate_path.exists():
        raise SystemExit("P0 gate missing. Run `python scripts/build_p0_gate.py` before training.")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("generated_by") != "scripts/build_p0_gate.py":
        raise SystemExit("P0 gate was not machine-generated; refusing to train.")
    if not gate.get("training_allowed"):
        violations = "; ".join(gate.get("violations", []))
        raise SystemExit(f"P0 gate is RED. Violations: {violations}")

    # --- Commit attestation ---
    # The gate stores ``attested_source_commit_sha`` (the commit where all
    # protected files were verified).  Current HEAD must contain that commit
    # (i.e.  the attested source is reachable from HEAD).
    attested = gate.get("attested_source_commit_sha") or gate.get("git_commit_sha")
    if not attested:
        raise SystemExit("P0 gate has no attested_source_commit_sha; rerun the gate.")

    try:
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Cannot determine current git HEAD: {exc}") from exc

    # Verify the attested commit is an ancestor of (or equal to) current HEAD.
    # This allows evidence-only commits on top of the attested source.
    if current_commit != attested:
        try:
            merge_base = subprocess.check_output(
                ["git", "merge-base", "--is-ancestor", attested, current_commit],
                cwd=root, text=True,
            )
            # If merge-base exits 0, attested is ancestor of HEAD — OK.
        except (OSError, subprocess.CalledProcessError):
            raise SystemExit(
                f"P0 gate attested commit {attested[:12]} is not an ancestor of "
                f"current HEAD {current_commit[:12]}; protected source may have changed; "
                "rerun the complete gate."
            )

    # --- Source/config/test hash verification ---
    frozen = gate.get("source_hashes", {})
    frozen_sources = frozen.get("source", {})
    if not frozen_sources:
        raise SystemExit("P0 gate has no protected source hashes")
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", *sorted(frozen_sources)],
            cwd=root, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Cannot verify protected working tree: {exc}") from exc
    if status:
        raise SystemExit("P0 gate protected working tree is dirty; rerun evidence after source commit.")

    attested_blob_sources = {}
    for relative, expected in frozen_sources.items():
        try:
            blob = subprocess.check_output(
                ["git", "show", f"{attested}:{relative}"], cwd=root
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"P0 gate attested source is missing: {relative}") from exc
        attested_blob_sources[relative] = hashlib.sha256(blob).hexdigest()
    if attested_blob_sources != frozen_sources:
        raise SystemExit("P0 gate attested Git tree hash mismatch")
    current_tree_hash = hashlib.sha256(
        "".join(f"{key}:{value}\n" for key, value in sorted(attested_blob_sources.items())).encode("utf-8")
    ).hexdigest()
    if current_tree_hash != gate.get("source_tree_hash"):
        raise SystemExit("P0 gate source_tree_hash mismatch; protected source changed after attestation.")

    for relative, key, field in (
        ("configs/random_event_protocol.json", "protocol", "protocol_sha256"),
        ("configs/seed_manifest.json", "seed_manifest", "seed_manifest_sha256"),
    ):
        try:
            current_hash = hashlib.sha256(
                subprocess.check_output(["git", "show", f"{attested}:{relative}"], cwd=root)
            ).hexdigest()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"P0 gate {key} is missing from attested tree") from exc
        expected = gate.get(field) or frozen.get(key)
        if current_hash != expected:
            raise SystemExit(f"P0 gate {key} hash mismatch; configuration changed after attestation.")


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    # Phase H: training entry point must read the P0 gate and refuse on red.
    _check_p0_gate()
    output_dir = _relative_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = train_variants(
        output_dir,
        variants=_parse_csv(args.variants),
        seeds=_parse_int_csv(args.seeds),
        timesteps=args.timesteps,
        events_per_episode=args.events_per_episode,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
    )
    _json_file(output_dir / "environment_metadata.json", environment_metadata())
    return result


def run_protocol_bank(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _relative_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.tier == "preliminary" and args.split == "test":
        # Official Preliminary Test is Phase J guarded. This closes the old
        # protocol-bank bypass at the public CLI boundary.
        from .phase_j import generate_test_bank
        result = generate_test_bank(output_dir)
    else:
        result = generate_protocol_bank(
        output_dir,
        tier=args.tier,
        split=args.split,
        seed_manifest_path=args.seed_manifest,
        protocol_path=args.protocol,
        events_per_tape=args.events_per_tape,
            limit_per_set=args.limit_per_set,
        )
    _json_file(output_dir / "environment_metadata.json", environment_metadata())
    return result


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _relative_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_tape_bank(
        output_dir,
        manifest_path=args.manifest,
        legacy_checkpoint=args.legacy_checkpoint,
        gppo_checkpoints=args.gppo_checkpoint or (),
        limit=args.limit,
        max_decisions=args.max_decisions,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    _json_file(output_dir / "environment_metadata.json", environment_metadata())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Random-event GPPO reproducible experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="generate and replay the four-mode smoke tape bank")
    smoke.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    smoke.add_argument("--bank-name", default="smoke")
    smoke.add_argument("--tapes-per-mode", type=int, default=20)
    smoke.add_argument("--events-per-tape", type=int, default=3)
    smoke.add_argument("--master-seed", type=int, default=20260820)
    smoke.add_argument("--max-decisions", type=int, default=100)
    smoke.add_argument("--allow-small-smoke", action="store_true", help=argparse.SUPPRESS)
    smoke.set_defaults(func=run_smoke)

    protocol_bank = subparsers.add_parser(
        "protocol-bank",
        help="generate frozen Validation/Test banks from configs/seed_manifest.json",
    )
    protocol_bank.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    protocol_bank.add_argument("--tier", choices=("preliminary", "formal"), default="preliminary")
    protocol_bank.add_argument("--split", choices=("validation", "test"), required=True)
    protocol_bank.add_argument("--seed-manifest", default="../configs/seed_manifest.json")
    protocol_bank.add_argument("--protocol", default="../configs/random_event_protocol.json")
    protocol_bank.add_argument("--events-per-tape", type=int, default=5)
    protocol_bank.add_argument("--limit-per-set", type=int, default=None, help=argparse.SUPPRESS)
    protocol_bank.set_defaults(func=run_protocol_bank)

    train = subparsers.add_parser("train", help="train GPPO-NoGate and/or GPPO-Adaptive on CPU")
    train.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    train.add_argument("--variants", default="PPO-MLP,GPPO-NoGate,GPPO-Adaptive")
    train.add_argument("--seeds", default="1101,2202,3303")
    train.add_argument("--timesteps", type=int, default=2000)
    train.add_argument("--events-per-episode", type=int, default=5)
    train.add_argument("--rollout-steps", type=int, default=128)
    train.add_argument("--update-epochs", type=int, default=4)
    train.add_argument("--minibatch-size", type=int, default=64)
    train.set_defaults(func=run_train)

    evaluate = subparsers.add_parser("evaluate", help="paired evaluation on a frozen tape manifest")
    evaluate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    evaluate.add_argument("--manifest", default="results/random_event/tapes/smoke/manifest.json")
    evaluate.add_argument("--legacy-checkpoint", default=None)
    evaluate.add_argument("--gppo-checkpoint", action="append", default=[])
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--max-decisions", type=int, default=100)
    evaluate.add_argument("--bootstrap-resamples", type=int, default=2000)
    evaluate.set_defaults(func=run_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    tape_count = result.get("tape_count")
    if tape_count is None and isinstance(result.get("manifest"), Mapping):
        tape_count = result["manifest"].get("tape_count")
    concise = {
        "command": args.command,
        "output_dir": _relative_label(_relative_path(args.output_dir)),
        "tape_count": tape_count,
        "algorithms": result.get("algorithms"),
    }
    print(stable_json_dumps(concise, indent=2))
    return 0


__all__ = [
    "CyclingTrainingEnv",
    "build_parser",
    "environment_metadata",
    "evaluate_tape_bank",
    "generate_protocol_bank",
    "generate_tape_bank",
    "load_tape_bank",
    "main",
    "run_episode",
    "train_variants",
]
