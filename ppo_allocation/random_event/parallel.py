"""Controlled parallel scheduler for independent minimum-validation runs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .progress import read_progress, write_progress

FORMAL_VARIANTS = ("PPO-MLP", "GPPO-Adaptive")
FORMAL_SEEDS = (1101, 2202, 3303)
FORMAL_BUDGET = 50_000
FORMAL_CHECKPOINTS = (25_000, 50_000)
FORMAL_RUN_COUNT = len(FORMAL_VARIANTS) * len(FORMAL_SEEDS)
FORMAL_CHECKPOINT_COUNT = FORMAL_RUN_COUNT * len(FORMAL_CHECKPOINTS)


@dataclass(frozen=True)
class WorkerSpec:
    variant: str
    seed: int
    output_dir: Path

    @property
    def key(self) -> str:
        return f"{self.variant}_{self.seed}"

    @property
    def progress_path(self) -> Path:
        return self.output_dir / "progress" / "live_progress.json"

    @property
    def stdout_path(self) -> Path:
        return self.output_dir / "logs" / "stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self.output_dir / "logs" / "stderr.log"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "models"


@dataclass
class WorkerResult:
    key: str
    variant: str
    seed: int
    status: str
    pid: int | None = None
    returncode: int | None = None
    error: str | None = None


def worker_specs(root: Path) -> list[WorkerSpec]:
    return [
        WorkerSpec(variant, seed, root / variant / f"seed_{seed}")
        for seed in FORMAL_SEEDS
        for variant in FORMAL_VARIANTS
    ]


def seed_batches(root: Path, *, max_workers: int = 3) -> list[list[WorkerSpec]]:
    """Return three seed batches, each containing both formal model variants."""
    if max_workers not in (1, 2, 3):
        raise ValueError("max_workers must be 1, 2 or 3")
    specs = worker_specs(root)
    return [[spec for spec in specs if spec.seed == seed] for seed in FORMAL_SEEDS]


def validate_worker_isolation(specs: Sequence[WorkerSpec]) -> None:
    attrs = {
        "output_dir": [str(s.output_dir.resolve()) for s in specs],
        "progress": [str(s.progress_path.resolve()) for s in specs],
        "stdout": [str(s.stdout_path.resolve()) for s in specs],
        "stderr": [str(s.stderr_path.resolve()) for s in specs],
        "checkpoint": [str(s.checkpoint_dir.resolve()) for s in specs],
    }
    for label, values in attrs.items():
        if len(values) != len(set(values)):
            raise ValueError(f"worker {label} paths are not unique")


def aggregate_progress(specs: Sequence[WorkerSpec], *, worker_limit: int = 3,
                       target_steps_per_run: int = FORMAL_BUDGET) -> dict[str, Any]:
    progress = [read_progress(spec.progress_path) for spec in specs]
    steps = [int(item.get("total_steps", 0)) if item else 0 for item in progress]
    target = len(specs) * int(target_steps_per_run)
    running = sum(1 for item in progress if item and item.get("status") == "running")
    done = sum(1 for item in progress if item and item.get("status") == "done")
    rates = [float(item["steps_per_second"]) for item in progress if item and item.get("steps_per_second")]
    return {
        "worker_limit": worker_limit,
        "total_steps": sum(steps),
        "target_steps": target,
        "campaign_fraction": sum(steps) / target if target else 0.0,
        "running_workers": running,
        "completed_runs": done,
        "checkpoint_count": sum(int(item.get("checkpoint_count", 0)) for item in progress if item),
        "aggregate_steps_per_second": sum(rates),
        "workers": [
            {"variant": spec.variant, "seed": spec.seed, **(item or {"total_steps": 0, "status": "pending"})}
            for spec, item in zip(specs, progress)
        ],
        "target_steps_per_run": int(target_steps_per_run),
    }


def format_progress(snapshot: dict[str, Any]) -> str:
    lines = [
        f"Campaign: {snapshot['campaign_fraction']:.1%}",
        f"Running workers: {snapshot['running_workers']}",
        f"Completed runs: {snapshot['completed_runs']}/{FORMAL_RUN_COUNT}",
        f"Checkpoints: {snapshot['checkpoint_count']}/{FORMAL_CHECKPOINT_COUNT}",
        f"Aggregate steps/s: {snapshot['aggregate_steps_per_second']:.2f}",
    ]
    for worker in snapshot["workers"]:
        target = int(worker.get("target_steps", FORMAL_BUDGET))
        steps = int(worker.get("total_steps", 0))
        lines.append(f"{worker['variant']} / {worker['seed']}: {steps}/{target} ({worker.get('status', 'pending')})")
    return "\n".join(lines)


def build_worker_command(spec: WorkerSpec, repo_root: Path, *, budget: int = FORMAL_BUDGET,
                         checkpoint_interval: int = FORMAL_CHECKPOINTS[0], formal: bool = True) -> list[str]:
    """Construct one isolated worker command; no resume or shared state."""
    # Keep every Python argument ASCII/relative where possible; the caller's
    # cwd is the repository root and each worker output path is passed only to
    # the worker process created by this controller.
    command = [sys.executable, "scripts/run_minimum_validation_worker.py", spec.variant,
               str(spec.seed), str(spec.output_dir)]
    if not formal:
        command.extend(["--developer", "--budget", str(budget),
                        "--checkpoint-interval", str(checkpoint_interval)])
    return command


def run_controlled_parallel(root: Path, *, max_workers: int = 3, dry_run: bool = False,
                            protocol: Any | None = None, formal: bool = True) -> dict[str, Any]:
    """Run independent workers in seed batches, never silently restarting failures."""
    if max_workers not in (1, 2, 3):
        raise ValueError("max_workers must be 1, 2 or 3")
    specs = worker_specs(root)
    validate_worker_isolation(specs)
    budget = int(getattr(protocol, "budget", FORMAL_BUDGET))
    checkpoint_interval = int(getattr(protocol, "checkpoint_interval", FORMAL_CHECKPOINTS[0]))
    if dry_run:
        return {"status": "dry_run", "expected_runs": FORMAL_RUN_COUNT,
                "expected_checkpoints": FORMAL_CHECKPOINT_COUNT,
                "max_workers": max_workers}
    if formal and (budget != FORMAL_BUDGET or checkpoint_interval != FORMAL_CHECKPOINTS[0]):
        raise ValueError("formal parallel controller received a protocol override")
    root.mkdir(parents=True, exist_ok=True)
    results: list[WorkerResult] = []
    repo_root = Path(__file__).resolve().parents[2]
    # The recommended schedule is one seed batch at a time. Within a batch,
    # at most ``max_workers`` independent model processes are alive.
    for batch in seed_batches(root, max_workers=max_workers):
        for chunk_start in range(0, len(batch), max_workers):
            active: list[tuple[WorkerSpec, subprocess.Popen[str]]] = []
            for spec in batch[chunk_start:chunk_start + max_workers]:
                for path in (spec.output_dir, spec.checkpoint_dir, spec.stdout_path.parent, spec.progress_path.parent):
                    path.mkdir(parents=True, exist_ok=True)
                if any(spec.checkpoint_dir.glob("*.pt")):
                    raise RuntimeError(f"refusing to overwrite existing checkpoint directory: {spec.checkpoint_dir}")
                stdout = spec.stdout_path.open("w", encoding="utf-8")
                stderr = spec.stderr_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    build_worker_command(spec, repo_root, budget=budget,
                                         checkpoint_interval=checkpoint_interval, formal=formal),
                    cwd=repo_root, stdout=stdout, stderr=stderr, text=True,
                )
                stdout.close(); stderr.close()
                results.append(WorkerResult(spec.key, spec.variant, spec.seed, "running", process.pid))
                active.append((spec, process))
            for spec, process in active:
                code = process.wait()
                result = next(item for item in results if item.key == spec.key)
                result.returncode = code
                result.status = "pass" if code == 0 else "fail"
                if code != 0:
                    raise RuntimeError(f"worker failed: {spec.key} (exit {code}); no automatic restart")
    snapshot = aggregate_progress(specs, worker_limit=max_workers,
                                  target_steps_per_run=budget)
    write_progress(root / "aggregate_progress.json", snapshot)
    return {"status": "complete", "max_workers": max_workers, "results": [result.__dict__ for result in results], "aggregate": snapshot}


__all__ = ["FORMAL_VARIANTS", "FORMAL_SEEDS", "FORMAL_BUDGET", "FORMAL_CHECKPOINTS",
           "FORMAL_RUN_COUNT", "FORMAL_CHECKPOINT_COUNT", "WorkerSpec", "aggregate_progress",
           "build_worker_command", "format_progress", "run_controlled_parallel",
           "validate_worker_isolation", "worker_specs"]
