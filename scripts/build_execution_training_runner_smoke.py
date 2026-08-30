#!/usr/bin/env python3
"""Build a tiny real-gradient training smoke without authorizing formal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_preemption.contract import LEARNED_METHODS  # noqa: E402
from execution_preemption.training import TrainingRunConfig, train_run, verify_checkpoint  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT / "experiments" / "dynamic_preemption" / "dev_v1"
    / "training_runner_smoke.json"
)


def _run_once(method: str, output: Path) -> dict[str, object]:
    config = TrainingRunConfig(
        method_id=method,
        policy_seed=1101,
        uav_count=4,
        accepted_decision_steps=2,
        checkpoint_steps=(1, 2),
        rollout_steps=1,
        update_epochs=1,
        hidden_dim=8,
        relation_layers=1,
        formal=False,
    )
    report = train_run(config, output)
    verified = [
        verify_checkpoint(
            output / "checkpoints" / item["path"],
            expected_file_sha256=item["sha256"],
            expected_method_id=method,
            expected_policy_seed=1101,
            expected_uav_count=4,
            expected_step=item["step"],
        )
        for item in report["checkpoints"]
    ]
    return {
        "method_id": method,
        "model_family": report["model_family"],
        "rule_context_exposed": report["rule_context_exposed"],
        "accepted_decision_steps": report["accepted_decision_steps"],
        "optimizer_step_count": report["optimizer_step_count"],
        "checkpoint_steps": report["checkpoint_steps"],
        "checkpoint_file_sha_verified": all(item["status"] == "PASS" for item in verified),
        "model_state_sha256": [item["model_state_sha256"] for item in verified],
        "rng_state_sha256": [item["rng_state_sha256"] for item in verified],
        "status": "PASS",
    }


def build_report() -> dict[str, object]:
    results = []
    with tempfile.TemporaryDirectory(prefix="execution_training_smoke_") as temporary:
        root = Path(temporary)
        for method in LEARNED_METHODS:
            first = _run_once(method, root / f"{method}_a")
            second = _run_once(method, root / f"{method}_b")
            deterministic = (
                first["model_state_sha256"] == second["model_state_sha256"]
                and first["rng_state_sha256"] == second["rng_state_sha256"]
            )
            first["same_seed_state_determinism"] = deterministic
            if not deterministic:
                raise RuntimeError(f"same-seed state drift for {method}")
            results.append(first)
    return {
        "schema_version": 1,
        "status": "PASS",
        "classification": "tiny_training_runner_smoke_not_model_effectiveness_evidence",
        "learned_method_count": len(LEARNED_METHODS),
        "methods": results,
        "accepted_decision_steps_per_smoke_run": 2,
        "checkpoint_steps_per_smoke_run": [1, 2],
        "real_optimizer_updates": True,
        "fresh_temporary_output_per_run": True,
        "legacy_checkpoint_loaded": False,
        "old_campaign_reused": False,
        "formal_training_allowed": False,
        "formal_training_started": False,
        "validation_started": False,
        "freeze_started": False,
        "test_started": False,
        "hidden_evaluation_started": False,
        "checkpoint_selection": False,
        "model_effectiveness_evaluated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_report(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": "PASS", "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
