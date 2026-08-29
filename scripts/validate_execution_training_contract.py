"""Validate the frozen Execution-Preemption V1 training contract.

This is a pre-training contract smoke only.  It does not create an environment,
load a checkpoint, start training, or run validation/test/hidden evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_preemption.contract import load_training_contract  # noqa: E402
from execution_preemption.metrics import (  # noqa: E402
    ExecutionMetricAccumulator,
    evaluate_acceptance,
)
from execution_preemption.reward import TransitionSignals, compute_transition_reward  # noqa: E402


DEFAULT_CONTRACT = ROOT / "configs" / "execution_training_contract_v1.json"
DEFAULT_OUTPUT = (
    ROOT / "experiments" / "dynamic_preemption" / "dev_v1" / "training_contract_smoke.json"
)


def build_report(contract_path: Path) -> dict[str, object]:
    contract = load_training_contract(contract_path)
    contract_record = contract.to_dict()
    try:
        contract_record["path"] = contract.path.relative_to(ROOT).as_posix()
    except ValueError:
        contract_record["path"] = contract.path.name
    reward = compute_transition_reward(TransitionSignals(
        weighted_progress_gain=0.50,
        urgent_deadline_miss_rate=0.10,
        weighted_vacancy_time=0.20,
        progress_loss=0.10,
        starvation_exposure=0.10,
        switch_time=0.20,
        energy_consumed=0.10,
        normalized_distance=0.40,
        load_gap=0.20,
    ))

    accumulator = ExecutionMetricAccumulator("contract_smoke_candidate", "contract-smoke", 16)
    accumulator.record_transition(
        TransitionSignals(weighted_vacancy_time=0.4, load_gap=0.2),
        inference_latency_ms=2.0,
        preemption_response_latency=1.0,
    )
    accumulator.record_event(urgent=True, p0=True, p0_handled=True)
    accumulator.record_displacement(resumed=True, recovery_latency=1.0)
    candidate = replace(
        accumulator.finalize(),
        urgent_deadline_miss_rate=0.17,
        cumulative_weighted_vacancy=80.0,
        normal_task_recovery_rate=0.95,
    )
    baseline = replace(
        candidate,
        algorithm_id="senior_legacy_method_v1",
        urgent_deadline_miss_rate=0.20,
        cumulative_weighted_vacancy=100.0,
    )
    acceptance = evaluate_acceptance(candidate, baseline)
    if acceptance.status != "PASS":
        raise RuntimeError(f"acceptance smoke failed: {acceptance.violations}")

    return {
        "schema_version": 1,
        "status": "PASS",
        "classification": "training_precondition_contract_smoke_not_model_evidence",
        "contract": contract_record,
        "reward_smoke": reward.to_dict(),
        "acceptance_smoke": acceptance.to_dict(),
        "legacy_checkpoint_compatible": False,
        "legacy_evidence_reusable": False,
        "source_bound_launch_gate_created": False,
        "training_allowed": False,
        "training_started": False,
        "validation_started": False,
        "freeze_started": False,
        "test_started": False,
        "hidden_evaluation_started": False,
        "checkpoint_selection": False,
    }


def write_report(path: Path, report: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = write_report(args.output.resolve(), build_report(args.contract.resolve()))
    print(json.dumps({"status": "PASS", "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
