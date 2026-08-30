"""Replay the frozen 200-tape development bank through the three baselines."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_preemption import (  # noqa: E402
    BeamMPCAllocator,
    GreedyPriorityAllocator,
    PreemptionController,
    SeniorLegacyMethodAllocator,
)
from execution_preemption.tapes import replay_tape, tape_sha256, validate_tape  # noqa: E402


TAPE_ROOT = ROOT / "experiments" / "dynamic_preemption" / "dev_v1" / "tapes"
DEFAULT_OUTPUT = (
    ROOT / "experiments" / "dynamic_preemption" / "dev_v1"
    / "baseline_replay_smoke.json"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _load_bank(tape_root: Path) -> tuple[dict, ...]:
    tapes: list[dict] = []
    for path in sorted(tape_root.rglob("*.json")):
        tape = json.loads(path.read_text(encoding="utf-8"))
        validate_tape(tape)
        tapes.append(tape)
    if len(tapes) != 200 or len({item["case_seed"] for item in tapes}) != 200:
        raise RuntimeError("baseline smoke requires the frozen 200 unique-seed dev tapes")
    return tuple(tapes)


def _run_allocator(allocator: object, bank: tuple[dict, ...]) -> dict[str, object]:
    decisions = Counter()
    selected = Counter()
    scenarios = Counter()
    request_count = 0
    trace_rows: list[dict[str, object]] = []
    for tape in bank:
        runtime, tape_decisions = replay_tape(
            tape,
            controller=PreemptionController(allocator),
        )
        runtime.validate_invariants()
        scenarios[str(tape["scenario_id"])] += 1
        for item in tape_decisions:
            decisions[item.decision.value] += 1
            if item.allocator_id is not None:
                request_count += 1
                if item.allocator_id != allocator.allocator_id:
                    raise RuntimeError("baseline allocator provenance mismatch")
            if item.selected_uav is not None:
                selected[item.selected_uav] += 1
        trace_rows.append({
            "scenario_id": tape["scenario_id"],
            "case_seed": tape["case_seed"],
            "runtime_sha256": runtime.state_sha256(),
            "decisions": [
                [item.event_id, item.decision.value, item.selected_uav, item.allocator_id]
                for item in tape_decisions
            ],
        })
    return {
        "allocator_id": allocator.allocator_id,
        "status": "PASS",
        "tape_count": len(bank),
        "decision_count": sum(decisions.values()),
        "allocation_request_count": request_count,
        "invariant_failures": 0,
        "decision_counts": dict(sorted(decisions.items())),
        "selected_uav_counts": dict(sorted(selected.items())),
        "scenario_run_counts": dict(sorted(scenarios.items())),
        "replay_sha256": hashlib.sha256(_canonical(trace_rows)).hexdigest(),
    }


def build_report(tape_root: Path = TAPE_ROOT) -> dict[str, object]:
    bank = _load_bank(tape_root)
    allocators = (
        SeniorLegacyMethodAllocator(),
        GreedyPriorityAllocator(),
        BeamMPCAllocator(horizon=3, beam_width=8),
    )
    results = [_run_allocator(item, bank) for item in allocators]
    bank_sha = hashlib.sha256(
        "\n".join(tape_sha256(item) for item in bank).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": 1,
        "contract_id": "execution-preemption-training-v1",
        "bank": "Dynamic-Preemption-Dev",
        "classification": "baseline_interface_and_safety_smoke_not_effectiveness_evidence",
        "status": "PASS",
        "tape_count": len(bank),
        "allocator_count": len(results),
        "allocator_tape_runs": len(bank) * len(results),
        "bank_canonical_sha256": bank_sha,
        "method_semantics": {
            "senior_legacy_method_v1": "adapted legacy first-legal lexical behaviour; not implementation equivalence",
            "greedy_priority_v1": "rule-priority arbitration then energy-margin/freshness greedy selection",
            "beam_mpc_v1": "deterministic horizon-3 width-8 planning over safe candidates and public pending-task forecast",
        },
        "results": results,
        "training_started": False,
        "validation_started": False,
        "freeze_started": False,
        "test_started": False,
        "hidden_evaluation_started": False,
        "model_effectiveness_evaluated": False,
        "checkpoint_selection": False,
    }


def write_report(path: Path, report: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(
        report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tape-root", type=Path, default=TAPE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.tape_root.resolve())
    output = write_report(args.output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "allocator_tape_runs": report["allocator_tape_runs"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
