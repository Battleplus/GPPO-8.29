"""Replay the frozen development bank through deterministic allocators."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution_preemption import (  # noqa: E402
    FirstAvailableAllocator,
    MaxEnergyMarginAllocator,
    PreemptionController,
)
from execution_preemption.tapes import replay_tape, tape_sha256, validate_tape  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    path.write_bytes((payload + "\n").encode("utf-8"))


def _load_bank(tape_root: Path) -> tuple[tuple[Path, dict], ...]:
    items: list[tuple[Path, dict]] = []
    for path in sorted(tape_root.rglob("*.json")):
        tape = json.loads(path.read_text(encoding="utf-8"))
        validate_tape(tape)
        items.append((path, tape))
    if len(items) != 200:
        raise RuntimeError(f"expected frozen 200-tape bank, got {len(items)}")
    if len({tape["case_seed"] for _, tape in items}) != 200:
        raise RuntimeError("development seeds are not unique")
    return tuple(items)


def run(tape_root: Path, output_path: Path) -> dict[str, object]:
    bank = _load_bank(tape_root)
    allocators = (FirstAvailableAllocator(), MaxEnergyMarginAllocator())
    results: list[dict[str, object]] = []
    input_hashes = [tape_sha256(tape) for _, tape in bank]
    bank_sha256 = hashlib.sha256("\n".join(input_hashes).encode("ascii")).hexdigest()

    for allocator in allocators:
        decisions = Counter()
        selected_uavs = Counter()
        scenario_runs = Counter()
        allocation_requests = 0
        invariant_failures = 0
        for _, tape in bank:
            try:
                runtime, tape_decisions = replay_tape(
                    tape,
                    controller=PreemptionController(allocator),
                )
                runtime.validate_invariants()
            except Exception:
                invariant_failures += 1
                raise
            scenario_runs[str(tape["scenario_id"])] += 1
            for decision in tape_decisions:
                decisions[decision.decision.value] += 1
                if decision.allocator_id is not None:
                    allocation_requests += 1
                    if decision.allocator_id != allocator.allocator_id:
                        raise RuntimeError("allocator provenance mismatch")
                if decision.selected_uav is not None:
                    selected_uavs[decision.selected_uav] += 1
        results.append({
            "allocator_id": allocator.allocator_id,
            "status": "PASS" if invariant_failures == 0 else "FAIL",
            "tape_count": len(bank),
            "decision_count": sum(decisions.values()),
            "allocation_request_count": allocation_requests,
            "invariant_failures": invariant_failures,
            "decision_counts": dict(sorted(decisions.items())),
            "selected_uav_counts": dict(sorted(selected_uavs.items())),
            "scenario_run_counts": dict(sorted(scenario_runs.items())),
        })

    summary = {
        "schema_version": 1,
        "contract_id": "execution-preemption-v1",
        "bank": "Dynamic-Preemption-Dev",
        "classification": "development_interface_validation_not_effectiveness_evidence",
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL",
        "tape_count": len(bank),
        "allocator_count": len(results),
        "allocator_tape_runs": len(bank) * len(results),
        "bank_canonical_sha256": bank_sha256,
        "training_started": False,
        "model_effectiveness_evaluated": False,
        "checkpoint_selection": False,
        "results": results,
    }
    _write_json(output_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tape-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "dynamic_preemption" / "dev_v1" / "tapes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT / "experiments" / "dynamic_preemption" / "dev_v1"
            / "allocator_replay_summary.json"
        ),
    )
    args = parser.parse_args()
    summary = run(args.tape_root.resolve(), args.output.resolve())
    print(json.dumps({
        "status": summary["status"],
        "tape_count": summary["tape_count"],
        "allocator_count": summary["allocator_count"],
        "allocator_tape_runs": summary["allocator_tape_runs"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
