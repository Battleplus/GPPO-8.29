"""Prove deferred policy transactions match direct deterministic replay."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_preemption import (  # noqa: E402
    FirstAvailableAllocator,
    MaxEnergyMarginAllocator,
    PreemptionController,
)
from execution_preemption.models import EventPriority, RuntimeEvent, RuntimeEventType  # noqa: E402
from execution_preemption.tapes import replay_tape, runtime_from_tape, validate_tape  # noqa: E402


DEFAULT_TAPE_ROOT = ROOT / "experiments" / "dynamic_preemption" / "dev_v1" / "tapes"
DEFAULT_OUTPUT = (
    ROOT / "experiments" / "dynamic_preemption" / "dev_v1"
    / "deferred_transaction_parity.json"
)


def _runtime_event(item: dict) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=item["event_id"],
        event_type=RuntimeEventType(item["event_type"]),
        priority=EventPriority[item["priority"]],
        occurred_at=float(item["occurred_at"]),
        received_at=float(item["received_at"]),
        task_id=item.get("task_id"),
        uav_id=item.get("uav_id"),
        task_priority=int(item.get("task_priority", 0)),
        deadline=item.get("deadline"),
        confidence=float(item.get("confidence", 1.0)),
        payload=item.get("payload", {}),
    )


def _batches(tape: dict) -> tuple[tuple[RuntimeEvent, ...], ...]:
    grouped: list[list[RuntimeEvent]] = []
    indexes: dict[str, int] = {}
    for item in tape["events"]:
        batch_id = str(item["batch_id"])
        if batch_id not in indexes:
            indexes[batch_id] = len(grouped)
            grouped.append([])
        grouped[indexes[batch_id]].append(_runtime_event(item))
    return tuple(tuple(items) for items in grouped)


def _load_bank(root: Path) -> tuple[dict, ...]:
    tapes: list[dict] = []
    for path in sorted(root.rglob("*.json")):
        tape = json.loads(path.read_text(encoding="utf-8"))
        validate_tape(tape)
        tapes.append(tape)
    if len(tapes) != 200:
        raise RuntimeError(f"expected 200 development tapes, got {len(tapes)}")
    return tuple(tapes)


def run(tape_root: Path, output: Path) -> dict[str, object]:
    bank = _load_bank(tape_root)
    rows: list[dict[str, object]] = []
    for allocator in (FirstAvailableAllocator(), MaxEnergyMarginAllocator()):
        parity_passes = 0
        request_count = 0
        decision_count = 0
        scenario_counts: Counter[str] = Counter()
        for tape in bank:
            direct_runtime, direct_decisions = replay_tape(
                tape, controller=PreemptionController(allocator)
            )
            deferred_runtime = runtime_from_tape(tape)
            deferred_decisions = []
            for events in _batches(tape):
                now = max(event.received_at for event in events)
                pending = deferred_runtime.begin_event_batch_transaction(
                    events, PreemptionController(allocator), now=now
                )
                while pending.awaiting_allocation:
                    request_count += 1
                    proposal = allocator.propose(pending.allocation_request)
                    deferred_runtime.submit_event_batch_proposal(pending, proposal)
                result = deferred_runtime.commit_event_batch_transaction(pending)
                deferred_decisions.extend(result.decisions)
            deferred_runtime.validate_invariants()
            if tuple(deferred_decisions) != tuple(direct_decisions):
                raise RuntimeError(f"decision parity failed for {tape['tape_id']}")
            if deferred_runtime.state_sha256() != direct_runtime.state_sha256():
                raise RuntimeError(f"state parity failed for {tape['tape_id']}")
            parity_passes += 1
            decision_count += len(deferred_decisions)
            scenario_counts[str(tape["scenario_id"])] += 1
        rows.append({
            "allocator_id": allocator.allocator_id,
            "tape_count": len(bank),
            "decision_count": decision_count,
            "allocation_request_count": request_count,
            "decision_parity_pass_count": parity_passes,
            "state_sha256_parity_pass_count": parity_passes,
            "scenario_counts": dict(sorted(scenario_counts.items())),
            "status": "PASS" if parity_passes == len(bank) else "FAIL",
        })
    report = {
        "schema_version": 1,
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "classification": "deferred_transaction_interface_parity_not_model_evidence",
        "bank": "Dynamic-Preemption-Dev",
        "tape_count": len(bank),
        "allocator_count": len(rows),
        "allocator_tape_runs": len(bank) * len(rows),
        "results": rows,
        "live_runtime_mutated_before_batch_commit": False,
        "graph_version_increment_per_atomic_batch": 1,
        "model_framework_loaded": False,
        "checkpoint_loaded": False,
        "training_allowed": False,
        "training_started": False,
        "validation_started": False,
        "freeze_started": False,
        "test_started": False,
        "hidden_evaluation_started": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tape-root", type=Path, default=DEFAULT_TAPE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.tape_root.resolve(), args.output.resolve())
    print(json.dumps({
        "status": report["status"],
        "allocator_tape_runs": report["allocator_tape_runs"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

