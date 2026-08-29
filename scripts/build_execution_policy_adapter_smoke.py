"""Build deterministic PPO/GPPO adapter smoke evidence for 4/8/16/32 UAV."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_preemption.adapter import (  # noqa: E402
    ACTION_CAPACITY,
    FLAT_OBSERVATION_DIMENSION,
    adapter_layout_sha256,
    build_flat_observation,
    build_hetero_observation,
    proposal_from_policy_action,
)
from execution_preemption.allocation import build_allocation_request  # noqa: E402
from execution_preemption.graph import build_execution_graph  # noqa: E402
from execution_preemption.models import (  # noqa: E402
    DecisionType,
    EventDecision,
    EventPriority,
    TaskRuntime,
    UAVRuntime,
)
from execution_preemption.runtime import ExecutionRuntime  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT / "experiments" / "dynamic_preemption" / "dev_v1" / "policy_adapter_smoke.json"
)


def _sha(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _runtime(uav_count: int) -> ExecutionRuntime:
    runtime = ExecutionRuntime()
    for index in range(uav_count):
        runtime.add_uav(UAVRuntime(
            f"U{index:02d}",
            energy_ratio=0.90 - index * 0.001,
            supported_task_types=frozenset({"SEARCH"}),
        ))
    for index in range(uav_count * 2):
        runtime.add_task(TaskRuntime(
            f"T{index:03d}",
            "SEARCH",
            priority=10 + index,
            deadline=100.0 + index,
        ))
    return runtime


def _hetero_canonical(observation) -> dict[str, object]:
    return {
        "node_ids": {key: list(value) for key, value in observation.node_ids.items()},
        "node_features": {
            key: [list(row) for row in value]
            for key, value in observation.node_features.items()
        },
        "edge_indices": {
            "|".join(key): [list(row) for row in value]
            for key, value in observation.edge_indices.items()
        },
        "rule_context": list(observation.rule_context),
        "action_mask": list(observation.action_space.mask),
        "action_bindings": [list(item) if item is not None else None
                            for item in observation.action_space.bindings],
    }


def build_report() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for uav_count in (4, 8, 16, 32):
        runtime = _runtime(uav_count)
        graph = build_execution_graph(runtime, now=1.0)
        task = runtime.tasks["T000"]
        request = build_allocation_request(
            request_id=f"adapter-smoke-{uav_count}",
            graph_version=runtime.graph_version,
            task=task,
            uavs=tuple(runtime.uavs.values()),
            decision_type=DecisionType.CONTINUE,
            reason="adapter smoke",
            generated_at=1.0,
        )
        decision = EventDecision(
            event_id=f"adapter-event-{uav_count}",
            priority=EventPriority.P3,
            information_age=0.0,
            confidence=1.0,
            decision=DecisionType.CONTINUE,
            displaced_task_id=None,
            selected_uav=None,
            reason="adapter smoke",
            graph_version=runtime.graph_version,
            allocation_request_id=request.request_id,
        )
        flat = build_flat_observation(graph, request=request, decision=decision)
        hetero = build_hetero_observation(graph, request=request, decision=decision)
        if flat.action_space != hetero.action_space:
            raise RuntimeError("flat and hetero action spaces differ")
        selected_index = next(
            index for index, enabled in enumerate(flat.action_space.mask)
            if enabled and index != 0
        )
        proposal = proposal_from_policy_action(
            request,
            flat.action_space,
            selected_index,
            allocator_id="adapter_smoke_v1",
            current_graph_version=runtime.graph_version,
            current_graph_sha256=graph.sha256(),
        )
        rows.append({
            "uav_count": uav_count,
            "task_count": len(runtime.tasks),
            "graph_sha256": graph.sha256(),
            "flat_observation_dimension": len(flat.vector),
            "flat_observation_sha256": _sha(list(flat.vector)),
            "hetero_observation_sha256": _sha(_hetero_canonical(hetero)),
            "action_capacity": len(flat.action_space.mask),
            "valid_action_count_including_noop": flat.action_space.valid_action_count,
            "request_candidate_count": len(request.candidates),
            "selected_action_index": selected_index,
            "selected_uav": proposal.uav_id,
            "selected_task": proposal.task_id,
            "shared_action_space": True,
            "status": "PASS",
        })
    return {
        "schema_version": 1,
        "status": "PASS",
        "classification": "adapter_pretraining_smoke_not_model_evidence",
        "adapter_layout_sha256": adapter_layout_sha256(),
        "flat_observation_dimension": FLAT_OBSERVATION_DIMENSION,
        "action_capacity": ACTION_CAPACITY,
        "uav_counts": [4, 8, 16, 32],
        "scales": rows,
        "model_framework_loaded": False,
        "checkpoint_loaded": False,
        "training_allowed": False,
        "training_started": False,
        "validation_started": False,
        "freeze_started": False,
        "test_started": False,
        "hidden_evaluation_started": False,
    }


def write_report(path: Path, report: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = write_report(args.output.resolve(), build_report())
    print(json.dumps({"status": "PASS", "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

