"""Build deterministic 4/8/16/32 UAV graph-schema smoke evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution_preemption import (  # noqa: E402
    EventPriority,
    ExecutionRuntime,
    RegionRuntime,
    RuntimeEvent,
    RuntimeEventType,
    TargetRuntime,
    TaskRuntime,
    UAVRuntime,
    build_execution_graph,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    path.write_bytes((payload + "\n").encode("utf-8"))


def build_scale(uav_count: int) -> dict[str, object]:
    runtime = ExecutionRuntime()
    for index in range(uav_count):
        runtime.add_uav(UAVRuntime(
            f"U{index:02d}",
            energy_ratio=0.9 - (index % 4) * 0.05,
            supported_task_types=frozenset({"SEARCH", "STRIKE"}),
        ))
    for index in range(uav_count * 2):
        runtime.add_task(TaskRuntime(
            f"T{index:03d}",
            "SEARCH" if index % 2 == 0 else "STRIKE",
            priority=(index * 7) % 100,
            deadline=100.0 + index,
            metadata={
                "region_id": f"R{index:03d}",
                "depends_on": [] if index == 0 else [f"T{index - 1:03d}"],
            },
        ))
    regions = [
        RegionRuntime(
            f"R{index:03d}", vacant=True, demand=(index % 5) / 4,
            priority=(index * 11) % 100,
        )
        for index in range(uav_count * 2)
    ]
    targets = [
        TargetRuntime(
            f"X{index:02d}", status=("UNKNOWN", "DETECTED", "TRACKED", "DESTROYED")[index % 4],
            confidence=(index % 5) / 4,
            priority=(index * 13) % 100,
        )
        for index in range(uav_count)
    ]
    event_types = tuple(RuntimeEventType)
    events = [
        RuntimeEvent(
            event_id=f"E{index:02d}",
            event_type=event_type,
            priority=EventPriority(index % len(EventPriority)),
            occurred_at=1.0 + index,
            received_at=1.1 + index,
            task_id=f"T{index % (uav_count * 2):03d}",
            uav_id=f"U{index % uav_count:02d}",
            payload={"severity": 0.8},
        )
        for index, event_type in enumerate(event_types)
    ]
    graph = build_execution_graph(
        runtime,
        now=10.0,
        events=events,
        regions=regions,
        targets=targets,
        preemption_links=[("T001", "T000")],
    )
    graph.validate()
    relation_counts: dict[str, int] = {}
    for edge in graph.edges:
        key = "|".join(edge.relation_key)
        relation_counts[key] = relation_counts.get(key, 0) + 1
    return {
        "uav_count": uav_count,
        "task_count": uav_count * 2,
        "node_counts": {key: len(value) for key, value in graph.nodes.items()},
        "edge_count": len(graph.edges),
        "relation_counts": dict(sorted(relation_counts.items())),
        "action_candidate_count": len(graph.action_candidates),
        "expected_action_candidate_count": uav_count * uav_count * 2,
        "noop_present": graph.noop_action == ("NOOP", "NOOP"),
        "graph_sha256": graph.sha256(),
        "status": "PASS",
    }


def build(output_path: Path) -> dict[str, object]:
    scales = [build_scale(item) for item in (4, 8, 16, 32)]
    if any(item["action_candidate_count"] != item["expected_action_candidate_count"] for item in scales):
        raise RuntimeError("dynamic action candidate cardinality mismatch")
    result = {
        "schema_version": 1,
        "schema_id": "execution-preemption-heterograph-v1",
        "classification": "training_precondition_smoke_not_model_evidence",
        "status": "PASS",
        "scale_count": 4,
        "uav_counts": [4, 8, 16, 32],
        "tasks_per_uav": 2,
        "old_checkpoint_compatible": False,
        "requires_new_training_contract": True,
        "training_started": False,
        "scales": scales,
    }
    _write_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT / "experiments" / "dynamic_preemption" / "dev_v1"
            / "graph_schema_smoke.json"
        ),
    )
    args = parser.parse_args()
    result = build(args.output.resolve())
    print(json.dumps({
        "status": result["status"],
        "uav_counts": result["uav_counts"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
