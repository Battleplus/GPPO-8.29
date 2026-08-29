"""Post-hoc extreme-scenario stress test for the frozen 50k policies.

This is deliberately outside the minimum-validation contract.  It does not
train, select, freeze, or modify a checkpoint.  It verifies the six frozen
checkpoint hashes, generates a paired exploratory tape bank, replays every
policy on every tape, and writes results under an isolated output directory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any, Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ppo_allocation.random_event.baselines import (
    CurrentPendingExactPlannerPolicy,
    GraphPolicyAdapter,
    GreedyCostPolicy,
    MinLoadPolicy,
    NearestLegalPolicy,
)
from ppo_allocation.random_event.environment import RandomEventAllocationEnv
from ppo_allocation.random_event.events import EventTape, RandomEvent, RandomEventType
from ppo_allocation.random_event.experiment import run_episode
from ppo_allocation.random_event.models import FairPPOMLP, GraphActorCritic
from ppo_allocation.random_event.scheduler import RandomEventScheduler, SchedulerState


SCENARIOS: dict[str, dict[str, Any]] = {
    "atomic_triple_shock": {
        "title": "原子三重冲击",
        "description": "UAV 损毁、目标发现、区域空缺在同一观测批次到达，随后目标释放与再次空缺。",
        "sequence": ["UAV_DAMAGE", "TARGET_DISCOVERED", "REGION_VACANCY", "TARGET_DESTROYED", "REGION_VACANCY"],
        "occurred": [10.00, 10.03, 10.06, 14.00, 16.00],
        "observed": [11.00, 11.00, 11.00, 14.50, 16.50],
        "stressors": ["atomic_batch", "coupled_reallocation", "task_release"],
    },
    "resource_collapse": {
        "title": "三级资源坍缩",
        "description": "连续损毁三架 UAV，仅剩一架承担全部搜索任务，再连续制造区域空缺。",
        "sequence": ["UAV_DAMAGE", "UAV_DAMAGE", "UAV_DAMAGE", "REGION_VACANCY", "REGION_VACANCY"],
        "occurred": [5.00, 5.10, 5.20, 6.00, 6.10],
        "observed": [6.00, 6.10, 6.20, 6.30, 6.30],
        "stressors": ["resource_scarcity", "rapid_damage", "near_capacity_limit"],
    },
    "tracking_saturation_release": {
        "title": "跟踪饱和后延迟释放",
        "description": "三架 UAV 依次转入 TRACK，最后一架搜索 UAV 损毁；系统等待目标销毁报告释放一架 UAV。",
        "sequence": ["TARGET_DISCOVERED", "TARGET_DISCOVERED", "TARGET_DISCOVERED", "UAV_DAMAGE", "TARGET_DESTROYED"],
        "occurred": [1.00, 2.00, 3.00, 4.00, 5.00],
        "observed": [2.00, 3.00, 4.00, 5.00, 15.00],
        "stressors": ["temporary_infeasibility", "task_contention", "delayed_release"],
    },
    "out_of_order_reports": {
        "title": "因果报告乱序",
        "description": "目标销毁报告先于目标发现报告到达，并交错 UAV 损毁与区域空缺报告。",
        "sequence": ["TARGET_DISCOVERED", "TARGET_DESTROYED", "UAV_DAMAGE", "REGION_VACANCY", "TARGET_DISCOVERED"],
        "occurred": [1.00, 2.00, 3.00, 4.00, 5.00],
        "observed": [12.00, 6.00, 10.00, 8.00, 11.00],
        "stressors": ["partial_observation", "out_of_order", "causal_inversion"],
    },
    "long_blind_burst": {
        "title": "长盲区突发批次",
        "description": "五个真实事件在 0.08 秒内发生，但全部延迟到 30 秒后才作为一个批次被观察。",
        "sequence": ["UAV_DAMAGE", "TARGET_DISCOVERED", "REGION_VACANCY", "TARGET_DESTROYED", "REGION_VACANCY"],
        "occurred": [1.00, 1.02, 1.04, 1.06, 1.08],
        "observed": [30.00, 30.00, 30.00, 30.00, 30.00],
        "stressors": ["long_information_gap", "atomic_batch", "stale_world_model"],
    },
    "task_churn": {
        "title": "任务反复变更",
        "description": "目标发现与销毁快速交替，UAV 在 SEARCH/TRACK/释放之间频繁切换，最后再出现区域空缺。",
        "sequence": ["TARGET_DISCOVERED", "TARGET_DESTROYED", "TARGET_DISCOVERED", "TARGET_DESTROYED", "REGION_VACANCY"],
        "occurred": [1.00, 1.20, 1.40, 1.60, 1.80],
        "observed": [2.00, 2.10, 2.20, 2.30, 2.40],
        "stressors": ["task_churn", "rapid_release", "switching_pressure"],
    },
    "event_storm_8": {
        "title": "八事件持续风暴",
        "description": "把有效事件序列压缩为八个高度重叠的观测，超过原协议每回合五事件的密度。",
        "sequence": None,
        "stressors": ["long_horizon", "dense_overlap", "distribution_shift"],
    },
}

INITIAL_SEEDS = (880001, 880002, 880003, 880004, 880005, 880006)
METRICS = (
    "event_success_rate",
    "legal_coverage_rate",
    "recovery_delay",
    "cumulative_uncovered_time",
    "normalized_distance",
    "load_gap",
    "switch_count",
    "episode_return",
    "final_infeasible_rate",
    "temporary_infeasible_rate",
    "inference_latency_ms",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_bytes(content.encode("utf-8"))


def initial_scheduler_state(initial_seed: int) -> SchedulerState:
    env = RandomEventAllocationEnv(initial_seed=initial_seed, event_seed=initial_seed + 700_000)
    env.rng = np.random.default_rng(initial_seed)
    env._init_regions()
    env._init_uavs()
    env._init_targets()
    env._initial_assignment()
    state = SchedulerState.from_entities(
        list(env.uavs.values()), list(env.regions.values()), list(env.targets.values())
    )
    env.close()
    return state


def forced_sequence_tape(scenario_id: str, case_index: int, initial_seed: int) -> EventTape:
    spec = SCENARIOS[scenario_id]
    state = initial_scheduler_state(initial_seed)
    events: list[RandomEvent] = []
    event_seed = 9_100_000 + case_index * 100 + sum(ord(ch) for ch in scenario_id)
    for index, kind_name in enumerate(spec["sequence"]):
        kind = RandomEventType(kind_name)
        weights = {candidate: float(candidate is kind) for candidate in RandomEventType}
        scheduler = RandomEventScheduler(event_count=1, weights=weights)
        rng = random.Random(event_seed + index)
        event = scheduler.sample_event(
            state,
            rng=rng,
            event_id=f"{scenario_id}-C{case_index:02d}-E{index:02d}",
            occurred_at=float(spec["occurred"][index]),
            observed_at=float(spec["observed"][index]),
            event_seed=event_seed + index,
        )
        event = replace(
            event,
            payload={
                **dict(event.payload),
                "exploratory_scenario": scenario_id,
                "stressors": list(spec["stressors"]),
            },
        )
        events.append(event)
        state = state.apply(event).with_canonical_recovery()
    return EventTape(
        initial_seed=initial_seed,
        event_seed=event_seed,
        mode=f"extreme/{scenario_id}",
        events=tuple(events),
    )


def event_storm_tape(case_index: int, initial_seed: int) -> EventTape:
    event_seed = 9_900_000 + case_index
    scheduler = RandomEventScheduler(
        event_count=8,
        weights={
            RandomEventType.UAV_DAMAGE: 0.08,
            RandomEventType.TARGET_DISCOVERED: 0.34,
            RandomEventType.TARGET_DESTROYED: 0.28,
            RandomEventType.REGION_VACANCY: 0.30,
        },
    )
    base = scheduler.generate_tape(
        initial_scheduler_state(initial_seed),
        initial_seed=initial_seed,
        event_seed=event_seed,
        mode="sequential",
        event_count=8,
    )
    events = []
    for index, event in enumerate(base.events):
        occurred = 10.0 + index * 0.12
        observed = 15.0 + index * 0.18
        events.append(
            replace(
                event,
                event_id=f"event_storm_8-C{case_index:02d}-E{index:02d}",
                occurred_at=occurred,
                observed_at=observed,
                payload={
                    **dict(event.payload),
                    "exploratory_scenario": "event_storm_8",
                    "stressors": list(SCENARIOS["event_storm_8"]["stressors"]),
                },
            )
        )
    return EventTape(
        initial_seed=initial_seed,
        event_seed=event_seed,
        mode="extreme/event_storm_8",
        events=tuple(events),
    )


def build_tape_bank(output_dir: Path) -> list[tuple[str, str, EventTape]]:
    tapes: list[tuple[str, str, EventTape]] = []
    for scenario_id in SCENARIOS:
        for case_index, initial_seed in enumerate(INITIAL_SEEDS):
            tape = (
                event_storm_tape(case_index, initial_seed)
                if scenario_id == "event_storm_8"
                else forced_sequence_tape(scenario_id, case_index, initial_seed)
            )
            tape_id = f"{scenario_id}-C{case_index:02d}"
            path = output_dir / "tapes" / scenario_id / f"{tape_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tape.to_json(indent=2).encode("utf-8"))
            tapes.append((scenario_id, tape_id, tape))
    return tapes


def load_frozen_policies(checkpoint_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen_path = checkpoint_root / "results/random_event/minimum_validation_50k_2afa8ec/preliminary/frozen_manifests.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for item in frozen["freezes"]:
        checkpoint = checkpoint_root / item["checkpoint_path"]
        actual = sha256_file(checkpoint)
        if actual != item["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint SHA-256 mismatch: {checkpoint}")
        if item["variant"] == "PPO-MLP":
            model, metadata = FairPPOMLP.load(checkpoint, map_location="cpu")
        else:
            model, metadata = GraphActorCritic.load(checkpoint, map_location="cpu")
        model.eval()
        records.append(
            {
                "family": item["variant"],
                "training_seed": int(item["training_seed"]),
                "policy": GraphPolicyAdapter(
                    model=model,
                    name=f"{item['variant']} seed={item['training_seed']}",
                ),
                # Publish a portable provenance label, never a workstation path.
                "checkpoint": str(item["checkpoint_path"]).replace("\\", "/"),
                "sha256": actual,
                "checkpoint_metadata_source": metadata.get("attested_source_commit_sha"),
            }
        )
    return records, frozen


def finite_numbers(values: Iterable[Any]) -> list[float]:
    result = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result.append(number)
    return result


def mean(values: Iterable[Any]) -> float | None:
    numbers = finite_numbers(values)
    return statistics.fmean(numbers) if numbers else None


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["scenario"], row["family"]), []).append(row)
    output = []
    for (scenario, family), items in sorted(groups.items()):
        record: dict[str, Any] = {
            "scenario": scenario,
            "family": family,
            "episode_count": len(items),
        }
        for metric in METRICS:
            record[metric] = mean(item.get(metric) for item in items)
        output.append(record)
    return output


def paired_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    learned = [row for row in rows if row["family"] in {"PPO-MLP", "GPPO-Adaptive"}]
    index = {
        (row["scenario"], row["tape_id"], row["training_seed"], row["family"]): row
        for row in learned
    }
    effects = []
    for scenario in SCENARIOS:
        pairs = []
        for tape_id in sorted({row["tape_id"] for row in learned if row["scenario"] == scenario}):
            for training_seed in (1101, 2202, 3303):
                ppo = index[(scenario, tape_id, training_seed, "PPO-MLP")]
                gppo = index[(scenario, tape_id, training_seed, "GPPO-Adaptive")]
                pairs.append((gppo, ppo))
        for metric in METRICS:
            differences = []
            for gppo, ppo in pairs:
                left, right = gppo.get(metric), ppo.get(metric)
                if left is not None and right is not None:
                    differences.append(float(left) - float(right))
            effects.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "paired_count": len(differences),
                    "mean_difference_gppo_minus_ppo": mean(differences),
                    "gppo_lower_count": sum(value < 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                    "gppo_higher_count": sum(value > 0 for value in differences),
                }
            )
    return effects


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report(
    output_dir: Path,
    aggregate_rows: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    checkpoint_count: int,
) -> None:
    aggregate_index = {(row["scenario"], row["family"]): row for row in aggregate_rows}
    effect_index = {(row["scenario"], row["metric"]): row for row in effects}
    lines = [
        "# 极端多事件场景探索性压力测试",
        "",
        "> 这是 post-hoc exploratory stress test，不属于 minimum-validation held-out 证据，不能用于 checkpoint selection 或正式优越性声明。",
        "",
        f"- 固定 50k checkpoints：{checkpoint_count}/6，运行前 SHA-256 全部匹配",
        f"- 场景数：{len(SCENARIOS)}",
        f"- 每场景事件 tapes：{len(INITIAL_SEEDS)}",
        f"- 模型配对：2 variants × 3 training seeds",
        "- 额外参照：Nearest Legal、Min Load、Greedy Cost、Current-Pending Exact Planner",
        "",
        "## 场景设计",
        "",
        "| 场景 | 设计 | 主要压力 |",
        "|---|---|---|",
    ]
    for scenario_id, spec in SCENARIOS.items():
        lines.append(
            f"| {scenario_id}（{spec['title']}） | {spec['description']} | {', '.join(spec['stressors'])} |"
        )
    lines.extend(
        [
            "",
            "## PPO-MLP 与 GPPO-Adaptive 的场景均值",
            "",
            "| 场景 | 方法 | 成功率 | 覆盖率 | 恢复延迟 | 累计空缺 | 距离 | 负载差 | 回报 | 最终不可行率 | 推理 ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario_id in SCENARIOS:
        for family in ("PPO-MLP", "GPPO-Adaptive"):
            row = aggregate_index[(scenario_id, family)]
            lines.append(
                "| " + " | ".join(
                    [
                        scenario_id,
                        family,
                        fmt(row["event_success_rate"]),
                        fmt(row["legal_coverage_rate"]),
                        fmt(row["recovery_delay"]),
                        fmt(row["cumulative_uncovered_time"]),
                        fmt(row["normalized_distance"]),
                        fmt(row["load_gap"]),
                        fmt(row["episode_return"]),
                        fmt(row["final_infeasible_rate"]),
                        fmt(row["inference_latency_ms"]),
                    ]
                ) + " |"
            )
    lines.extend(
        [
            "",
            "## 配对差值（GPPO − PPO）",
            "",
            "负数表示 GPPO 数值更低；是否更好取决于指标方向。回报和成功率越高越好，其余代价类指标越低越好。",
            "",
            "| 场景 | 成功率差 | 恢复延迟差 | 累计空缺差 | 距离差 | 负载差 | 回报差 | 推理延迟差 ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario_id in SCENARIOS:
        get = lambda metric: effect_index[(scenario_id, metric)]["mean_difference_gppo_minus_ppo"]
        lines.append(
            f"| {scenario_id} | {fmt(get('event_success_rate'))} | {fmt(get('recovery_delay'))} | "
            f"{fmt(get('cumulative_uncovered_time'))} | {fmt(get('normalized_distance'))} | "
            f"{fmt(get('load_gap'))} | {fmt(get('episode_return'))} | {fmt(get('inference_latency_ms'))} |"
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- 这些场景由看到既有结果之后设计，是探索性压力测试，不是新的 held-out test。",
            "- 当前环境在事件 observed_at 时才把事件送入 belief/runtime；长盲区指标主要衡量观测后的恢复，不能代表真实世界盲区损失。",
            "- 当前策略没有循环记忆或 belief-state，因此乱序/缺失信息测试主要暴露系统合同和前馈策略的限制。",
            "- 只有 3 个独立训练 seeds；差值仅描述本次样本，不作统计显著性或普遍优越性声明。",
        ]
    )
    (output_dir / "REPORT.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def build_inventory(output_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_inventory.json":
            files.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    inventory = {"schema_version": 1, "file_count": len(files), "files": files}
    write_json(output_dir / "sha256_inventory.json", inventory)
    return inventory


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="ppo_allocation directory containing the frozen manifest and six 50k checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "experiments/extreme_scenarios/results_20260827",
    )
    parser.add_argument("--max-decisions", type=int, default=150)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    policies, frozen = load_frozen_policies(args.checkpoint_root.resolve())
    tapes = build_tape_bank(output_dir)
    baselines = [
        ("Nearest Legal", lambda: NearestLegalPolicy()),
        ("Min Load", lambda: MinLoadPolicy()),
        ("Greedy Cost", lambda: GreedyCostPolicy()),
        ("Current-Pending Exact Planner", lambda: CurrentPendingExactPlannerPolicy()),
    ]

    rows: list[dict[str, Any]] = []
    trace_index = []
    total_runs = len(tapes) * (len(policies) + len(baselines))
    completed = 0
    for scenario, tape_id, tape in tapes:
        candidates = [
            (item["family"], item["training_seed"], item["policy"], item["checkpoint"], item["sha256"])
            for item in policies
        ]
        candidates.extend((name, None, factory(), None, None) for name, factory in baselines)
        for family, training_seed, policy, checkpoint, checkpoint_sha in candidates:
            algorithm = family if training_seed is None else f"{family} seed={training_seed}"
            episode, trace = run_episode(
                policy,
                tape_id=tape_id,
                tape=tape,
                algorithm=algorithm,
                max_decisions=args.max_decisions,
            )
            record = episode.to_dict()
            record.update(
                {
                    "scenario": scenario,
                    "family": family,
                    "training_seed": training_seed,
                    "checkpoint": checkpoint,
                    "checkpoint_sha256": checkpoint_sha,
                }
            )
            rows.append(record)
            safe_algorithm = algorithm.lower().replace(" ", "_").replace("=", "_").replace("-", "_")
            trace_path = output_dir / "traces" / scenario / safe_algorithm / f"{tape_id}.json"
            write_json(trace_path, trace)
            trace_index.append(
                {
                    "scenario": scenario,
                    "algorithm": algorithm,
                    "tape_id": tape_id,
                    "path": trace_path.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(trace_path),
                }
            )
            completed += 1
            if completed % 25 == 0 or completed == total_runs:
                print(f"completed {completed}/{total_runs}", flush=True)

    aggregate_rows = aggregate(rows)
    effects = paired_effects(rows)
    write_json(output_dir / "scenario_catalog.json", SCENARIOS)
    write_json(output_dir / "episode_results.json", rows)
    write_csv(output_dir / "episode_results.csv", rows)
    write_json(output_dir / "aggregate_results.json", aggregate_rows)
    write_csv(output_dir / "aggregate_results.csv", aggregate_rows)
    write_json(output_dir / "paired_effects_gppo_minus_ppo.json", effects)
    write_csv(output_dir / "paired_effects_gppo_minus_ppo.csv", effects)
    write_json(output_dir / "trace_index.json", trace_index)

    head = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "classification": "post_hoc_exploratory_stress_test",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_head": head,
        "formal_minimum_validation_modified": False,
        "training_started": False,
        "checkpoint_selection": False,
        "checkpoint_count": len(policies),
        "checkpoint_sha256_verified": True,
        "scenario_count": len(SCENARIOS),
        "tape_count": len(tapes),
        "policy_episode_count": len(rows),
        "frozen_attested_source_commit_sha": frozen.get("attested_source_commit_sha"),
        "fixed_evaluation_checkpoint": frozen.get("fixed_evaluation_checkpoint"),
    }
    write_json(output_dir / "run_summary.json", summary)
    write_report(output_dir, aggregate_rows, effects, len(policies))
    inventory = build_inventory(output_dir)
    summary["inventory_file_count"] = inventory["file_count"]
    write_json(output_dir / "run_summary.json", summary)
    # Refresh inventory because run_summary gained its final inventory count.
    build_inventory(output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
