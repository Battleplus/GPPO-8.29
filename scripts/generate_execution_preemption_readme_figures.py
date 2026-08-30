"""Generate static figures embedded by the execution-preemption Chinese README.

The figures deliberately separate development/interface evidence from the older
post-hoc exploratory campaign.  They must not be used as formal model
effectiveness evidence.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "experiments" / "dynamic_preemption" / "dev_v1"
OLD = ROOT / "experiments" / "extreme_scenarios" / "results_20260827"
OUT = ROOT / "docs" / "assets" / "execution_preemption_results"

INK = "#172033"
MUTED = "#667085"
GRID = "#D9E0EA"
BLUE = "#2563EB"
BLUE_LIGHT = "#DBEAFE"
ORANGE = "#D97706"
ORANGE_LIGHT = "#FFEDD5"
GREY = "#98A2B3"
PAPER = "#FFFFFF"


def load_json(name: str) -> dict:
    with (DEV / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
            "font.size": 11,
        }
    )


def title(fig: plt.Figure, main: str, subtitle: str) -> None:
    fig.suptitle(main, x=0.06, y=0.98, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.06, 0.925, subtitle, ha="left", va="top", color=MUTED, fontsize=10.5)


def save(fig: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def plot_validation_evidence() -> None:
    allocator = load_json("allocator_replay_summary.json")
    baseline = load_json("baseline_replay_smoke.json")
    parity = load_json("deferred_transaction_parity.json")
    adapter = load_json("policy_adapter_smoke.json")
    rollout = load_json("framework_rollout_smoke.json")
    runner = load_json("training_runner_smoke.json")

    progress = (ROOT / "docs" / "EXECUTION_PREEMPTION_PROGRESS_ZH.md").read_text(encoding="utf-8")
    match = re.search(r"execution_preemption_tests:\s*(\d+)/(\d+)\s+PASS", progress)
    if not match:
        raise RuntimeError("Cannot find execution_preemption_tests count in progress document")
    test_passed, test_total = map(int, match.groups())

    decision_parity = sum(item["decision_parity_pass_count"] for item in parity["results"])
    state_parity = sum(item["state_sha256_parity_pass_count"] for item in parity["results"])
    items = [
        ("专项自动化测试", test_passed, test_total),
        ("开发带 × 分配器回放", allocator["allocator_tape_runs"], allocator["allocator_tape_runs"]),
        ("基线接口/安全回放", baseline["allocator_tape_runs"], baseline["allocator_tape_runs"]),
        ("事务决策一致", decision_parity, parity["allocator_tape_runs"]),
        ("最终状态哈希一致", state_parity, parity["allocator_tape_runs"]),
        ("4/8/16/32 规模适配", len(adapter["scales"]), len(adapter["scales"])),
        ("PPO/GPPO 框架回放", sum(run["status"] == "PASS" for run in rollout["runs"]), rollout["run_count"]),
        ("四方法 tiny 训练链路", sum(method["status"] == "PASS" for method in runner["methods"]), runner["learned_method_count"]),
    ]

    labels = [item[0] for item in items][::-1]
    passed = np.array([item[1] for item in items], dtype=float)[::-1]
    totals = np.array([item[2] for item in items], dtype=float)[::-1]
    rates = passed / totals * 100

    fig, ax = plt.subplots(figsize=(12, 7.3))
    title(
        fig,
        "执行中动态重分配：当前验证证据概览",
        "均为开发/接口/安全验证；横轴为通过率，条末 n/n 为各自样本量，不代表算法效果。",
    )
    y = np.arange(len(labels))
    bars = ax.barh(y, rates, color=BLUE, edgecolor="#1D4ED8", height=0.58)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 112)
    ax.set_xlabel("通过率（%）")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, numerator, denominator in zip(bars, passed, totals):
        ax.text(
            101.2,
            bar.get_y() + bar.get_height() / 2,
            f"{int(numerator)}/{int(denominator)}",
            va="center",
            ha="left",
            fontsize=10.5,
            color=INK,
            fontweight="bold",
        )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.25, right=0.94, top=0.86, bottom=0.12)
    save(fig, "01_validation_evidence.png")


def plot_decision_distribution() -> None:
    allocator = load_json("allocator_replay_summary.json")
    result = allocator["results"][0]
    counts = result["decision_counts"]
    order = ["CONTINUE", "PREEMPT", "QUEUE", "RTB", "MIGRATE", "ABORT"]
    labels = ["继续执行", "抢占", "排队", "返航", "迁移", "终止"]
    values = [counts[key] for key in order]
    total = result["decision_count"]

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    title(
        fig,
        "开发事件带的仲裁决策分布",
        "每个分配器 200 条开发带共产生 280 个决策；两种分配器的安全仲裁分布完全一致。",
    )
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, values, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=1.5, height=0.62)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_xlabel("决策次数（每个分配器）")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            value + 2.2,
            bar.get_y() + bar.get_height() / 2,
            f"{value}（{value / total:.1%}）",
            va="center",
            fontsize=10.5,
            color=INK,
            fontweight="bold",
        )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.93, top=0.84, bottom=0.14)
    save(fig, "02_decision_distribution.png")


def plot_atomicity_and_safety() -> None:
    rollout = load_json("framework_rollout_smoke.json")
    parity = load_json("deferred_transaction_parity.json")
    columns = [
        ("mask_violations", "动作掩码违规"),
        ("resource_conflicts", "资源冲突"),
        ("stale_command_resurrections", "旧命令复活"),
        ("energy_safety_violations", "能源安全违规"),
    ]
    rows = []
    row_labels = []
    policy_names = {
        "ppo_mlp_framework_smoke_v1": "PPO",
        "gppo_adaptive_framework_smoke_v1": "GPPO",
    }
    scenario_names = {
        "execution_uav_destroyed": "UAV损毁",
        "simultaneous_p1": "并发P1",
    }
    for run in rollout["runs"]:
        rows.append([run[key] for key, _ in columns])
        row_labels.append(f"{policy_names[run['policy_id']]} · {scenario_names[run['scenario_id']]}")

    matrix = np.asarray(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 6.9))
    title(
        fig,
        "PPO/GPPO 框架回放的安全约束检查",
        "4 次开发 smoke 均为零违规；事务回放另取得 400/400 决策一致与 400/400 状态哈希一致。",
    )
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), [label for _, label in columns])
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=15, fontweight="bold")
    for x in np.arange(-0.5, matrix.shape[1], 1):
        ax.axvline(x, color=PAPER, linewidth=3)
    for y in np.arange(-0.5, matrix.shape[0], 1):
        ax.axhline(y, color=PAPER, linewidth=3)
    decision_parity = sum(item["decision_parity_pass_count"] for item in parity["results"])
    state_parity = sum(item["state_sha256_parity_pass_count"] for item in parity["results"])
    fig.text(
        0.06,
        0.075,
        f"原子事务校验：决策一致 {decision_parity}/{parity['allocator_tape_runs']}  ·  "
        f"状态一致 {state_parity}/{parity['allocator_tape_runs']}  ·  "
        f"提交前修改 live runtime = {str(parity['live_runtime_mutated_before_batch_commit']).lower()}",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.78, bottom=0.18)
    save(fig, "03_atomicity_and_safety.png")


def plot_graph_scaling() -> None:
    schema = load_json("graph_schema_smoke.json")
    uavs = np.array([item["uav_count"] for item in schema["scales"]])
    nodes = np.array([sum(item["node_counts"].values()) for item in schema["scales"]])
    edges = np.array([item["edge_count"] for item in schema["scales"]])
    candidates = np.array([item["action_candidate_count"] for item in schema["scales"]])

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    title(
        fig,
        "五类节点图的结构规模测试",
        "4/8/16/32 架 UAV 均通过 schema smoke；纵轴为对数刻度，仅说明结构增长，不代表推理时延。",
    )
    ax.plot(uavs, candidates, marker="o", markersize=8, linewidth=2.4, color=ORANGE, label="UAV–Task 动作候选")
    ax.plot(uavs, edges, marker="s", markersize=7, linewidth=2.4, color=BLUE, label="图边数")
    ax.plot(uavs, nodes, marker="^", markersize=7, linewidth=2.2, color=INK, linestyle="--", label="当前节点总数")
    ax.set_yscale("log", base=2)
    ax.set_xticks(uavs)
    ax.set_xlabel("UAV 数量（Task 数量固定为 UAV 的 2 倍）")
    ax.set_ylabel("数量（log2）")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for series, color in ((candidates, ORANGE), (edges, BLUE), (nodes, INK)):
        for x, value in zip(uavs, series):
            ax.annotate(str(value), (x, value), xytext=(0, 8), textcoords="offset points", ha="center", color=color, fontsize=9)
    ax.legend(loc="upper left", frameon=False, ncol=3, bbox_to_anchor=(0, 1.03))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.1, right=0.96, top=0.79, bottom=0.14)
    save(fig, "04_graph_scaling.png")


def plot_parameter_count() -> None:
    rollout = load_json("framework_rollout_smoke.json")
    counts = rollout["model_parameter_counts"]
    gppo = counts["gppo_adaptive_framework_smoke_v1"]
    ppo = counts["ppo_mlp_framework_smoke_v1"]
    labels = ["GPPO-Adaptive", "PPO-MLP"]
    values = [gppo, ppo]

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    title(
        fig,
        "当前框架策略网络参数量",
        "同一 smoke 配置下的可训练参数计数；参数更少不等于回报更高或推理一定更快。",
    )
    y = np.arange(2)[::-1]
    bars = ax.barh(y, values, color=[BLUE, BLUE_LIGHT], edgecolor=BLUE, linewidth=1.5, height=0.55)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, ppo * 1.16)
    ax.set_xlabel("参数量")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + ppo * 0.018, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", fontweight="bold")
    ax.text(
        0.99,
        0.08,
        f"PPO 参数量约为 GPPO 的 {ppo / gppo:.1f} 倍",
        transform=ax.transAxes,
        ha="right",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.78, bottom=0.18)
    save(fig, "05_policy_parameter_count.png")


SCENARIO_LABELS = {
    "atomic_triple_shock": "三重原子冲击",
    "event_storm_8": "8事件风暴",
    "long_blind_burst": "长盲区突发",
    "out_of_order_reports": "乱序报告",
    "resource_collapse": "资源坍缩",
    "task_churn": "任务高频变化",
    "tracking_saturation_release": "跟踪饱和释放",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_old_latency() -> None:
    rows = read_csv(OLD / "aggregate_results.csv")
    selected = [row for row in rows if row["family"] in {"GPPO-Adaptive", "PPO-MLP"}]
    scenario_order = list(SCENARIO_LABELS)
    by_key = {(row["scenario"], row["family"]): row for row in selected}
    gppo = np.array([float(by_key[(scenario, "GPPO-Adaptive")]["inference_latency_ms"]) for scenario in scenario_order])
    ppo = np.array([float(by_key[(scenario, "PPO-MLP")]["inference_latency_ms"]) for scenario in scenario_order])
    labels = [SCENARIO_LABELS[scenario] for scenario in scenario_order]

    fig, ax = plt.subplots(figsize=(12, 7.2))
    title(
        fig,
        "旧极端场景探索实验：推理时延",
        "2026-08-27 post-hoc 数据，GPPO/PPO 各场景 18 episodes；不是本次 execution-preemption-v1 正式效果证据。",
    )
    y = np.arange(len(labels))
    height = 0.34
    ax.barh(y + height / 2, gppo, height=height, color=BLUE, edgecolor="#1D4ED8", label="GPPO-Adaptive")
    ax.barh(y - height / 2, ppo, height=height, color=ORANGE_LIGHT, edgecolor=ORANGE, label="PPO-MLP")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("平均推理时延（ms）")
    ax.set_xlim(0, max(gppo) * 1.23)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for index, (g_value, p_value) in enumerate(zip(gppo, ppo)):
        ax.text(g_value + 0.25, index + height / 2, f"{g_value:.1f}", va="center", fontsize=9, color=BLUE)
        ax.text(p_value + 0.25, index - height / 2, f"{p_value:.1f}", va="center", fontsize=9, color=ORANGE)
    ax.legend(frameon=False, loc="lower right")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.19, right=0.95, top=0.82, bottom=0.12)
    save(fig, "06_legacy_exploratory_latency.png")


def plot_old_vacancy_delta() -> None:
    rows = read_csv(OLD / "paired_effects_gppo_minus_ppo.csv")
    selected = {row["scenario"]: row for row in rows if row["metric"] == "cumulative_uncovered_time"}
    scenario_order = list(SCENARIO_LABELS)
    values = np.array([float(selected[scenario]["mean_difference_gppo_minus_ppo"]) for scenario in scenario_order])
    labels = [SCENARIO_LABELS[scenario] for scenario in scenario_order]
    colors = [BLUE if value < 0 else ORANGE_LIGHT for value in values]
    edges = ["#1D4ED8" if value < 0 else ORANGE for value in values]

    fig, ax = plt.subplots(figsize=(12, 7.2))
    title(
        fig,
        "旧极端场景探索实验：累计任务空缺差值",
        "GPPO − PPO，负值表示 GPPO 空缺更少；每场景 18 个配对样本。结果方向混合，不能支持普遍优越性。",
    )
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor=edges, linewidth=1.4, height=0.58)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.set_xlabel("累计任务空缺时间差（GPPO − PPO）")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    margin = max(abs(values)) * 0.025
    for bar, value in zip(bars, values):
        if value < -2:
            label_x = value + margin
            label_ha = "left"
            label_color = PAPER
        elif value < 0:
            label_x = value - margin
            label_ha = "right"
            label_color = INK
        else:
            label_x = value + margin
            label_ha = "left"
            label_color = INK
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            va="center",
            ha=label_ha,
            fontsize=9.5,
            color=label_color,
            fontweight="bold",
        )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.19, right=0.95, top=0.82, bottom=0.12)
    save(fig, "07_legacy_exploratory_vacancy_delta.png")


def main() -> None:
    setup_style()
    plot_validation_evidence()
    plot_decision_distribution()
    plot_atomicity_and_safety()
    plot_graph_scaling()
    plot_parameter_count()
    plot_old_latency()
    plot_old_vacancy_delta()
    print(f"Generated 7 README figures in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
