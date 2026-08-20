"""
连续快照测试 v2：多 AOI 排序 + 逐区域连续执行。

作战想定：
  战场 300×300 km，6×6 个 AOI。上级指定 3 个重点关注 AOI：
    - A_5_6：疑似雷达阵地（高威胁、高价值）
    - A_3_4：疑似指挥所 + 装甲车集结（中等）
    - A_1_5：前方装甲车机动（最高指挥员优先级）

测试流程：
  1. 展示 3 个 AOI 的指标计算（target_prior / value / threat）
  2. AOIRouter 全排列排序，输出每种排列的得分与最优路径
  3. 按最优顺序逐 AOI 执行（含反馈推进），每个 AOI 求解 MILP
  4. 跨 AOI 对比分析

使用方法:
    python demo_continuous_snapshots.py
"""

import sys
import os
import numpy as np
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aoi.aoi_state import AoiInfo, AOIRouteState, ExecutionFeedback
from aoi.aoi_router import (
    AOIRouter, _aoi_center, _aoi_value, _score_sequence,
    W_PRIORITY, W_TARGET_VALUE, W_TARGET_THREAT, W_TARGET_PRIOR,
    DISCOUNT_GAMMA, DISTANCE_PENALTY_ALPHA,
)
from multi_aoi_interface import MultiAOITaskAllocator


def sep(title, char="="):
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


def sub(title):
    print(f"\n  ── {title} ──")


# ================================================================
# 场景定义：3 个 AOI，目标分布在各自 AOI 内
# ================================================================

# AOI A_5_6：row=5, col=6, 中心 (275, 275)
#   x ∈ [250, 300], y ∈ [200, 250]
#   g1: RADAR at (265, 238), 高价值/高威胁——雷达阵地
AOI_5_6 = {
    "id": "A_5_6", "row": 5, "col": 6,
    "priority": 0.8,
    "target_prior": 0.70,
    "target_value": 0.97,
    "target_threat": 0.91,
}

# AOI A_3_4：row=3, col=4, 中心 (175, 175)
#   x ∈ [150, 200], y ∈ [100, 150]
#   g2: CP at (175, 125), g3: AV at (185, 135)
AOI_3_4 = {
    "id": "A_3_4", "row": 3, "col": 4,
    "priority": 0.6,
    "target_prior": 0.55,
    "target_value": 0.775,
    "target_threat": 0.55,
}

# AOI A_1_5：row=1, col=5, 中心 (225, 25)
#   x ∈ [200, 250], y ∈ [0, 50]
#   g4: AV at (225, 25), g5: AV at (235, 35)
#   指挥员最高优先级（掩护主力侧翼）
AOI_1_5 = {
    "id": "A_1_5", "row": 1, "col": 5,
    "priority": 0.9,
    "target_prior": 0.40,
    "target_value": 0.675,
    "target_threat": 0.475,
}

ALL_AOIS_RAW = [AOI_5_6, AOI_3_4, AOI_1_5]

# 目标定义：每个目标归属其 AOI
ALL_TARGETS = [
    # A_5_6 内
    {"tid": "g1", "type": "RADAR", "pos": [265, 238], "value": 0.97, "threat": 0.91, "confirmed": True, "alive": True},
    # A_3_4 内
    {"tid": "g2", "type": "CP",    "pos": [175, 125], "value": 0.85, "threat": 0.60, "confirmed": True, "alive": True},
    {"tid": "g3", "type": "AV",    "pos": [185, 135], "value": 0.70, "threat": 0.50, "confirmed": True, "alive": True},
    # A_1_5 内
    {"tid": "g4", "type": "AV",    "pos": [225,  25], "value": 0.70, "threat": 0.50, "confirmed": True, "alive": True},
    {"tid": "g5", "type": "AV",    "pos": [235,  35], "value": 0.65, "threat": 0.45, "confirmed": True, "alive": True},
]

STAGING_POS = [150.0, -50.0]
GRID_WEATHER = {"c0": 0.20, "c1": 0.25, "c2": 0.40, "c3": 0.55, "c4": 0.72}

PLATFORMS_CFG = {
    "UAV": {
        "count": 5,
        "pos": STAGING_POS,
        "sensors": ["EO", "SAR", "ESM"],
        "munitions": {"HF": 0, "RKT": 0, "GUN": 0},
    },
    "HELI": {
        "count": 2,
        "pos": STAGING_POS,
        "sensors": ["MMW", "EOIR"],
        "munitions": {"HF": 16, "RKT": 76, "GUN": 1200},
    },
}


# ================================================================
# Part 1: AOI 指标计算与排序展示
# ================================================================
def show_aoi_stats():
    sep("Part 1: AOI 指标计算")

    print(f"\n  综合价值公式:")
    print(f"    V(AOI) = {W_PRIORITY}×priority + {W_TARGET_VALUE}×target_value")
    print(f"            + {W_TARGET_THREAT}×target_threat + {W_TARGET_PRIOR}×target_prior")
    print(f"\n  注：本场景中 target_prior/value/threat 由态势理解模块预计算提供。")
    print(f"      实际系统应根据 aoi_stats_spec.md 中的 compute_aoi_stats() 动态计算。")

    aoi_objs = []
    for raw in ALL_AOIS_RAW:
        aoi = AoiInfo(**raw)
        v = _aoi_value(aoi)
        aoi_objs.append(aoi)

        print(f"\n  AOI {aoi.id} (row={aoi.row}, col={aoi.col}):")
        print(f"    priority      = {aoi.priority:.2f}  × {W_PRIORITY} = {W_PRIORITY * aoi.priority:.3f}")
        print(f"    target_value  = {aoi.target_value:.3f}  × {W_TARGET_VALUE} = {W_TARGET_VALUE * aoi.target_value:.3f}")
        print(f"    target_threat = {aoi.target_threat:.3f}  × {W_TARGET_THREAT} = {W_TARGET_THREAT * aoi.target_threat:.3f}")
        print(f"    target_prior  = {aoi.target_prior:.3f}  × {W_TARGET_PRIOR} = {W_TARGET_PRIOR * aoi.target_prior:.3f}")
        print(f"    ──────────────────────────────────────────")
        print(f"    V({aoi.id})    = {v:.4f}")

        # 显示 AOI 内目标
        aoi_targets = [t for t in ALL_TARGETS
                       if (aoi.col - 1) * 50 <= t["pos"][0] <= aoi.col * 50
                       and (aoi.row - 1) * 50 <= t["pos"][1] <= aoi.row * 50]
        if aoi_targets:
            print(f"    包含目标: {', '.join(t['tid'] + '(' + t['type'] + ')' for t in aoi_targets)}")

    return aoi_objs


def show_aoi_sorting(aoi_objs):
    sep("Part 2: AOI 全排列排序")

    router = AOIRouter(grid_size_km=50.0)
    start = np.array(STAGING_POS, dtype=np.float64)

    print(f"\n  出发点: staging ({STAGING_POS[0]:.0f}, {STAGING_POS[1]:.0f})")
    print(f"  折扣因子 γ = {DISCOUNT_GAMMA}  距离惩罚 α = {DISTANCE_PENALTY_ALPHA}")
    print(f"\n  全排列枚举 ({len(aoi_objs)}! = {len(list(permutations(aoi_objs)))} 种):")

    all_perms = list(permutations(aoi_objs))
    scored = []
    for i, perm in enumerate(all_perms):
        score = _score_sequence(perm, start)
        centers = [_aoi_center(a) for a in perm]
        total_dist = float(np.linalg.norm(start - centers[0]))
        for j in range(len(centers) - 1):
            total_dist += float(np.linalg.norm(centers[j] - centers[j + 1]))

        ids = " → ".join(a.id for a in perm)
        value_part = sum(DISCOUNT_GAMMA ** k * _aoi_value(perm[k]) for k in range(len(perm)))
        dist_pen = DISTANCE_PENALTY_ALPHA * total_dist

        scored.append((perm, score, ids, total_dist, value_part, dist_pen))
        marker = ""
        print(f"    [{i + 1}] {ids}")
        print(f"        总距离={total_dist:.0f} km | 折扣价值={value_part:.4f} | 距离惩罚={dist_pen:.4f} | 得分={score:.4f}")

    # 最优路径
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0]
    sep("最优路径", "-")
    print(f"\n  排序结果: {best[2]}")
    print(f"  综合得分: {best[1]:.4f}")
    print(f"  路径总长: {best[3]:.0f} km")

    # 用 AOIRouter 验证
    state = router.sort(aoi_objs, start_pos=start)
    print(f"\n  AOIRouter 输出: {state.aoi_sequence}")
    print(f"  与枚举最优一致: {'[OK]' if state.aoi_sequence == [a.id for a in best[0]] else '[FAIL]'}")

    return state, best


# ================================================================
# Part 3: 多 AOI 连续执行
# ================================================================
def run_sequential_execution(state, best_perm):
    sep("Part 3: 多 AOI 连续执行（含反馈推进）")

    allocator = MultiAOITaskAllocator(solver="cbc", time_limit_s=5.0, verbose=0)

    # 初始输入
    input_data = {
        "aois": ALL_AOIS_RAW,
        "platforms": PLATFORMS_CFG,
        "targets": ALL_TARGETS,
        "staging_position": STAGING_POS,
        "grid_weather": GRID_WEATHER,
        "aoi_route_state": None,
        "execution_feedback": None,
        "cycle_id": 0,
    }

    print(f"\n  将按此顺序执行 {len(best_perm[0])} 个 AOI:")
    centers = [_aoi_center(a) for a in best_perm[0]]
    print(f"    staging ({STAGING_POS[0]:.0f},{STAGING_POS[1]:.0f})", end="")
    for a, c in zip(best_perm[0], centers):
        print(f"  →  {a.id} ({c[0]:.0f},{c[1]:.0f})", end="")
    total_path = float(np.linalg.norm(np.array(STAGING_POS) - centers[0]))
    for j in range(len(centers) - 1):
        total_path += float(np.linalg.norm(centers[j] - centers[j + 1]))
    print(f"\n    总路径 ≈ {total_path:.0f} km")

    execution_log = []
    heli_ammo = {"HF": 16, "RKT": 76, "GUN": 1200}  # H1 + H2 各一半

    for step in range(len(best_perm[0])):
        sep(f"Step {step + 1}: 执行 {state.aoi_sequence[step]}", "-")

        result = allocator.run(input_data)
        plan = result["current_aoi_plan"]

        if plan is None:
            print(f"  状态: {result['status']} —— 全部完成")
            break

        aoi_id = plan["aoi"]
        execution_log.append({
            "aoi": aoi_id,
            "status": plan["solve_status"],
            "objective": plan["objective"],
            "solve_ms": plan["solve_time_ms"],
            "n_tasks": len(plan["tasks"]),
            "tasks": plan["tasks"],
        })

        print(f"\n  求解状态: {plan['solve_status']}")
        print(f"  目标值:   {plan['objective']:.2f}")
        print(f"  求解耗时: {plan['solve_time_ms']:.0f} ms")
        print(f"  任务总数: {len(plan['tasks'])} 条")

        # 分类显示任务
        recon_tasks = [t for t in plan["tasks"] if t["task_type"] == "recon"]
        strike_tasks = [t for t in plan["tasks"] if t["task_type"] == "strike"]

        # 按 UAV 分组
        from collections import defaultdict
        uav_map = defaultdict(list)
        for t in recon_tasks:
            uav_map[t["platform"]].append(f"{t['sensor']}→{t['cell']}")
        print(f"\n  侦察任务 ({len(recon_tasks)} 条):")
        for pid in sorted(uav_map):
            sensors = set(t.split("→")[0] for t in uav_map[pid])
            print(f"    {pid}[{'+'.join(sorted(sensors))}]: {', '.join(uav_map[pid])}")

        if strike_tasks:
            print(f"\n  打击任务 ({len(strike_tasks)} 条):")
            for t in strike_tasks:
                if t["qty"] > 0:
                    print(f"    {t['platform']} → {t['target']} {t['munition']}×{t['qty']} ({t['role']})")
                    heli_ammo[t["munition"]] -= t["qty"]
                else:
                    print(f"    {t['platform']} → {t['target']} 编队支援 ({t['role']})")

            print(f"\n  弹药余量: HF={heli_ammo['HF']}, RKT={heli_ammo['RKT']}, GUN={heli_ammo['GUN']}")
        else:
            print(f"\n  无打击任务（纯侦察或目标已清除）")

        # 模拟执行反馈：当前 AOI 完成
        # 更新打击后的目标状态（标记摧毁目标）
        destroyed_tids = set()
        for t in strike_tasks:
            if t["qty"] > 0:
                destroyed_tids.add(t["target"])
        if destroyed_tids:
            for tgt in ALL_TARGETS:
                if tgt["tid"] in destroyed_tids and tgt["alive"]:
                    tgt["alive"] = False

        # 准备下一次调用的输入
        input_data["aoi_route_state"] = result["aoi_route_state"]
        input_data["execution_feedback"] = {
            "aoi_id": aoi_id,
            "aoi_status": "FINISHED",
            "coverage_rate": 0.85,
            "detected_targets": [t["tid"] for t in ALL_TARGETS if t["alive"] or t["tid"] in destroyed_tids],
            "destroyed_targets": list(destroyed_tids),
            "elapsed_time": 12.0,
        }
        input_data["cycle_id"] = step + 1

        # 更新弹药配置（下一次调用反映消耗）
        input_data["platforms"] = {
            "UAV": PLATFORMS_CFG["UAV"],
            "HELI": {
                **PLATFORMS_CFG["HELI"],
                "munitions": {k: max(0, v // 2) for k, v in heli_ammo.items()},
            },
        }

    # 最后一次调用确认全部完成
    final = allocator.run(input_data)
    print(f"\n  最终状态: {final['status']}")
    if final["status"] == "ALL_AOI_FINISHED":
        print(f"  全部 {len(best_perm[0])} 个 AOI 已执行完毕!")

    return execution_log


# ================================================================
# Part 4: 跨 AOI 对比分析
# ================================================================
def show_cross_aoi_analysis(execution_log):
    sep("Part 4: 跨 AOI 对比分析")

    print(f"\n  {'AOI':<10} {'求解状态':<12} {'目标值':<12} {'求解耗时':<10} {'任务数':<8}")
    print(f"  {'-' * 52}")
    for entry in execution_log:
        aoi = entry["aoi"]
        status = entry["status"]
        obj = entry["objective"]
        ms = entry["solve_ms"]
        n = entry["n_tasks"]
        print(f"  {aoi:<10} {status:<12} {obj:<12.2f} {ms:<10.0f}ms {n:<8}")

    # 传感器使用趋势
    print(f"\n  各 AOI 传感器使用:")
    print(f"  {'AOI':<10} {'EO':<6} {'SAR':<6} {'ESM':<6}")
    for entry in execution_log:
        eo = sum(1 for t in entry["tasks"] if t["task_type"] == "recon" and t["sensor"] == "EO")
        sar = sum(1 for t in entry["tasks"] if t["task_type"] == "recon" and t["sensor"] == "SAR")
        esm = sum(1 for t in entry["tasks"] if t["task_type"] == "recon" and t["sensor"] == "ESM")
        print(f"  {entry['aoi']:<10} {eo:<6} {sar:<6} {esm:<6}")

    # 打击统计
    print(f"\n  各 AOI 打击统计:")
    for entry in execution_log:
        strikes = [t for t in entry["tasks"] if t["task_type"] == "strike" and t.get("qty", 0) > 0]
        supports = [t for t in entry["tasks"] if t["task_type"] == "strike" and t.get("qty", 0) == 0]
        if strikes:
            for s in strikes:
                print(f"  {entry['aoi']}: {s['platform']} → {s['target']} {s['munition']}×{s['qty']} ({s['role']})")
        elif supports:
            print(f"  {entry['aoi']}: 仅编队支援，无实际打击")
        else:
            print(f"  {entry['aoi']}: 无打击分配")

    # 可行性汇总
    sep("测试汇总", "=")
    feasible = sum(1 for e in execution_log if e["status"] in ("OPTIMAL", "FEASIBLE"))
    total = len(execution_log)
    print(f"\n  共处理 {total} 个 AOI")
    print(f"  可行解: {feasible}/{total}")
    all_ok = feasible == total
    print(f"  结果: {'[OK] 全部通过' if all_ok else '[FAIL]'}")

    return all_ok


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 70)
    print("  连续快照测试 v2 —— 多 AOI 排序 + 逐区域连续执行")
    print("=" * 70)
    print(f"  场景: 3 个 AOI, 5 UAV + 2 HELI, 5 个目标")

    # Part 1
    aoi_objs = show_aoi_stats()

    # Part 2
    state, best = show_aoi_sorting(aoi_objs)

    # Part 3
    execution_log = run_sequential_execution(state, best)

    # Part 4
    ok = show_cross_aoi_analysis(execution_log)

    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
