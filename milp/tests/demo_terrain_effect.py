"""
地形影响完整演示：从 terrain_level 输入到 MILP 分配结果
遍历每一步，打印中间计算值和约束效果
"""
import numpy as np
from task_interface import (
    TaskAllocator, make_snapshot, make_grid, make_target,
    make_platform, make_sensor_params, generate_aoi_grids,
)
from core.snapshot import SituationSnapshot


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ====================================================================
# 第 1 步：构造场景 —— 两种地形对比
# ====================================================================
print_separator("第 1 步：构造两个场景 —— 平原 vs 山地，其余条件完全相同")

# AOI A_3_4，中心 (175, 125) km
aoi_row, aoi_col = 3, 4

# 场景 A：全部平原
grids_plain = generate_aoi_grids(aoi_row, aoi_col)
print("\n[场景 A - 全平原] terrain_level:")
for g in grids_plain:
    print(f"  {g.cell_id}: terrain={g.terrain_level}, center=({g.center[0]:.0f}, {g.center[1]:.0f})")

# 场景 B：混合地形
grids_mixed = generate_aoi_grids(aoi_row, aoi_col, terrain_levels={
    "c0": 0,   # 巡逻区：平原
    "c1": 2,   # NW子区：山地
    "c2": 1,   # NE子区：丘陵
    "c3": 0,   # SW子区：平原
    "c4": 2,   # SE子区：山地
})
print("\n[场景 B - 混合地形] terrain_level:")
for g in grids_mixed:
    tl_name = {0: "平原", 1: "丘陵", 2: "山地"}[g.terrain_level]
    print(f"  {g.cell_id}: terrain={g.terrain_level}({tl_name}), center=({g.center[0]:.0f}, {g.center[1]:.0f})")

# 相同的平台和目标
platforms = [
    make_platform("U1", "UAV", (150, -50)),
    make_platform("U2", "UAV", (150, -50)),
    make_platform("U3", "UAV", (150, -50)),
    make_platform("U4", "UAV", (150, -50)),
    make_platform("U5", "UAV", (150, -50)),
    make_platform("H1", "HELI", (150, -50)),
    make_platform("H2", "HELI", (150, -50)),
]

targets = [
    make_target("g1", "RADAR", (162, 112), value=1.0, threat=0.9),  # 在 c1 内
    make_target("g2", "CP",    (188, 112), value=0.85, threat=0.6), # 在 c2 内
    make_target("g3", "AV",    (162, 138), value=0.7, threat=0.5),  # 在 c3 内
    make_target("g4", "AV",    (188, 138), value=0.7, threat=0.5),  # 在 c4 内
]

sensor_params = [
    make_sensor_params("EO",  P0=0.85, R=15.0),
    make_sensor_params("SAR", P0=0.90, R=50.0),
    make_sensor_params("ESM", P0=0.80, R=100.0),
]

staging = np.array([150.0, -50.0])

snap_plain = make_snapshot(
    cycle_id=0, grids=grids_plain, targets=targets,
    platforms=platforms, sensor_params=sensor_params,
    staging_position=staging,
)

snap_mixed = make_snapshot(
    cycle_id=0, grids=grids_mixed, targets=targets,
    platforms=platforms, sensor_params=sensor_params,
    staging_position=staging,
)


# ====================================================================
# 第 2 步：展示 ParameterBuilder 内部的地形系数计算
# ====================================================================
print_separator("第 2 步：地形系数查表 —— terrain_level → 四个修正系数")

from allocation.parameters import ParameterBuilder
from config.settings import GlobalSettings

settings = GlobalSettings()

# 演示查表逻辑
print(f"\n设置中的地形四系数 tuple（下标 = terrain_level）：")
print(f"  terrain_occ   [0,1,2] = {settings.terrain_occ}   ← 平原/丘陵/山地的遮挡系数")
print(f"  terrain_time  [0,1,2] = {settings.terrain_time}  ← 平原/丘陵/山地的扫描时间系数")
print(f"  terrain_dist  [0,1,2] = {settings.terrain_dist}  ← 平原/丘陵/山地的距离系数")
print(f"  terrain_shield[0,1,2] = {settings.terrain_shield} ← 平原/丘陵/山地的掩护系数")

# 展示混合场景的系数向量
params_mixed = ParameterBuilder(snap_mixed, settings)
print(f"\n[场景 B - 混合地形] 每个栅格的四系数：")
print(f"{'Cell':<6} {'terrain':<8} {'occ':<8} {'time':<8} {'dist':<8} {'shield':<8}")
print("-" * 50)
for ci in range(params_mixed.N_C):
    tl = params_mixed.cells[ci].terrain_level
    print(f"{params_mixed.cell_ids[ci]:<6} {tl:<8} "
          f"{params_mixed.occ_factor[ci]:<8.2f} {params_mixed.time_factor[ci]:<8.2f} "
          f"{params_mixed.dist_factor[ci]:<8.2f} {params_mixed.shield_factor[ci]:<8.2f}")


# ====================================================================
# 第 3 步：展示探测概率如何被地形遮挡衰减
# ====================================================================
print_separator("第 3 步：探测概率 = P0 × 天气增益 × 可行性 × 传感器兼容 × 地形遮挡")

# 用场景 B 的参数
print("\n以 SAR 传感器 (P0=0.9) 为例，对比场景 A（全平原）和场景 B（混合地形）：")
params_plain = ParameterBuilder(snap_plain, settings)

sar_idx = 0
for si, sn in enumerate(params_mixed.sensor_names):
    if sn == "SAR":
        sar_idx = si
        break

print(f"\n{'Cell':<6} {'天气w':<8} {'P_det(平原)':<12} {'P_det(混合)':<12} {'衰减原因':<20}")
print("-" * 65)
for ci in range(params_mixed.N_C):
    w = params_mixed.cell_weather[ci]
    p_plain = params_plain.P_det[0, sar_idx, ci]
    p_mixed = params_mixed.P_det[0, sar_idx, ci]
    tl = params_mixed.terrain_level[ci]
    occ = params_mixed.occ_factor[ci]
    reason = f"terrain={tl}, occ_factor={occ:.1f}"
    print(f"{params_mixed.cell_ids[ci]:<6} {w:<8.2f} {p_plain:<12.4f} {p_mixed:<12.4f} {reason:<20}")


# ====================================================================
# 第 4 步：展示扫描时间如何被地形修正
# ====================================================================
print_separator("第 4 步：扫描时间 = 基础时间 × time_factor（地形越复杂越慢）")

print(f"\n以 SAR 传感器 (R=50km, swath=100km) 为例：")
print(f"{'Cell':<6} {'宽(km)':<8} {'高(km)':<8} {'平原耗时(min)':<14} {'混合耗时(min)':<14} {'time_factor':<12}")
print("-" * 65)
for ci in range(params_mixed.N_C):
    cell = params_mixed.cells[ci]
    # 计算基础耗时（无地形修正）
    R_s = 50.0
    swath = 2.0 * R_s
    n_passes = max(1, int(np.ceil(cell.width_km / swath)))
    scan_dist = n_passes * cell.height_km
    base_time = (scan_dist / settings.uav_loiter_speed_kmh) * 60.0
    mixed_time = base_time * params_mixed.time_factor[ci]
    print(f"{params_mixed.cell_ids[ci]:<6} {cell.width_km:<8.0f} {cell.height_km:<8.0f} "
          f"{base_time:<14.1f} {mixed_time:<14.1f} {params_mixed.time_factor[ci]:<12.2f}")


# ====================================================================
# 第 5 步：展示有效距离如何被地形修正
# ====================================================================
print_separator("第 5 步：有效距离 = 直线距离 × dist_factor（山地需要绕飞）")

print(f"\n从集结区 (150, -50) 到各栅格的直线距离 vs 有效距离：")
print(f"{'Cell':<6} {'直线距离(km)':<14} {'有效距离(平原)':<14} {'有效距离(混合)':<16} {'dist_factor':<12}")
print("-" * 65)
for ci in range(params_mixed.N_C):
    cell = params_mixed.cells[ci]
    straight = np.sqrt((cell.center[0]-150)**2 + (cell.center[1]-(-50))**2)
    d_plain = straight  # dist_factor=1.0
    d_mixed = straight * params_mixed.dist_factor[ci]
    print(f"{params_mixed.cell_ids[ci]:<6} {straight:<14.1f} {d_plain:<14.1f} "
          f"{d_mixed:<16.1f} {params_mixed.dist_factor[ci]:<12.2f}")


# ====================================================================
# 第 6 步：展示有效威胁如何被地形掩护修正（打击任务相关）
# ====================================================================
print_separator("第 6 步：有效威胁 = 原始威胁 × (1 - k_shield × shield_factor) —— 打击任务受此影响")

print(f"\nk_shield = {settings.k_shield}（地形掩护强度系数）")
print(f"\n{'目标':<6} {'所在Cell':<8} {'原始威胁':<10} {'地形':<6} {'shield':<8} {'有效威胁':<10} {'变化':<10}")
print("-" * 65)
for gi in range(params_mixed.N_G):
    ci = params_mixed.target_cell_idx[gi]
    tid = params_mixed.target_types[gi] if gi < len(params_mixed.target_types) else f"g{gi+1}"
    tl = params_mixed.terrain_level[ci]
    tl_name = {0: "平原", 1: "丘陵", 2: "山地"}[tl]
    raw = params_mixed.target_threats[gi]
    eff = params_mixed.effective_threat[gi]
    shield = params_mixed.shield_factor[ci]
    delta = (eff - raw)
    print(f"{tid:<6} {params_mixed.cell_ids[ci]:<8} {raw:<10.2f} {tl_name:<6} {shield:<8.2f} {eff:<10.2f} {delta:+<10.2f}")

print("\n→ 山地里的目标威胁降低（掩护效应），MILP 在风险约束下更可能选山地目标")


# ====================================================================
# 第 7 步：求解两个场景，对比分配结果
# ====================================================================
print_separator("第 7 步：MILP 求解 —— 对比平原 vs 混合地形的分配结果")

allocator = TaskAllocator(solver="cbc", time_limit_s=5.0, verbose=0)

print("\n--------------- 场景 A：全平原 ---------------")
plan_plain = allocator.solve(snap_plain)
print(f"求解状态: {plan_plain.status}")
print(f"目标函数值: {plan_plain.objective:.2f}")
print(f"侦察分配 ({len(plan_plain.recon_assignments)} 条):")
for ra in plan_plain.recon_assignments:
    print(f"  {ra.pid} 挂载 {ra.sensor} → 扫描 {ra.cell} (role={ra.role})")
print(f"打击分配 ({len(plan_plain.strike_assignments)} 条):")
for sa in plan_plain.strike_assignments:
    print(f"  {sa.pid} → {sa.target} 用 {sa.munition}×{sa.qty} (role={sa.role})")

print("\n--------------- 场景 B：混合地形 ---------------")
plan_mixed = allocator.solve(snap_mixed)
print(f"求解状态: {plan_mixed.status}")
print(f"目标函数值: {plan_mixed.objective:.2f}")
print(f"侦察分配 ({len(plan_mixed.recon_assignments)} 条):")
for ra in plan_mixed.recon_assignments:
    print(f"  {ra.pid} 挂载 {ra.sensor} → 扫描 {ra.cell} (role={ra.role})")
print(f"打击分配 ({len(plan_mixed.strike_assignments)} 条):")
for sa in plan_mixed.strike_assignments:
    print(f"  {sa.pid} → {sa.target} 用 {sa.munition}×{sa.qty} (role={sa.role})")


# ====================================================================
# 第 8 步：对比分析
# ====================================================================
print_separator("第 8 步：地形影响分析")

# 获取每个场景的侦察栅格偏好
plain_cells = set(ra.cell for ra in plan_plain.recon_assignments)
mixed_cells = set(ra.cell for ra in plan_mixed.recon_assignments)

print(f"\n侦察覆盖对比：")
print(f"  平原场景覆盖的栅格: {sorted(plain_cells)}")
print(f"  混合场景覆盖的栅格: {sorted(mixed_cells)}")

# 分析山地栅格是否被避开
mountain_cells = {"c1", "c4"}  # 混合场景中的山地区域
plain_cells_in_mixed = {"c0", "c2", "c3"}  # 非山地

mountain_selected = mixed_cells & mountain_cells
plain_selected = mixed_cells & plain_cells_in_mixed

print(f"\n  混合场景中，EO/SAR 选择的栅格：")
print(f"    平原/丘陵区被选: {sorted(plain_selected)}")
print(f"    山地区被选:     {sorted(mountain_selected) if mountain_selected else '(无)'}")
print(f"  → 山地栅格的 P_det 低（遮挡）、T_scan 长（时间成本高）、D_eff 远（距离成本高）")
print(f"  → MILP 自然倾向选择探测效率高的平原/丘陵栅格")

# 分析打击任务
print(f"\n打击任务对比：")
plain_strikes = {sa.target for sa in plan_plain.strike_assignments}
mixed_strikes = {sa.target for sa in plan_mixed.strike_assignments}
print(f"  平原场景打击目标: {sorted(plain_strikes)}")
print(f"  混合场景打击目标: {sorted(mixed_strikes)}")

# 检查山地里的目标是否被选
print(f"\n  各目标所在地形：")
for gi in range(params_mixed.N_G):
    ci = params_mixed.target_cell_idx[gi]
    tl = params_mixed.terrain_level[ci]
    tl_name = {0: "平原", 1: "丘陵", 2: "山地"}[tl]
    tid = f"g{gi+1}"
    eff_threat = params_mixed.effective_threat[gi]
    cell_id = params_mixed.cell_ids[ci]
    print(f"    {tid} 在 {cell_id}（{tl_name}），原始威胁={params_mixed.target_threats[gi]:.2f}，有效威胁={eff_threat:.2f}")

print(f"\n→ 打击任务受地形影响的方式：")
print(f"  1. _c12_survivability 约束：Σ effective_threat × z ≤ theta_max")
print(f"     山地目标有效威胁更低 → 直升机可以'更安全'地攻击")
print(f"  2. _c15_heli_transit 约束：转场距离 × dist_factor")
print(f"     山地目标有效距离更远 → 可能超出转场时限")
print(f"  3. 目标函数风险项：-λ_T × Σ thr×(1-η) × z")
print(f"     注意：此处仍用 occlusion_matrix(η_hg)，不是 shield_factor")
print(f"     地形掩护通过 effective_threat 影响的是约束，不是风险惩罚项")

print_separator("演示结束")
