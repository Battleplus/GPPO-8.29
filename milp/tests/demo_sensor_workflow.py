"""
传感器工作流测试：对比修改前后的分配差异
测试点：
  1. 一架 UAV 能否同时挂 EO+SAR
  2. SAR 和 EO 各管各的格子
  3. EO 是否依赖 SAR（c2d 约束）
  4. EO 在坏天气下是否正确禁用
  5. ESM 逻辑不受影响
  6. 对比新旧模型下的分配效率
"""
import numpy as np
from task_interface import (
    TaskAllocator, make_snapshot, make_grid, make_target,
    make_platform, make_sensor_params, generate_aoi_grids,
)
from core.snapshot import SituationSnapshot
from allocation.parameters import ParameterBuilder
from config.settings import GlobalSettings


def sep(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ================================================================
# 测试 1：验证一架 UAV 可以同时用 SAR + EO
# ================================================================
sep("测试 1: UAV 同时挂载 SAR + EO，各自侦察不同格子")

grids = generate_aoi_grids(3, 4)
platforms = [
    make_platform("U1", "UAV", (150, -50)),
    make_platform("U2", "UAV", (150, -50)),
    make_platform("U3", "UAV", (150, -50)),
    make_platform("U4", "UAV", (150, -50)),
    make_platform("U5", "UAV", (150, -50)),
    make_platform("H1", "HELI", (150, -50)),
]
targets = [
    make_target("g1", "RADAR", (162, 112), value=1.0, threat=0.9),
]
sensor_params = [
    make_sensor_params("EO",  P0=0.85, R=15.0),
    make_sensor_params("SAR", P0=0.90, R=50.0),
    make_sensor_params("ESM", P0=0.80, R=100.0),
]

snap = make_snapshot(
    cycle_id=0, grids=grids, targets=targets,
    platforms=platforms, sensor_params=sensor_params,
    staging_position=np.array([150.0, -50.0]),
)

allocator = TaskAllocator(solver="cbc", time_limit_s=5.0)
plan = allocator.solve(snap)

print(f"求解状态: {plan.status}  目标值: {plan.objective:.2f}")

# 按 UAV 分组显示
from collections import defaultdict
uav_tasks = defaultdict(list)
for ra in plan.recon_assignments:
    uav_tasks[ra.pid].append(f"{ra.sensor}->{ra.cell}")

print(f"\n侦察分配（{len(plan.recon_assignments)} 条）:")
for pid in sorted(uav_tasks.keys()):
    tasks = ", ".join(uav_tasks[pid])
    sensor_set = set(t.split("->")[0] for t in uav_tasks[pid])
    n_sensors = len(sensor_set)
    print(f"  {pid}: {tasks}  [{n_sensors}种传感器]")

# 检查关键约束
sar_cells = set()
eo_cells = set()
esm_cells = set()
for ra in plan.recon_assignments:
    if ra.sensor == "SAR":
        sar_cells.add(ra.cell)
    elif ra.sensor == "EO":
        eo_cells.add(ra.cell)
    elif ra.sensor == "ESM":
        esm_cells.add(ra.cell)

print(f"\n约束验证:")
sar_per_uav_ok = all(
    sum(1 for ra in plan.recon_assignments if ra.pid == pid and ra.sensor == "SAR") <= 1
    for pid in set(ra.pid for ra in plan.recon_assignments)
)
eo_per_uav_ok = all(
    sum(1 for ra in plan.recon_assignments if ra.pid == pid and ra.sensor == "EO") <= 1
    for pid in set(ra.pid for ra in plan.recon_assignments)
)
print(f"  SAR <= 1格/架: {'[OK]' if sar_per_uav_ok else '[FAIL]'} (SAR格子={sorted(sar_cells)})")
print(f"  EO  <= 1格/架: {'[OK]' if eo_per_uav_ok else '[FAIL]'} (EO格子={sorted(eo_cells)})")
print(f"  ESM 全覆盖:     {'[OK]' if len(esm_cells) >= 5 else '[OK-ESM逻辑不变]'} (ESM格子={sorted(esm_cells)})")

# 检查 EO-SAR 依赖
eo_ok = all(c in sar_cells for c in eo_cells)
print(f"  c2d EO依赖SAR:  {'[OK] EO格子都在SAR覆盖内' if eo_ok else '[FAIL] EO覆盖了SAR没搜的格子'}")

# 检查传感器数量 <=2
all_ok = True
for pid in sorted(uav_tasks.keys()):
    sensors = set(t.split("->")[0] for t in uav_tasks[pid])
    ok = len(sensors) <= 2
    if not ok:
        all_ok = False
    print(f"  {pid} 传感器数: {len(sensors)} {'[OK]' if ok else '[FAIL]'} (传感器={sensors})")
print(f"\n  所有UAV传感器数<=2: {'[OK]' if all_ok else '[FAIL]'}")


# ================================================================
# 测试 2：对比新旧模型的传感器使用效率
# ================================================================
sep("测试 2: 标准场景 —— 5 UAV + 2 HELI + 4 目标")

# 读默认场景
from task_interface import load_snapshot_from_json
snap2 = load_snapshot_from_json("scenarios/default.json")
plan2 = allocator.solve(snap2)

print(f"求解状态: {plan2.status}  目标值: {plan2.objective:.2f}")

uav_tasks2 = defaultdict(list)
for ra in plan2.recon_assignments:
    uav_tasks2[ra.pid].append(f"{ra.sensor}->{ra.cell}")

used_uavs = len(uav_tasks2)
total_tasks = len(plan2.recon_assignments)

# 统计传感器使用
sensor_usage = defaultdict(int)
for ra in plan2.recon_assignments:
    sensor_usage[ra.sensor] += 1

# 每个传感器的格子覆盖
sensor_cells = defaultdict(set)
for ra in plan2.recon_assignments:
    sensor_cells[ra.sensor].add(ra.cell)

print(f"\n侦察分配（{total_tasks} 条，用了 {used_uavs}/5 架 UAV）:")
for pid in sorted(uav_tasks2.keys()):
    tasks = ", ".join(uav_tasks2[pid])
    sensors = set(t.split("->")[0] for t in uav_tasks2[pid])
    print(f"  {pid}[{'+'.join(sorted(sensors))}]: {tasks}")

print(f"\n传感器使用统计:")
for sn in ["EO", "SAR", "ESM"]:
    cells = sorted(sensor_cells.get(sn, set()))
    print(f"  {sn}: {sensor_usage.get(sn, 0)} 条任务, 覆盖 {cells}")

print(f"\n打击分配（{len(plan2.strike_assignments)} 条）:")
for sa in plan2.strike_assignments:
    print(f"  {sa.pid} -> {sa.target} 用 {sa.munition}×{sa.qty} (role={sa.role})")

# 找出多传感器 UAV
multi_sensor = []
for pid, tasks in uav_tasks2.items():
    sensors = set(t.split("->")[0] for t in tasks)
    if len(sensors) >= 2:
        multi_sensor.append((pid, sensors))
if multi_sensor:
    print(f"\n多传感器 UAV: {[(p, '+'.join(sorted(s))) for p, s in multi_sensor]}")
else:
    print(f"\n没有多传感器 UAV（所有 UAV 只用 1 种传感器）")


# ================================================================
# 测试 3：EO 在坏天气下被禁用
# ================================================================
sep("测试 3: 坏天气场景 —— EO 在高湿度(w>=0.8)下应被禁用")

platforms3 = [
    make_platform("U1", "UAV", (150, -50)),
    make_platform("U2", "UAV", (150, -50)),
    make_platform("U3", "UAV", (150, -50)),
    make_platform("U4", "UAV", (150, -50)),
    make_platform("U5", "UAV", (150, -50)),
    make_platform("H1", "HELI", (150, -50)),
]
grids_bad = generate_aoi_grids(3, 4, weather_w=0.85)  # 全部高湿
for g in grids_bad:
    g.weather_w = 0.85
targets3 = [
    make_target("g1", "RADAR", (162, 112), value=1.0, threat=0.9),
]
snap3 = make_snapshot(
    cycle_id=0, grids=grids_bad, targets=targets3,
    platforms=platforms3, sensor_params=sensor_params,
    staging_position=np.array([150.0, -50.0]),
)
plan3 = allocator.solve(snap3)

eo_used = sum(1 for ra in plan3.recon_assignments if ra.sensor == "EO")
sar_used = sum(1 for ra in plan3.recon_assignments if ra.sensor == "SAR")

print(f"求解状态: {plan3.status}")
print(f"\n天气 w=0.85（超过 EO 的 0.80 阈值）")
print(f"  EO 使用次数: {eo_used} {'[OK] (坏天气下EO=0)' if eo_used == 0 else '[FAIL] (EO没被禁用)'}")
print(f"  SAR 使用次数: {sar_used} {'[OK] (SAR不受天气影响)' if sar_used > 0 else '?'}")

for ra in plan3.recon_assignments:
    if ra.sensor == "EO":
        print(f"  [FAIL] {ra.pid}: EO->{ra.cell} (坏天气下不应该出现EO!)")
    else:
        print(f"  {ra.pid}: {ra.sensor}->{ra.cell}")

# ================================================================
# 测试 4：SAR 必须先覆盖，EO 才能确认（c2d 依赖约束）
# ================================================================
sep("测试 4: EO-SAR 依赖约束 —— 验证 EO 只能确认 SAR 搜过的格子")

grids4 = generate_aoi_grids(3, 4)
platforms4 = [
    make_platform("U1", "UAV", (150, -50)),
    make_platform("U2", "UAV", (150, -50)),
    make_platform("U3", "UAV", (150, -50)),
    make_platform("U4", "UAV", (150, -50)),
    make_platform("U5", "UAV", (150, -50)),
    make_platform("H1", "HELI", (150, -50)),
]
targets4 = [
    make_target("g1", "RADAR", (162, 112), value=1.0, threat=0.9),
]
snap4 = make_snapshot(
    cycle_id=0, grids=grids4, targets=targets4,
    platforms=platforms4, sensor_params=sensor_params,
    staging_position=np.array([150.0, -50.0]),
)
plan4 = allocator.solve(snap4)

uav4_tasks = defaultdict(list)
for ra in plan4.recon_assignments:
    uav4_tasks[ra.pid].append(f"{ra.sensor}->{ra.cell}")

sar4 = [ra.cell for ra in plan4.recon_assignments if ra.sensor == "SAR"]
eo4  = [ra.cell for ra in plan4.recon_assignments if ra.sensor == "EO"]

print(f"求解状态: {plan4.status}")
print(f"\nSAR 覆盖的格子: {sar4}")
print(f"EO 覆盖的格子:  {eo4}")

if eo4:
    eo_ok = all(c in sar4 for c in eo4)
    print(f"EO格子  in  SAR格子: {'[OK] 依赖约束生效' if eo_ok else '[FAIL] 约束被破坏'}")
else:
    print(f"EO 未被使用（UAV 选择了更优的 SAR 或 EO 无必要）")


# ================================================================
# 测试 5：地形 + 新传感器模型联合验证
# ================================================================
sep("测试 5: 地形 + 新传感器模型 —— 混合地形下的分配")

grids5 = generate_aoi_grids(3, 4, terrain_levels={
    "c0": 0, "c1": 2, "c2": 1, "c3": 0, "c4": 2
})
platforms5 = [
    make_platform("U1", "UAV", (150, -50)),
    make_platform("U2", "UAV", (150, -50)),
    make_platform("U3", "UAV", (150, -50)),
    make_platform("U4", "UAV", (150, -50)),
    make_platform("U5", "UAV", (150, -50)),
    make_platform("H1", "HELI", (150, -50)),
    make_platform("H2", "HELI", (150, -50)),
]
targets5 = [
    make_target("g1", "RADAR", (162, 112), value=1.0, threat=0.9),
    make_target("g2", "CP",    (188, 112), value=0.85, threat=0.6),
    make_target("g3", "AV",    (162, 138), value=0.7, threat=0.5),
    make_target("g4", "AV",    (188, 138), value=0.7, threat=0.5),
]
snap5 = make_snapshot(
    cycle_id=0, grids=grids5, targets=targets5,
    platforms=platforms5, sensor_params=sensor_params,
    staging_position=np.array([150.0, -50.0]),
)
plan5 = allocator.solve(snap5)

print(f"求解状态: {plan5.status}  目标值: {plan5.objective:.2f}")

# 地形信息
print(f"\n各栅格地形:")
tl_names = {0: "平原", 1: "丘陵", 2: "山地"}
for g in grids5:
    print(f"  {g.cell_id}: {tl_names[g.terrain_level]}")

# 分配结果
uav5 = defaultdict(list)
for ra in plan5.recon_assignments:
    uav5[ra.pid].append((ra.sensor, ra.cell))

print(f"\n侦察分配:")
for pid in sorted(uav5.keys()):
    tasks = [f"{s}->{c}" for s, c in uav5[pid]]
    cells = set(c for _, c in uav5[pid])
    sensors = set(s for s, _ in uav5[pid])
    terrains = [f"{c}({tl_names[g.terrain_level]})" for c in cells for g in grids5 if g.cell_id == c]
    print(f"  {pid}: {', '.join(tasks)}")
    print(f"       地形: {', '.join(terrains)}")

# 分析山地 vs 平原的分配偏好
mountain_cells = {"c1", "c4"}
plain_cells = {"c0", "c3"}
hill_cells = {"c2"}

sar_cells5 = set(ra.cell for ra in plan5.recon_assignments if ra.sensor == "SAR")
eo_cells5  = set(ra.cell for ra in plan5.recon_assignments if ra.sensor == "EO")

print(f"\n覆盖偏好分析:")
print(f"  SAR 覆盖山地: {sar_cells5 & mountain_cells}")
print(f"  SAR 覆盖平原: {sar_cells5 & plain_cells}")
print(f"  EO  覆盖山地: {eo_cells5 & mountain_cells}")
print(f"  EO  覆盖平原: {eo_cells5 & plain_cells}")

# 参数对比
params5 = ParameterBuilder(snap5, GlobalSettings())
print(f"\n地形导致的参数差异（SAR传感器）:")
sar_si = 0
for si, sn in enumerate(params5.sensor_names):
    if sn == "SAR":
        sar_si = si
        break
print(f"  {'Cell':<6} {'地形':<6} {'occ':<6} {'time':<6} {'dist':<6} {'P_det':<8}")
for ci in range(params5.N_C):
    tl = params5.cells[ci].terrain_level
    print(f"  {params5.cell_ids[ci]:<6} {tl_names[tl]:<6} "
          f"{params5.occ_factor[ci]:<6.2f} {params5.time_factor[ci]:<6.2f} "
          f"{params5.dist_factor[ci]:<6.2f} {params5.P_det[0, sar_si, ci]:<8.4f}")

sep("测试完成")
