cd "d:/my_document/研究生/项目/54所直升机和无人机协同项目/代码-v2"

# 方式 A: pytest（最规范）
python -m pytest tests/test_end_to_end.py -v

# 方式 B: 直接运行脚本
python tests/test_end_to_end.py



================================================================================
有注释版测试
================================================================================
python tests/test_end_to_end.py


模块化的调用
from config.settings import GlobalSettings, SolverType
from core.snapshot import SituationSnapshot, GridInfo, TargetInfo, PlatformInfo, SensorParams
from allocation.milp_allocator import MILPAllocator

# 1. 一次性初始化（程序启动时）
settings = GlobalSettings(
    active_solver=SolverType.CBC,
    solver_time_limit_s=3.0,
    verbose=1,  # 主循环中建议设为 0 减少终端输出
)
allocator = MILPAllocator(settings)

# 2. 每轮仿真循环
def main_loop():
    for cycle_id in range(max_cycles):
        # 态势理解模块产出快照
        snapshot = situation_awareness_module.get_snapshot()
        # 或者手动构造:
        snapshot = SituationSnapshot(
            cycle_id=cycle_id,
            timestamp=current_time,
            grids=[...],        # List[GridInfo]
            targets=[...],      # List[TargetInfo]
            platforms=[...],    # List[PlatformInfo]
            sensor_params=[...],# List[SensorParams]
            commander_AOI=[...],
        )

        # 调用任务分配
        plan = allocator.solve(snapshot)

        # 执行分配方案
        execute_recon(plan.recon_assignments)    # List[ReconAssignment]
        execute_strike(plan.strike_assignments)  # List[StrikeAssignment]

        # 检查求解状态
        if plan.status == "INFEASIBLE":
            handle_infeasible()
================================================================================
新接口的用法
================================================================================
from task_interface import generate_aoi_grids, make_snapshot

# 选择任意 AOI（行 1-6, 列 1-6）
grids = generate_aoi_grids(aoi_row=2, aoi_col=5)  # A_2_5

snapshot = make_snapshot(
    grids=grids,
    platforms=[...],                        # pos 全设为集结区
    staging_position=np.array([150, -50]),  # 集结区坐标
    commander_AOI=["A_2_5"],
)


=========================================================================================
可视化前端（Streamlit）
=========================================================================================
# 1. 安装依赖（含 streamlit / plotly）
pip install -r requirements.txt

# 2. 启动可视化页面
streamlit run frontend_app.py

# 3. 使用
#  - 左侧「运行任务分配」按钮：调用 MILP 求解默认场景
#    （5 UAV + 2 HELI + AOI A_3_4 + 集结区 [150,-50]）
#  - 主区「时间步」滑块：播放 t0(集结区) → t1(分配完成)
#    → t2(移动中) → t3(到达栅格/目标) 的全过程
#  - 下方表格：侦察分配 / 打击分配明细

# 原有测试不受影响：
python -m pytest tests/test_end_to_end.py -v



=================================================================================
场景修改
================================================================================

tests/test_end_to_end.py 文件中
直接修改 build_sample_snapshot()（推荐）



================================================================================
===进行模块调用 V2
===============================================================================
第一步：创建分配器（全局复用一次）

from task_interface import TaskAllocator

allocator = TaskAllocator(
    solver="cbc",        # cbc / highs / ortools / gurobi
    time_limit_s=5.0,    # 单次求解时间上限
    verbose=0,           # 0=静默 1=摘要 2=详细
)
第二步：构造态势快照（每轮仿真一次）

import numpy as np
from task_interface import make_snapshot, make_platform, make_target, generate_aoi_grids

# 栅格：选 AOI 区域
grids = generate_aoi_grids(aoi_row=2, aoi_col=5)

# 修改天气（可选）
grids[1].weather_w = 0.15  # c1 晴好
grids[2].weather_w = 0.55  # c2 多云

# 平台
platforms = [
    make_platform("U1", "UAV", pos_xy=(150, -50)),
    make_platform("U2", "UAV", pos_xy=(150, -50)),
    make_platform("H1", "HELI", pos_xy=(150, -50)),
]

# 目标
targets = [
    make_target("g1", "RADAR", pos_xy=(270, 260), value=1.0, threat=0.9),
    make_target("g2", "CP",    pos_xy=(35, 230),  value=0.85, threat=0.6),
]

# 装配
snapshot = make_snapshot(
    grids=grids,
    platforms=platforms,
    targets=targets,
    staging_position=np.array([150, -50]),
    commander_AOI=["A_2_5"],
)
第三步：求解 + 读取结果

plan = allocator.solve(snapshot)

print(plan.status)       # OPTIMAL / FEASIBLE / TIME_LIMIT / INFEASIBLE
print(plan.objective)     # 目标函数值
print(plan.solve_time_ms) # 求解耗时

# 侦察分配
for ra in plan.recon_assignments:
    print(ra.pid, ra.sensor, ra.cell, ra.role)

# 打击分配
for sa in plan.strike_assignments:
    print(sa.pid, sa.target, sa.munition, sa.qty, sa.role)

================================================================================
===现在往里面加了json测试文件
================================================================================
只需要将场景写成下面的格式就行了
{
  "scenario_name": "场景名(仅备注用)",
  "aoi": { "row": 3, "col": 4 },
  "staging_position": [150, -50],
  "commander_AOI": ["A_3_4"],
  "grid_weather": { "c0": 0.2, "c1": 0.15, "c2": 0.4, "c3": 0.55, "c4": 0.7 },
  "platforms": {
    "UAV": { "count": 5, "sensors": ["EO","SAR","ESM"],
             "munitions": {"HF":0,"RKT":0,"GUN":0} },
    "HELI": { "count": 2, "sensors": ["MMW","EOIR"],
              "munitions": {"HF":16,"RKT":76,"GUN":1200} }
  },
  "targets": [
    { "tid": "g1", "type": "RADAR", "pos": [270, 260], "value": 1.0, "threat": 0.9 }
  ],
  "sensor_params": [...]  // 可选, 有默认值
}

使用方式

# 测试：自动发现全部 JSON 场景
python -m pytest tests/test_end_to_end.py::test_all_json_scenarios -v

# 前端：侧边栏下拉选择场景 → 点运行
streamlit run frontend_app.py

# 代码调用：
from task_interface import load_snapshot_from_json, TaskAllocator

snap = load_snapshot_from_json("scenarios/heavy_strike.json")
plan = TaskAllocator(solver="cbc").solve(snap)

=======================================================================================
最新调用说明
=======================================================================================
方式一：用工厂函数构造输入（适合上层代码已有态势数据）

import numpy as np
from task_interface import (
    TaskAllocator, make_snapshot, make_platform,
    make_target, generate_aoi_grids,
)

# 1. 创建求解器
allocator = TaskAllocator(solver="cbc", time_limit_s=5.0, verbose=0)

# 2. 构造输入: 5个网格（c0巡逻区 + c1~c4四个子区域）
grids = generate_aoi_grids(aoi_row=3, aoi_col=4)

# 3. 构造平台
platforms = []
# 5架无人机 (编号 U1~U5)
for i in range(1, 6):
    platforms.append(make_platform(
        f"U{i}", "UAV",
        pos_xy=(150, -50),      # 出发位置 (x, y) km
        sensors=["EO","SAR","ESM"],
        munitions={},            # 无人机不携带武器
        alt=2.0,                 # 高度 km
    ))
# 2架直升机 (编号 H1~H2)
for i in range(1, 3):
    platforms.append(make_platform(
        f"H{i}", "HELI",
        pos_xy=(150, -50),
        sensors=["MMW","EOIR"],
        munitions={"HF": 16, "RKT": 76, "GUN": 1200},  # 弹药量
        alt=3.0,
    ))

# 4. 构造目标
targets = [
    make_target("g1", "RADAR", pos_xy=(270, 260), value=1.0, threat=0.9),
    make_target("g2", "CP",    pos_xy=(35,  230), value=0.85, threat=0.6),
    make_target("g3", "AV",    pos_xy=(220, 45),  value=0.7,  threat=0.5,
                velocity=(0.02, 0.01), pos_cov=[[0.2, 0], [0, 0.2]]),
]

# 5. 组装快照
snapshot = make_snapshot(
    grids=grids,
    platforms=platforms,
    targets=targets,
    staging_position=np.array([150, -50]),
    commander_AOI=["A_3_4"],
)

# 6. 求解
plan = allocator.solve(snapshot)

# 7. 获取执行清单
order = make_execution_order(plan, snapshot)

# 8. 使用结果
print(plan.status)       # "OPTIMAL" | "FEASIBLE" | "INFEASIBLE"
print(plan.objective)    # 目标函数值
print(plan.solve_time_ms)  # 求解耗时(毫秒)

for task in order["tasks"]:
    print(task["platform"], task["task_type"], task["role"])
方式二：从 JSON 文件加载输入（适合已有 JSON 格式数据）

from task_interface import (
    load_snapshot_from_json, TaskAllocator,
    make_execution_order, save_execution_order,
)

# 加载 JSON
snapshot = load_snapshot_from_json("scenarios/default.json")

# 求解
plan = TaskAllocator(solver="cbc", time_limit_s=3.0).solve(snapshot)

# 保存结果
save_execution_order(plan, snapshot, "output/execution_order.json")
输入 JSON 格式
上层代码传入的 JSON 结构如下（参考 situation_input_template.json）：


{
  "scenario_name": "...",
  "staging_position": [150, -50],
  "commander_AOI": ["A_3_4"],
  "grid_weather": { "c0": 0.2, "c1": 0.15, "c2": 0.4, "c3": 0.55, "c4": 0.7 },
  "platforms": {
    "UAV": {"count": 5, "pos": [150,-50], "sensors": ["EO","SAR","ESM"],
            "munitions": {}, "alt": 2.0},
    "HELI": {"count": 2, "pos": [150,-50], "sensors": ["MMW","EOIR"],
             "munitions": {"HF": 16, "RKT": 76, "GUN": 1200}, "alt": 3.0}
  },
  "targets": [
    {"tid": "g1", "type": "RADAR", "pos": [270,260], "value": 1.0, "threat": 0.9,
     "confirmed": false, "alive": true}
  ]
}
字段	含义
staging_position	平台出发点坐标 [x, y] (km)
commander_AOI	指挥员关注的任务区域列表
grid_weather	5个网格的天气权重 (0~1, 越高越差)
platforms.UAV/HELI	无人直升机平台: 数量、位置、传感器、弹药
targets	目标列表: 类型(RADAR/CP/AV)、位置、价值、威胁
targets[*].alive	目标是否存活
targets[*].confirmed	目标是否已确认（未确认为侦察诱因）
输出执行清单格式
返回的 order dict 结构如下（参考 execution_order_template.json）：


{
  "cycle_id": 0,
  "timestamp": 0.0,
  "status": "OPTIMAL",
  "tasks": [
    { "platform": "U1", "task_type": "recon",  "cell": "c0", "sensor": "ESM", "role": "area_scan" },
    { "platform": "U2", "task_type": "recon",  "cell": "c1", "sensor": "EO",  "role": "subarea_search" },
    { "platform": "H1", "task_type": "strike", "target": "g1", "munition": "HF", "qty": 2, "role": "lead" },
    { "platform": "H2", "task_type": "strike", "target": "g1", "munition": "",   "qty": 0, "role": "wing_support" }
  ]
}
任务字段	含义
platform	执行平台编号 (U1-U5/H1-H2)
task_type	"recon" 侦察任务 或 "strike" 打击任务
cell	侦察网格编号 (c0 巡逻区, c1-c4 子区域)
sensor	使用的传感器 (EO/SAR/ESM/MMW/IR)
role	侦察: area_scan/subarea_search；打击: lead/wing_support
target	打击目标编号
munition	使用的弹药类型 (HF/RKT/GUN)
qty	弹药使用数量
AllocationPlan 原始输出
如果你想直接使用 dataclass 输出而不是执行清单：


plan = allocator.solve(snapshot)

# plan.recon_assignments: List[ReconAssignment]
for ra in plan.recon_assignments:
    ra.pid     # 平台ID
    ra.sensor  # 传感器类型
    ra.cell    # 网格编号
    ra.role    # 角色

# plan.strike_assignments: List[StrikeAssignment]
for sa in plan.strike_assignments:
    sa.pid       # 平台ID
    sa.target    # 目标ID
    sa.munition  # 弹药类型 (空字符串表示僚机支援，不开火)
    sa.qty       # 弹药数量 (0表示僚机支援)
    sa.role      # "lead" 或 "wing_support"
参数配置
参数	默认值	含义
solver	"cbc"	求解器: cbc/gurobi/ortools/highs
time_limit_s	3.0	单次求解时间上限 (秒)
mip_gap	1e-3	MIP 最优性间隙
verbose	0	0=静默 1=摘要 2=详细
弹药需求参考
目标类型	所需直升机数	所需弹药
RADAR (雷达)	2	HF × 2
CP (指挥所)	2	HF × 1, RKT × 4
AV (装甲车)	1	HF × 1, RKT × 2, GUN × 50
总结：上层代码只需拿到态势数据 → 调用 make_* 工厂函数或直接传 JSON → TaskAllocator.solve(snapshot) → 拿到 AllocationPlan 或 make_execution_order() 输出的任务清单。