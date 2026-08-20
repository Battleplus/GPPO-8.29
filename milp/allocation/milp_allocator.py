"""
任务分配模块对外唯一入口。

职责:
  1. 根据 GlobalSettings.active_solver 实例化对应求解器
  2. 提供 solve(snapshot) → AllocationPlan 的统一调用接口
  3. 管理热启动缓存（上一轮最优解）
  4. 集成 DisplayManager 实现终端进度输出

对外接口:
  - MILPAllocator  — 任务分配模块入口类

使用示例:
    >>> from config.settings import GlobalSettings, SolverType
    >>> from allocation import MILPAllocator
    >>> settings = GlobalSettings(active_solver=SolverType.CBC, verbose=1)
    >>> allocator = MILPAllocator(settings)
    >>> plan = allocator.solve(snapshot)
"""

import time
from config.settings import GlobalSettings
from core.snapshot import SituationSnapshot
from core.allocation import AllocationPlan
from allocation.solver_interface import SolverFactory
from allocation.model_builder import ModelBuilder
from utils.display import DisplayManager


class MILPAllocator:
    """
    任务分配模块对外唯一入口。

    求解器切换: 仅需修改 settings.active_solver 枚举值。
    终端显示: 通过 settings.verbose 控制输出级别 (0/1/2)。

    属性:
        settings: 全局配置
        _solver: 当前激活的 SolverInterface 实例
        _last_solution: 上一轮 AllocationPlan（用于热启动）
        display: DisplayManager 终端显示管理器
    """

    def __init__(self, settings: GlobalSettings):
        """
        Args:
            settings: 全局配置对象，启动时校验求解器合法性
        """
        self.settings = settings
        settings.validate_solver()
        self._solver = SolverFactory.create(
            settings.active_solver, settings
        )
        self._last_solution: AllocationPlan = None
        self.display = DisplayManager(settings)       # 终端显示管理器

    def solve(self, snapshot: SituationSnapshot) -> AllocationPlan:
        """
        主调用接口。

        流程:
            1. 显示态势快照
            2. 构建 MILP 模型（参数计算 → 变量 → 约束 → 目标）
            3. 热启动注入（如有上轮解）
            4. 求解
            5. 提取分配方案
            6. 显示求解结果与分配方案

        Args:
            snapshot: 由态势理解模块产出的 SituationSnapshot

        Returns:
            AllocationPlan（含侦察分配列表 + 打击分配列表）
        """
        t0 = time.time()

        # ---- Step 1: 显示态势快照 ----
        self.display.show_snapshot(snapshot)

        # ---- Step 2: 构建 MILP 模型 ----
        builder = ModelBuilder(snapshot, self.settings, self._solver,
                               self.display)
        model, vars_ = builder.build()

        # ---- Step 3: 准备热启动 ----
        warm_start = (self._build_warm_start(self._last_solution)
                      if self._last_solution else None)

        # ---- Step 4: 求解 ----
        plan = self._solver.solve(
            model,
            warm_start=warm_start,
            time_limit_s=self.settings.solver_time_limit_s,
            mip_gap=self.settings.solver_mip_gap,
        )

        # ---- Step 5: 从变量值提取分配方案 ----
        plan.cycle_id = snapshot.cycle_id
        builder.extract_assignments(plan, vars_, model)

        # ---- Step 6: 显示结果 ----
        total_ms = (time.time() - t0) * 1000
        self.display.show_solve_result(plan, total_ms)
        self.display.show_assignments(plan)
        self.display.show_cycle_summary(snapshot.cycle_id, plan)

        # ---- Step 7: 缓存结果用于下轮热启动 ----
        self._last_solution = plan
        return plan

    def _build_warm_start(self, last_plan: AllocationPlan) -> dict:
        """
        将上一轮解转换为 {var_name: value} 字典。

        NOTE: 当前变量命名规则与 VariableBuilder 的名称不完全对齐
        （如 plan 中的 "U1" vs 模型中的 "U0"），热启动效果受限。
        此问题在后续迭代中修复。
        """
        if last_plan is None:
            return None
        warm_start = {}
        for ra in last_plan.recon_assignments:
            warm_start[f"x_{ra.pid}_{ra.sensor_used}_{ra.cell}"] = 1.0
        for sa in last_plan.strike_assignments:
            warm_start[f"z_{sa.pid}_{sa.target}"] = 1.0
            warm_start[f"y_{sa.pid}_{sa.munition}_{sa.target}"] = float(sa.qty)
            if sa.role == "lead":
                warm_start[f"rL_{sa.pid}_{sa.target}"] = 1.0
            elif sa.role == "wing":
                warm_start[f"rW_{sa.pid}_{sa.target}"] = 1.0
        return warm_start if warm_start else None
