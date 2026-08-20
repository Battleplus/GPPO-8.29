"""
CBC 求解器实现 —— 基于 python-mip 库的 COIN-OR CBC 开源 MILP 求解器。

职责:
  1. 实现 SolverInterface 全部抽象方法
  2. 通过 python-mip 库调用 CBC 后端
  3. 支持热启动（部分）、时间盒截断、MIP gap 控制
  4. 根据 GlobalSettings.verbose 控制求解器日志输出

特点:
  - 完全开源，无商业授权限制
  - 接口与 Gurobi 高度相似，便于后续平滑切换
  - 对于本问题规模（~150 变量, ~120 约束），求解时间 < 100 ms

对外接口:
  - CBCSolver  — SolverInterface 的 CBC 实现

参考:
  设计方案 §5.3; 实施方案 §2.11
"""

import time
import mip
from allocation.solver_interface import SolverInterface
from config.settings import GlobalSettings
from core.allocation import AllocationPlan


class CBCSolver(SolverInterface):
    """
    CBC 求解器实现（基于 python-mip 库）。

    CBC 是 COIN-OR 开源 MILP 求解器，与 Gurobi 接口高度相似，
    无商业授权限制，适合作为默认求解器。

    属性:
        settings: 全局配置（用于读取 verbose 等参数）
    """

    def __init__(self, settings: GlobalSettings):
        """存储配置引用，供 solve() 阶段使用。"""
        self.settings = settings

    @property
    def solver_name(self) -> str:
        """返回求解器标识名，写入 AllocationPlan.solver_used。"""
        return "CBC (python-mip)"

    def create_model(self, name: str) -> mip.Model:
        """
        创建 CBC 模型对象。

        Args:
            name: 模型名称

        Returns:
            mip.Model 实例（最大化方向，CBC 后端）
        """
        return mip.Model(name=name, sense=mip.MAXIMIZE, solver_name=mip.CBC)

    def add_binary_var(self, model: mip.Model, name: str) -> mip.Var:
        """添加二值变量 x ∈ {0, 1}。"""
        return model.add_var(name=name, var_type=mip.BINARY)

    def add_integer_var(self, model: mip.Model, name: str,
                        lb: int = 0, ub: int = 1000) -> mip.Var:
        """添加整数变量 y ∈ [lb, ub] ∩ Z。"""
        return model.add_var(name=name, var_type=mip.INTEGER, lb=lb, ub=ub)

    def add_continuous_var(self, model: mip.Model, name: str,
                           lb: float = 0.0, ub: float = None) -> mip.Var:
        """添加连续变量 z ∈ [lb, ub] ⊂ R。ub=None 表示无上界。"""
        if ub is None:
            return model.add_var(name=name, var_type=mip.CONTINUOUS, lb=lb)
        return model.add_var(name=name, var_type=mip.CONTINUOUS, lb=lb, ub=ub)

    def add_constraint(self, model: mip.Model, expr, name: str = ""):
        """向模型添加一条线性约束。若提供 name 则使用命名约束。"""
        if name:
            model.add_constr(expr, name=name)
        else:
            model += expr

    def set_objective(self, model: mip.Model, expr, maximize: bool = True):
        """设置目标函数方向。默认最大化。"""
        if maximize:
            model.objective = mip.maximize(expr)
        else:
            model.objective = mip.minimize(expr)

    def solve(self, model: mip.Model,
              warm_start: dict = None,
              time_limit_s: float = 3.0,
              mip_gap: float = 1e-3) -> AllocationPlan:
        """
        调用 CBC 求解模型。

        参数:
            model: mip.Model 实例
            warm_start: {var_name: value} 热启动字典（CBC 部分支持）
            time_limit_s: 求解时间上限 (s)，超时返回当前最佳可行解
            mip_gap: 相对 MIP gap 阈值

        返回:
            AllocationPlan（含求解状态、目标值、gap 等元数据）

        NOTE: 本问题规模（~150 变量, ~120 约束）在 100ms 内可求得最优解，
              远低于默认 3s 时限。
        """
        # 求解器参数
        model.max_gap = mip_gap

        # 根据 verbose 控制 CBC 自身日志输出
        # verbose >= 2: 显示 CBC 分支定界日志
        # verbose < 2:  静默
        model.verbose = 1 if self.settings.verbose >= 2 else 0

        # 热启动注入（CBC 通过 python-mip 的 var.start 部分支持）
        if warm_start:
            try:
                applied = 0
                for var in model.vars:
                    if var.name in warm_start:
                        var.start = warm_start[var.name]
                        applied += 1
            except Exception:
                pass  # 热启动失败不影响求解

        t0 = time.time()
        status = model.optimize(max_seconds=time_limit_s)
        solve_time_ms = (time.time() - t0) * 1000

        return self._extract_plan(model, status, solve_time_ms)

    def _extract_plan(self, model: mip.Model,
                      status: mip.OptimizationStatus,
                      solve_time_ms: float) -> AllocationPlan:
        """
        从 CBC 求解结果中提取 AllocationPlan 元数据。

        将 mip.OptimizationStatus 枚举映射为人类可读状态字符串：
          OPTIMAL → "OPTIMAL"（全局最优）
          FEASIBLE → "FEASIBLE"（可行解，可能非最优）
          INFEASIBLE → "INFEASIBLE"（无可行解）
          NO_SOLUTION_FOUND → "TIME_LIMIT"（超时未找到解）

        Args:
            model: 已求解的 mip.Model
            status: CBC 返回的优化状态
            solve_time_ms: 求解耗时 (ms)

        Returns:
            填充了元数据的 AllocationPlan
        """
        status_map = {
            mip.OptimizationStatus.OPTIMAL: "OPTIMAL",
            mip.OptimizationStatus.FEASIBLE: "FEASIBLE",
            mip.OptimizationStatus.INFEASIBLE: "INFEASIBLE",
            mip.OptimizationStatus.NO_SOLUTION_FOUND: "TIME_LIMIT",
        }
        status_str = status_map.get(status, "UNKNOWN")

        has_solution = status in (mip.OptimizationStatus.OPTIMAL,
                                  mip.OptimizationStatus.FEASIBLE)

        plan = AllocationPlan(
            cycle_id=0,
            solve_time_ms=solve_time_ms,
            solver_used=self.solver_name,
            objective=model.objective_value if has_solution else 0.0,
            mip_gap=model.gap if has_solution else 1.0,
            status=status_str,
        )
        return plan
