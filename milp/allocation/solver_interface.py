from abc import ABC, abstractmethod
from config.settings import SolverType, GlobalSettings
from core.allocation import AllocationPlan


class SolverInterface(ABC):
    """所有求解器必须实现此接口。上层代码只依赖此接口，与具体求解器解耦。"""

    @abstractmethod
    def create_model(self, name: str):
        """创建空模型对象，返回模型 handle。"""
        ...

    @abstractmethod
    def add_binary_var(self, model, name: str):
        """添加二值变量，返回变量 handle。"""
        ...

    @abstractmethod
    def add_integer_var(self, model, name: str, lb: int = 0, ub: int = 1000):
        """添加整数变量，返回变量 handle。"""
        ...

    @abstractmethod
    def add_continuous_var(self, model, name: str, lb: float = 0.0, ub: float = None):
        """添加连续变量，返回变量 handle。ub=None 表示无上界。"""
        ...

    @abstractmethod
    def add_constraint(self, model, expr, name: str = ""):
        """向模型添加一条约束。"""
        ...

    @abstractmethod
    def set_objective(self, model, expr, maximize: bool = True):
        """设置目标函数方向。"""
        ...

    @abstractmethod
    def solve(self, model, warm_start: dict = None,
              time_limit_s: float = 3.0,
              mip_gap: float = 1e-3) -> AllocationPlan:
        """求解模型，返回 AllocationPlan。"""
        ...

    @property
    @abstractmethod
    def solver_name(self) -> str:
        """返回求解器名称（写入 AllocationPlan.solver_used）。"""
        ...


class SolverFactory:
    """根据 SolverType 枚举实例化对应求解器。"""

    @classmethod
    def create(cls, solver_type: SolverType,
               settings: GlobalSettings) -> SolverInterface:
        if solver_type not in settings.enabled_solvers:
            raise ValueError(
                f"求解器 {solver_type} 不在 enabled_solvers 白名单中，"
                f"请先将其加入 GlobalSettings.enabled_solvers。"
            )

        if solver_type == SolverType.GUROBI:
            from allocation.solvers.gurobi_solver import GurobiSolver
            return GurobiSolver(settings)

        elif solver_type == SolverType.CBC:
            from allocation.solvers.cbc_solver import CBCSolver
            return CBCSolver(settings)

        elif solver_type == SolverType.ORTOOLS:
            from allocation.solvers.ortools_solver import OrtoolsSolver
            return OrtoolsSolver(settings)

        elif solver_type == SolverType.HIGHS:
            from allocation.solvers.highs_solver import HiGHSSolver
            return HiGHSSolver(settings)

        else:
            raise ValueError(f"未知求解器类型：{solver_type}")
