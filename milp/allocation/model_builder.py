"""
MILP 模型组装器 —— 按顺序创建变量、约束、目标函数，并提供解提取方法。

职责:
  1. 按标准化流程构建完整 MILP: 变量 → 约束 → 目标
  2. 从求解结果中提取 ReconAssignment 与 StrikeAssignment
  3. 为 DisplayManager 提供模型构建耗时信息

组装流程:
  VariableBuilder(变量) → ConstraintBuilder(约束) → ObjectiveBuilder(目标)
  三者通过 SolverInterface 抽象层操作模型，与具体求解器解耦。

对外接口:
  - ModelBuilder  — 模型组装类

参考:
  设计方案 §4.3-4.5
"""

import time
from core.snapshot import SituationSnapshot
from core.allocation import AllocationPlan, ReconAssignment, StrikeAssignment
from config.settings import GlobalSettings
from allocation.solver_interface import SolverInterface
from allocation.parameters import ParameterBuilder
from allocation.decision_variables import VariableIndices, DecisionVariableSet, VariableBuilder
from allocation.objective import ObjectiveBuilder
from allocation.constraints import ConstraintBuilder


class ModelBuilder:
    """
    整合变量创建、约束添加、目标函数设置，并提供解提取方法。

    属性:
        snap: 态势快照
        s: 全局设置
        solver: 求解器接口实例
        idx: 变量索引
        params: MILP 参数构建器
        display: 终端显示管理器（可空）
        _var_builder: 变量构建器实例（含变量注册表）

    参考:
        设计方案 §4.3-4.5
    """

    def __init__(self, snapshot: SituationSnapshot,
                 settings: GlobalSettings,
                 solver: SolverInterface,
                 display=None):
        """
        Args:
            snapshot: 态势快照
            settings: 全局配置
            solver: 求解器接口实例
            display: DisplayManager 实例（可选，用于计时输出）
        """
        self.snap = snapshot
        self.s = settings
        self.solver = solver
        self.display = display
        self.idx = VariableIndices(snapshot)
        self.params = ParameterBuilder(snapshot, settings)
        self._var_builder: VariableBuilder = None

    def build(self) -> tuple:
        """
        构建完整 MILP 模型。

        步骤:
            1. 创建空模型
            2. 创建全部决策变量
            3. 添加全部约束
            4. 设置目标函数（最大化）

        Returns:
            (model, vars_) 元组，model 为求解器原生模型对象，
            vars_ 为 DecisionVariableSet
        """
        t0 = time.time()

        # Step 1: 创建空模型
        model = self.solver.create_model("MILP_Task_Allocation")

        # Step 2: 创建决策变量
        var_builder = VariableBuilder(self.idx, model, self.solver)
        vars_ = var_builder.create_all()
        self._var_builder = var_builder

        # Step 3: 添加约束
        cb = ConstraintBuilder(self.params, self.idx, self.s)
        cb.build_all(vars_, model, self.solver)

        # Step 4: 设置目标函数
        ob = ObjectiveBuilder(self.params, self.idx, self.s)
        obj_expr = ob.build(vars_)
        self.solver.set_objective(model, obj_expr, maximize=True)

        elapsed_ms = (time.time() - t0) * 1000

        # 通知显示层
        if self.display is not None:
            self.display.show_model_built(self.idx, elapsed_ms)

        return model, vars_

    def extract_assignments(self, plan: AllocationPlan, vars_: DecisionVariableSet,
                            model) -> None:
        """
        从求解后的变量值中提取侦察/打击分配，填充 AllocationPlan。

        提取规则:
          - x_{u,s,c} > 0.5 → ReconAssignment
          - z_{h,g} > 0.5 → 检查 y 变量:
              - 有弹药发射 → StrikeAssignment（具体弹药条目）
              - 无弹药发射 → StrikeAssignment（qty=0, role+="_support"）表示提供平台数

        Args:
            plan: 待填充的 AllocationPlan
            vars_: 求解后的 DecisionVariableSet
            model: 求解器模型对象（保留以兼容其他求解器）
        """
        idx = self.idx

        # -- 提取传感器挂载状态: UAV ℓ_{u,s} --
        for u in range(idx.N_U):
            pid = idx.uav_pid[u]
            sensors = []
            for s in range(idx.N_S):
                val = vars_.ell[(u, s)].x
                if val is not None and val > 0.5:
                    sensors.append(idx.sensor_names[s])
            plan.mounted_sensors[pid] = sensors

        # HELI 传感器直接从输入获取（不做 MILP 优化）
        for h in range(idx.N_H):
            pid = idx.heli_pid[h]
            plan.mounted_sensors[pid] = list(idx.heli_platforms[h].sensors_mounted)

        # -- 侦察分配: 遍历 x_{p,s,c} --
        for (u, s, c), var in vars_.x.items():
            val = var.x if (hasattr(var, 'x') and var.x is not None) else 0.0
            if val > 0.5:
                pid = idx.uav_pid[u]
                sensor = idx.sensor_names[s]
                cell = idx.cell_cid[c]
                plan.recon_assignments.append(ReconAssignment(
                    pid=pid,
                    sensors_mounted=plan.mounted_sensors.get(pid, []),
                    sensor_used=sensor,
                    cell=cell,
                    role="area_scan"
                ))

        # -- 打击分配: 遍历 z_{p,g} --
        for (h, g), var in vars_.z.items():
            val = var.x if (hasattr(var, 'x') and var.x is not None) else 0.0
            if val > 0.5:
                pid = idx.heli_pid[h]
                tid = idx.target_tid[g]
                # 判定角色: lead / wing / striker
                role = "striker"
                rL = vars_.r_L.get((h, g))
                rW = vars_.r_W.get((h, g))
                if rL is not None and (rL.x if hasattr(rL, 'x') else 0) > 0.5:
                    role = "lead"
                elif rW is not None and (rW.x if hasattr(rW, 'x') else 0) > 0.5:
                    role = "wing"

                has_weapon = False
                for w in range(idx.N_W):
                    yvar = vars_.y.get((h, w, g))
                    if yvar is not None:
                        qty_val = int(round(yvar.x)) if hasattr(yvar, 'x') else 0
                        if qty_val > 0:
                            has_weapon = True
                            plan.strike_assignments.append(StrikeAssignment(
                                pid=pid,
                                target=tid,
                                munition=idx.weapon_names[w],
                                qty=qty_val,
                                role=role
                            ))
                # 无武器发射但仍被指派 → 提供平台数量需求（支援角色）
                if not has_weapon:
                    plan.strike_assignments.append(StrikeAssignment(
                        pid=pid,
                        target=tid,
                        munition="",
                        qty=0,
                        role=role + "_support"
                    ))

    def get_var_value(self, var_name: str, model) -> float:
        """
        根据变量名获取求解后的值（用于调试/内省）。

        Args:
            var_name: 变量名（如 "x_U0_ESM_c1"）
            model: 求解器模型对象（保留接口兼容性）

        Returns:
            变量的求解值，若不存在返回 0.0
        """
        if self._var_builder is None:
            return 0.0
        entry = self._var_builder.registry.get(var_name)
        if entry is None:
            return 0.0
        var_handle, _ = entry
        return var_handle.x if hasattr(var_handle, 'x') else 0.0
