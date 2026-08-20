from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from core.snapshot import SituationSnapshot
from allocation.solver_interface import SolverInterface


class VariableIndices:
    """维护所有索引范围，自动适配实际快照中的平台/目标/栅格数量。"""

    def __init__(self, snapshot: SituationSnapshot):
        self.uav_platforms = snapshot.get_uav_platforms()
        self.heli_platforms = snapshot.get_heli_platforms()
        self.active_targets = snapshot.get_active_targets()
        self.cells = list(snapshot.grids)

        self.N_U = len(self.uav_platforms)
        self.N_H = len(self.heli_platforms)
        self.N_G = len(self.active_targets)
        self.N_C = len(self.cells)

        if self.uav_platforms:
            all_sensors = set()
            for p in self.uav_platforms:
                all_sensors.update(p.sensors_mounted)
            self.sensor_names = sorted(all_sensors)
        else:
            self.sensor_names = ["EO", "SAR", "ESM"]
        self.N_S = len(self.sensor_names)

        self.weapon_names = ["HF", "RKT", "GUN"]
        self.N_W = len(self.weapon_names)

        # 平台索引 → pid 映射
        self.uav_pid = [p.pid for p in self.uav_platforms]
        self.heli_pid = [p.pid for p in self.heli_platforms]
        self.target_tid = [t.tid for t in self.active_targets]
        self.cell_cid = [c.cell_id for c in self.cells]


@dataclass
class DecisionVariableSet:
    """MILP 决策变量集合。变量按 (索引) 或 (索引, 索引, 索引) 组织为嵌套字典。"""

    # -- 侦察类 --
    ell: Dict[Tuple[int, int], object] = field(default_factory=dict)   # (u, s)
    x: Dict[Tuple[int, int, int], object] = field(default_factory=dict)  # (u, s, c)
    q: Dict[int, object] = field(default_factory=dict)                   # (c,)
    xi: Dict[int, object] = field(default_factory=dict)                  # (c,)

    # -- 打击类 --
    z: Dict[Tuple[int, int], object] = field(default_factory=dict)       # (h, g)
    y: Dict[Tuple[int, int, int], object] = field(default_factory=dict)  # (h, w, g)
    k: Dict[int, object] = field(default_factory=dict)                   # (g,)
    r_L: Dict[Tuple[int, int], object] = field(default_factory=dict)     # (h, g)
    r_W: Dict[Tuple[int, int], object] = field(default_factory=dict)     # (h, g)
    u: Dict[int, object] = field(default_factory=dict)                   # (g,)

    def get_all_vars(self) -> List[object]:
        """返回所有变量的 flat 列表。"""
        all_v = []
        all_v.extend(self.ell.values())
        all_v.extend(self.x.values())
        all_v.extend(self.q.values())
        all_v.extend(self.xi.values())
        all_v.extend(self.z.values())
        all_v.extend(self.y.values())
        all_v.extend(self.k.values())
        all_v.extend(self.r_L.values())
        all_v.extend(self.r_W.values())
        all_v.extend(self.u.values())
        return all_v


class VariableBuilder:
    """通过 SolverInterface 创建所有决策变量，并维护 name→handle 注册表。"""

    def __init__(self, idx: VariableIndices, model, solver: SolverInterface):
        self.idx = idx
        self.model = model
        self.solver = solver
        self.registry: Dict[str, Tuple[object, str]] = {}  # name → (handle, type)

    def create_all(self) -> DecisionVariableSet:
        vars_ = DecisionVariableSet()
        self._create_recon_vars(vars_)
        self._create_strike_vars(vars_)
        return vars_

    def _reg(self, name: str, var_handle, vtype: str):
        self.registry[name] = (var_handle, vtype)

    def _create_recon_vars(self, vars_: DecisionVariableSet):
        idx = self.idx

        # ℓ_{p,s}: UAV 挂载传感器
        for u in range(idx.N_U):
            for s in range(idx.N_S):
                name = f"ell_U{u}_{idx.sensor_names[s]}"
                v = self.solver.add_binary_var(self.model, name)
                vars_.ell[(u, s)] = v
                self._reg(name, v, "binary")

        # x_{p,s,c}: UAV 侦察分配
        for u in range(idx.N_U):
            for s in range(idx.N_S):
                for c in range(idx.N_C):
                    name = f"x_U{u}_{idx.sensor_names[s]}_{idx.cell_cid[c]}"
                    v = self.solver.add_binary_var(self.model, name)
                    vars_.x[(u, s, c)] = v
                    self._reg(name, v, "binary")

        # q_c: 覆盖质量（无上界，允许叠加传感器效能超 1.0）
        for c in range(idx.N_C):
            name = f"q_{idx.cell_cid[c]}"
            v = self.solver.add_continuous_var(self.model, name, lb=0.0)
            vars_.q[c] = v
            self._reg(name, v, "continuous")

        # ξ_c: 覆盖松弛
        for c in range(idx.N_C):
            name = f"xi_{idx.cell_cid[c]}"
            v = self.solver.add_continuous_var(self.model, name, lb=0.0, ub=10.0)
            vars_.xi[c] = v
            self._reg(name, v, "continuous")

    def _create_strike_vars(self, vars_: DecisionVariableSet):
        idx = self.idx
        if idx.N_G == 0 or idx.N_H == 0:
            return

        # z_{p,g}: 直升机交战
        for h in range(idx.N_H):
            for g in range(idx.N_G):
                name = f"z_H{h}_{idx.target_tid[g]}"
                v = self.solver.add_binary_var(self.model, name)
                vars_.z[(h, g)] = v
                self._reg(name, v, "binary")

        # y_{p,w,g}: 弹药数量
        for h in range(idx.N_H):
            for w in range(idx.N_W):
                for g in range(idx.N_G):
                    name = f"y_H{h}_{idx.weapon_names[w]}_{idx.target_tid[g]}"
                    max_qty = int(idx.active_targets[g].value * 10) + 5 if idx.N_G > 0 else 10
                    v = self.solver.add_integer_var(self.model, name, lb=0, ub=max_qty)
                    vars_.y[(h, w, g)] = v
                    self._reg(name, v, "integer")

        # k_g: 火力达标指示
        for g in range(idx.N_G):
            name = f"k_{idx.target_tid[g]}"
            v = self.solver.add_binary_var(self.model, name)
            vars_.k[g] = v
            self._reg(name, v, "binary")

        # r^L_{p,g}, r^W_{p,g}: 长机/僚机角色
        for h in range(idx.N_H):
            for g in range(idx.N_G):
                name = f"rL_H{h}_{idx.target_tid[g]}"
                v = self.solver.add_binary_var(self.model, name)
                vars_.r_L[(h, g)] = v
                self._reg(name, v, "binary")

                name = f"rW_H{h}_{idx.target_tid[g]}"
                v = self.solver.add_binary_var(self.model, name)
                vars_.r_W[(h, g)] = v
                self._reg(name, v, "binary")

        # u_g: 毁伤期望
        for g in range(idx.N_G):
            name = f"u_{idx.target_tid[g]}"
            v = self.solver.add_continuous_var(self.model, name, lb=0.0, ub=1.0)
            vars_.u[g] = v
            self._reg(name, v, "continuous")
