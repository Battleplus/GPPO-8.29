import mip
import numpy as np
from allocation.parameters import ParameterBuilder
from allocation.decision_variables import VariableIndices, DecisionVariableSet
from allocation.solver_interface import SolverInterface
from config.settings import GlobalSettings


def _weather_gain(sensor_name: str, weather_w: float) -> float:
    """天气-传感器增益 β_s(w_c)，与 parameters.py 中保持一致。"""
    if sensor_name == "EO":
        if weather_w < 0.30:
            return 1.0
        elif weather_w < 0.45:
            return 0.7
        elif weather_w < 0.65:
            return 0.4
        elif weather_w < 0.80:
            return 0.1
        else:
            return 0.0
    elif sensor_name == "SAR":
        if weather_w < 0.30:
            return 0.6
        elif weather_w < 0.45:
            return 0.7
        elif weather_w < 0.65:
            return 0.9
        elif weather_w < 0.80:
            return 1.0
        else:
            return 1.0
    else:
        return 1.0


class ConstraintBuilder:
    """构建所有 MILP 约束（§4.5 约束 1-13）。"""

    def __init__(self, params: ParameterBuilder, idx: VariableIndices,
                 settings: GlobalSettings):
        self.params = params
        self.idx = idx
        self.settings = settings

    def build_all(self, vars_: DecisionVariableSet, model,
                  solver: SolverInterface):
        self._c1_uav_sensor_limit(vars_, model, solver)
        self._c1b_sensor_availability(vars_, model, solver)
        self._c2_uav_mission_capacity(vars_, model, solver)
        self._c2b_sar_one_cell(vars_, model, solver)
        self._c2b_eo_one_cell(vars_, model, solver)
        self._c2d_eo_requires_sar(vars_, model, solver)
        self._c2c_esm_wide_area(vars_, model, solver)
        self._c_esm_count(vars_, model, solver)
        self._c3_weather_sensor(vars_, model, solver)
        self._c4_range_pre_fix(vars_, model, solver)
        self._c5_coverage(vars_, model, solver)
        self._c5b_coverage_quality_link(vars_, model, solver)
        self._c_eo_sar_per_subcell(vars_, model, solver)
        self._c6_esm_patrol(vars_, model, solver)
        self._c7_los(vars_, model, solver)
        self._c8_detect_strike_gate(vars_, model, solver)
        self._c8b_heli_mission_capacity(vars_, model, solver)
        self._c8c_strike_needs_sar(vars_, model, solver)
        self._c8d_strike_needs_eo(vars_, model, solver)
        self._c9_firepower(vars_, model, solver)
        self._c10_lead_wing(vars_, model, solver)
        self._c11_ammo(vars_, model, solver)
        self._c12_survivability(vars_, model, solver)
        self._c13_destruction_link(vars_, model, solver)
        self._c14_weapon_z_link(vars_, model, solver)
        self._c15_heli_transit(vars_, model, solver)
        self._c4b_scan_time_budget(vars_, model, solver)
        self._c4c_distance_budget(vars_, model, solver)
        self._c12b_uav_risk_budget(vars_, model, solver)

    # ---- 约束 (1): UAV 挂载 1~2 个传感器 ----
    def _c1_uav_sensor_limit(self, vars_, model, solver):
        idx = self.idx
        for u in range(idx.N_U):
            s_sum = mip.xsum(vars_.ell[(u, s)] for s in range(idx.N_S))
            solver.add_constraint(model, s_sum >= 1,
                                  name=f"c1_min_one_sensor_U{u}")
            solver.add_constraint(model, s_sum <= 2,
                                  name=f"c1_max_two_sensors_U{u}")

    # ---- 约束 (1b): UAV 只能使用自己携带的传感器 ----
    def _c1b_sensor_availability(self, vars_, model, solver):
        idx = self.idx
        for u in range(idx.N_U):
            available = set(idx.uav_platforms[u].sensors_mounted)
            for s in range(idx.N_S):
                if idx.sensor_names[s] not in available:
                    solver.add_constraint(
                        model,
                        vars_.ell[(u, s)] == 0,
                        name=f"c1b_sensor_unavail_U{u}_S{s}"
                    )

    # ---- 约束 (2): UAV 单架次任务容量 (x ≤ ℓ) ----
    def _c2_uav_mission_capacity(self, vars_, model, solver):
        idx = self.idx
        for u in range(idx.N_U):
            for s in range(idx.N_S):
                for c in range(idx.N_C):
                    solver.add_constraint(
                        model,
                        vars_.x[(u, s, c)] <= vars_.ell[(u, s)],
                        name=f"c2a_x_le_ell_U{u}_S{s}_C{c}"
                    )

    # ---- 约束 (2b): SAR 每架次最多 1 个栅格 ----
    def _c2b_sar_one_cell(self, vars_, model, solver):
        idx = self.idx
        sar_idx = None
        for si, s_name in enumerate(idx.sensor_names):
            if s_name == "SAR":
                sar_idx = si
                break
        if sar_idx is None:
            return
        for u in range(idx.N_U):
            solver.add_constraint(
                model,
                mip.xsum(vars_.x[(u, sar_idx, c)] for c in range(idx.N_C)) <= 1,
                name=f"c2b_sar_one_cell_U{u}"
            )

    # ---- 约束 (2b'): EO 每架次最多 1 个栅格 ----
    def _c2b_eo_one_cell(self, vars_, model, solver):
        idx = self.idx
        eo_idx = None
        for si, s_name in enumerate(idx.sensor_names):
            if s_name == "EO":
                eo_idx = si
                break
        if eo_idx is None:
            return
        for u in range(idx.N_U):
            solver.add_constraint(
                model,
                mip.xsum(vars_.x[(u, eo_idx, c)] for c in range(idx.N_C)) <= 1,
                name=f"c2b_eo_one_cell_U{u}"
            )

    # ---- 约束 (2d): EO 确认依赖 SAR 先搜索 ----
    def _c2d_eo_requires_sar(self, vars_, model, solver):
        """x[u,EO,c] <= sum_{u'} x[u',SAR,c]"""
        idx = self.idx
        eo_idx = None
        sar_idx = None
        for si, s_name in enumerate(idx.sensor_names):
            if s_name == "EO":
                eo_idx = si
            elif s_name == "SAR":
                sar_idx = si
        if eo_idx is None or sar_idx is None or idx.N_U == 0:
            return
        for u in range(idx.N_U):
            for c in range(idx.N_C):
                solver.add_constraint(
                    model,
                    vars_.x[(u, eo_idx, c)] <= mip.xsum(
                        vars_.x[(u2, sar_idx, c)] for u2 in range(idx.N_U)
                    ),
                    name=f"c2d_eo_needs_sar_U{u}_C{c}"
                )

    # ---- 约束 (2c): ESM UAV 覆盖全部栅格 ----
    def _c2c_esm_wide_area(self, vars_, model, solver):
        idx = self.idx
        esm_idx = None
        for si, s_name in enumerate(idx.sensor_names):
            if s_name == "ESM":
                esm_idx = si
                break
        if esm_idx is None or idx.N_U == 0:
            return
        for u in range(idx.N_U):
            for c in range(idx.N_C):
                solver.add_constraint(
                    model,
                    vars_.x[(u, esm_idx, c)] == vars_.ell[(u, esm_idx)],
                    name=f"c2c_esm_wide_U{u}_C{c}"
                )

    # ---- 约束: 恰好 1 架 UAV 挂载 ESM ----
    def _c_esm_count(self, vars_, model, solver):
        idx = self.idx
        esm_idx = None
        for si, s_name in enumerate(idx.sensor_names):
            if s_name == "ESM":
                esm_idx = si
                break
        if esm_idx is None or idx.N_U == 0:
            return
        solver.add_constraint(
            model,
            mip.xsum(vars_.ell[(u, esm_idx)] for u in range(idx.N_U)) == 1,
            name="c_esm_count_eq1"
        )

    # ---- 约束: 每个子区域至少 1 架 EO/SAR UAV ----
    def _c_eo_sar_per_subcell(self, vars_, model, solver):
        idx = self.idx
        p = self.params
        eo_idx = None
        sar_idx = None
        for si, s_name in enumerate(idx.sensor_names):
            if s_name == "EO":
                eo_idx = si
            elif s_name == "SAR":
                sar_idx = si
        eo_sar_indices = [si for si in (eo_idx, sar_idx) if si is not None]
        if not eo_sar_indices:
            return
        for ci in p.sub_cell_indices:
            solver.add_constraint(
                model,
                mip.xsum(vars_.x[(u, si, ci)]
                         for u in range(idx.N_U)
                         for si in eo_sar_indices) >= 1,
                name=f"c_eo_sar_per_subcell_C{ci}"
            )

    # ---- 约束 (3): 天气-传感器硬约束 (EO 在高湿雾下禁用) ----
    def _c3_weather_sensor(self, vars_, model, solver):
        idx = self.idx
        p = self.params
        for u in range(idx.N_U):
            for ci, c in enumerate(range(idx.N_C)):
                w = p.cell_weather[ci]
                if w >= 0.80:
                    # 查找 EO 传感器的索引
                    for si, s_name in enumerate(idx.sensor_names):
                        if s_name == "EO":
                            solver.add_constraint(
                                model,
                                vars_.x[(u, si, ci)] == 0,
                                name=f"c3_no_EO_U{u}_C{ci}"
                            )

    # ---- 约束 (4): 传感器扫描时间可行性 —— 扫描超时的 (传感器, 栅格) 组合强制为 0 ----
    def _c4_range_pre_fix(self, vars_, model, solver):
        idx = self.idx
        p = self.params
        for u in range(idx.N_U):
            for si in range(idx.N_S):
                for ci in range(idx.N_C):
                    if p.scan_feasible[si, ci] < 0.5:
                        solver.add_constraint(
                            model,
                            vars_.x[(u, si, ci)] == 0,
                            name=f"c4_scan_infeasible_U{u}_S{si}_C{ci}"
                        )

    # ---- 约束 (5): 覆盖要求 ----
    def _c5_coverage(self, vars_, model, solver):
        idx = self.idx
        p = self.params
        for ci in range(idx.N_C):
            solver.add_constraint(
                model,
                mip.xsum(p.beta_sc[si, ci] * vars_.x[(u, si, ci)]
                         for u in range(idx.N_U)
                         for si in range(idx.N_S)) + vars_.xi[ci] >= 1.0,
                name=f"c5_coverage_C{ci}"
            )

    # ---- 约束 (5b): 覆盖质量上界 q_c ≤ Σ_{u,s} E_det_{u,s,c} * x_{u,s,c} ----
    def _c5b_coverage_quality_link(self, vars_, model, solver):
        idx = self.idx
        p = self.params
        for ci in range(idx.N_C):
            solver.add_constraint(
                model,
                vars_.q[ci] <= mip.xsum(
                    p.E_det[u, si, ci] * vars_.x[(u, si, ci)]
                    for u in range(idx.N_U)
                    for si in range(idx.N_S)
                ),
                name=f"c5b_q_le_Edet_C{ci}"
            )

    # ---- 约束 (6): ESM 巡逻区覆盖 ----
    def _c6_esm_patrol(self, vars_, model, solver):
        idx = self.idx
        # 寻找 ESM 传感器索引和 c0 栅格索引
        esm_idx = None
        for si, s_name in enumerate(idx.sensor_names):
            if s_name == "ESM":
                esm_idx = si
                break
        c0_idx = None
        for ci, cid in enumerate(idx.cell_cid):
            if cid == "c0":
                c0_idx = ci
                break
        if esm_idx is not None and c0_idx is not None and idx.N_U > 0:
            solver.add_constraint(
                model,
                mip.xsum(vars_.x[(u, esm_idx, c0_idx)]
                         for u in range(idx.N_U)) >= 1,
                name="c6_esm_patrol_c0"
            )

    # ---- 约束 (7): 通视约束 ----
    def _c7_los(self, vars_, model, solver):
        idx = self.idx
        p = self.params
        for h in range(idx.N_H):
            for g in range(idx.N_G):
                if p.V_hg[h, g] < 0.5:
                    solver.add_constraint(
                        model,
                        vars_.z[(h, g)] == 0,
                        name=f"c7_los_H{h}_G{g}"
                    )
                else:
                    solver.add_constraint(
                        model,
                        vars_.z[(h, g)] <= p.V_hg[h, g],
                        name=f"c7_los_le_H{h}_G{g}"
                    )

    # ---- 约束 (8): 先侦后打 + 存活门控 ----
    def _c8_detect_strike_gate(self, vars_, model, solver):
        # 已通过只包含 alive && confirmed 的目标来隐式保证
        # 此处显式添加门控约束以确保鲁棒性
        idx = self.idx
        p = self.params
        for h in range(idx.N_H):
            for g in range(idx.N_G):
                solver.add_constraint(
                    model,
                    vars_.z[(h, g)] <= p.target_confirmed[g],
                    name=f"c8a_conf_H{h}_G{g}"
                )
                solver.add_constraint(
                    model,
                    vars_.z[(h, g)] <= p.target_alive[g],
                    name=f"c8b_alive_H{h}_G{g}"
                )

    # ---- 约束 (8b): 直升机单架次交战容量 Σ_g z_{h,g} ≤ 1 ----
    def _c8b_heli_mission_capacity(self, vars_, model, solver):
        idx = self.idx
        if idx.N_H == 0 or idx.N_G == 0:
            return
        for h in range(idx.N_H):
            solver.add_constraint(
                model,
                mip.xsum(vars_.z[(h, g)] for g in range(idx.N_G)) <= 1,
                name=f"c8b_heli_capacity_H{h}"
            )

    # ---- 约束 (9): 火力需求 ----
    def _c9_firepower(self, vars_, model, solver):
        idx = self.idx
        p = self.params
        if idx.N_H == 0 or idx.N_G == 0:
            return
        for g in range(idx.N_G):
            req = p.req_plat[g]
            if req > 0:
                solver.add_constraint(
                    model,
                    mip.xsum(vars_.z[(h, g)] for h in range(idx.N_H)) >= req * vars_.k[g],
                    name=f"c9_firepower_G{g}"
                )
            # 武器数量需求
            for w in range(idx.N_W):
                req_w = p.req_weapon[g, w]
                if req_w > 0:
                    solver.add_constraint(
                        model,
                        mip.xsum(vars_.y[(h, w, g)] for h in range(idx.N_H)) >= req_w * vars_.k[g],
                        name=f"c9_weapon_G{g}_W{w}"
                    )

    # ---- 约束 (10): 长机/僚机协同 ----
    def _c10_lead_wing(self, vars_, model, solver):
        idx = self.idx
        p = self.params
        if idx.N_H == 0 or idx.N_G == 0:
            return
        for h in range(idx.N_H):
            for g in range(idx.N_G):
                solver.add_constraint(
                    model,
                    vars_.r_L[(h, g)] + vars_.r_W[(h, g)] <= vars_.z[(h, g)],
                    name=f"c10_role_le_z_H{h}_G{g}"
                )
        for g in range(idx.N_G):
            tgt_type = p.target_types[g] if g < len(p.target_types) else ""
            if tgt_type in ("RADAR", "CP"):
                solver.add_constraint(
                    model,
                    mip.xsum(vars_.r_L[(h, g)] for h in range(idx.N_H)) == vars_.k[g],
                    name=f"c10_lead_eq_k_G{g}"
                )
            else:
                solver.add_constraint(
                    model,
                    mip.xsum(vars_.r_L[(h, g)] for h in range(idx.N_H)) <= vars_.k[g],
                    name=f"c10_lead_le_k_G{g}"
                )
            solver.add_constraint(
                model,
                mip.xsum(vars_.r_W[(h, g)] for h in range(idx.N_H)) <= 1,
                name=f"c10_wing_le1_G{g}"
            )

    # ---- 约束 (11): 弹药余量 ----
    def _c11_ammo(self, vars_, model, solver):
        idx = self.idx
        p = self.params
        if idx.N_H == 0:
            return
        for h in range(idx.N_H):
            for w in range(idx.N_W):
                available = p.ammo_available[h, w]
                solver.add_constraint(
                    model,
                    mip.xsum(vars_.y[(h, w, g)] for g in range(idx.N_G)) <= available,
                    name=f"c11_ammo_H{h}_W{w}"
                )

    # ---- 约束 (12): 生存性 ----
    def _c12_survivability(self, vars_, model, solver):
        idx = self.idx
        p = self.params
        s = self.settings
        if idx.N_H == 0 or idx.N_G == 0:
            return
        for h in range(idx.N_H):
            solver.add_constraint(
                model,
                mip.xsum(p.effective_threat[g] * vars_.z[(h, g)]
                         for g in range(idx.N_G)) <= s.theta_max,
                name=f"c12_survival_H{h}"
            )

    # ---- 约束 (13): 毁伤链接 u_g ≤ k_g ----
    def _c13_destruction_link(self, vars_, model, solver):
        idx = self.idx
        if idx.N_G == 0 or idx.N_H == 0:
            return
        for g in range(idx.N_G):
            solver.add_constraint(
                model,
                vars_.u[g] <= vars_.k[g],
                name=f"c13_u_le_k_G{g}"
            )
            # 也绑定到实际平台分配: u_g ≤ (1/req_plat) * Σ_h z_{h,g}
            p = self.params
            req = max(p.req_plat[g], 1)
            if idx.N_H > 0:
                solver.add_constraint(
                    model,
                    vars_.u[g] <= (1.0 / req) * mip.xsum(
                        vars_.z[(h, g)] for h in range(idx.N_H)),
                    name=f"c13_u_le_z_G{g}"
                )

    # ---- 约束 (14): 武器使用必须伴随交战 ----
    def _c14_weapon_z_link(self, vars_, model, solver):
        idx = self.idx
        if idx.N_H == 0 or idx.N_G == 0:
            return
        big_M = 100
        for h in range(idx.N_H):
            for w in range(idx.N_W):
                for g in range(idx.N_G):
                    solver.add_constraint(
                        model,
                        vars_.y[(h, w, g)] <= big_M * vars_.z[(h, g)],
                        name=f"c14_y_le_Mz_H{h}_W{w}_G{g}"
                    )

    # ---- 约束 (15): 直升机转场可行性 ----
    def _c15_heli_transit(self, vars_, model, solver):
        idx = self.idx
        p = self.params
        s = self.settings
        if idx.N_H == 0 or idx.N_G == 0:
            return
        cruise_speed = s.uav_loiter_speed_kmh
        max_time_min = s.mission_total_time_max_min
        for h in range(idx.N_H):
            for g in range(idx.N_G):
                transit_time_min = (p.D_hg[h, g] / cruise_speed) * 60.0
                if transit_time_min > max_time_min:
                    solver.add_constraint(
                        model,
                        vars_.z[(h, g)] == 0,
                        name=f"c15_transit_H{h}_G{g}"
                    )

    # ---- 约束 (4b): UAV 扫描时间预算（仅 EO/SAR） ----
    def _c4b_scan_time_budget(self, vars_, model, solver):
        """Sigma_{s,c} T_scan_uc[u,c] * x[u,s,c] <= T_available, 仅 EO/SAR"""
        idx = self.idx
        p = self.params
        eo_sar = self._eo_sar_indices()
        if not eo_sar:
            return
        T_avail = self.settings.uav_available_time_min
        for u in range(idx.N_U):
            solver.add_constraint(
                model,
                mip.xsum(
                    p.T_scan_uc[u, c] * vars_.x[(u, si, c)]
                    for si in eo_sar
                    for c in range(idx.N_C)
                ) <= T_avail,
                name=f"c4b_scan_time_U{u}"
            )

    # ---- 约束 (4c): UAV 航程预算（仅 EO/SAR） ----
    def _c4c_distance_budget(self, vars_, model, solver):
        """Sigma_{s,c} D_uc[u,c] * x[u,s,c] <= D_max, 仅 EO/SAR"""
        idx = self.idx
        p = self.params
        eo_sar = self._eo_sar_indices()
        if not eo_sar:
            return
        D_max = self.settings.uav_max_range_km
        for u in range(idx.N_U):
            solver.add_constraint(
                model,
                mip.xsum(
                    p.D_uc[u, c] * vars_.x[(u, si, c)]
                    for si in eo_sar
                    for c in range(idx.N_C)
                ) <= D_max,
                name=f"c4c_distance_U{u}"
            )

    # ---- 约束 (12b): UAV 风险预算（仅 EO/SAR） ----
    def _c12b_uav_risk_budget(self, vars_, model, solver):
        """Sigma_{s,c} threat_of_cell[c] * x[u,s,c] <= risk_budget, 仅 EO/SAR"""
        idx = self.idx
        p = self.params
        eo_sar = self._eo_sar_indices()
        if not eo_sar:
            return
        cell_threat = np.zeros(idx.N_C)
        for ci in range(idx.N_C):
            ts = [p.effective_threat[gi] for gi in range(idx.N_G)
                  if p.target_cell_idx[gi] == ci]
            cell_threat[ci] = float(np.mean(ts)) if ts else 0.0

        budget = self.settings.risk_budget_uav
        for u in range(idx.N_U):
            solver.add_constraint(
                model,
                mip.xsum(
                    cell_threat[c] * vars_.x[(u, si, c)]
                    for si in eo_sar
                    for c in range(idx.N_C)
                ) <= budget,
                name=f"c12b_uav_risk_U{u}"
            )

    def _eo_sar_indices(self):
        """返回 EO 和 SAR 传感器的索引列表。"""
        eo_sar = []
        for si, s_name in enumerate(self.idx.sensor_names):
            if s_name in ("EO", "SAR"):
                eo_sar.append(si)
        return eo_sar

    # ---- 约束 (8c): 打击前必须 SAR 搜索过目标所在栅格 ----
    def _c8c_strike_needs_sar(self, vars_, model, solver):
        """z[h,g] <= sum_u x[u,SAR,cell_of_g]"""
        idx = self.idx
        p = self.params
        if idx.N_H == 0 or idx.N_G == 0:
            return
        sar_idx = None
        for si, s_name in enumerate(idx.sensor_names):
            if s_name == "SAR":
                sar_idx = si
                break
        if sar_idx is None:
            return
        for h in range(idx.N_H):
            for g in range(idx.N_G):
                ci = p.target_cell_idx[g]
                solver.add_constraint(
                    model,
                    vars_.z[(h, g)] <= mip.xsum(
                        vars_.x[(u, sar_idx, ci)] for u in range(idx.N_U)
                    ),
                    name=f"c8c_strike_needs_sar_H{h}_G{g}"
                )

    # ---- 约束 (8d): 好天气时打击还需要 EO 确认（坏天气跳过） ----
    def _c8d_strike_needs_eo(self, vars_, model, solver):
        """w[ci] < 0.80 时: z[h,g] <= sum_u x[u,EO,cell_of_g]"""
        idx = self.idx
        p = self.params
        if idx.N_H == 0 or idx.N_G == 0:
            return
        eo_idx = None
        for si, s_name in enumerate(idx.sensor_names):
            if s_name == "EO":
                eo_idx = si
                break
        if eo_idx is None:
            return
        for h in range(idx.N_H):
            for g in range(idx.N_G):
                ci = p.target_cell_idx[g]
                if p.cell_weather[ci] >= 0.80:
                    continue
                solver.add_constraint(
                    model,
                    vars_.z[(h, g)] <= mip.xsum(
                        vars_.x[(u, eo_idx, ci)] for u in range(idx.N_U)
                    ),
                    name=f"c8d_strike_needs_eo_H{h}_G{g}"
                )
