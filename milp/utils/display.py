"""
终端显示管理器 —— 以非侵入方式向终端输出执行进度、态势快照、分配结果。

职责:
  1. 通过 GlobalSettings.verbose 控制输出级别
  2. 以表格/分隔线形式格式化输出态势、参数、求解结果、分配方案
  3. 纯显示模块，不修改任何 MILP 核心逻辑

对外接口:
  - DisplayManager  — 唯一的显示管理类

使用示例:
    >>> from config.settings import GlobalSettings
    >>> from utils.display import DisplayManager
    >>> dm = DisplayManager(settings)
    >>> dm.show_snapshot(snapshot)
    >>> dm.show_solve_result(plan, 24.1)
    >>> dm.show_assignments(plan)
"""

import numpy as np
from config.settings import GlobalSettings

# Sentinel for "all" in cell selection (c0 = patrol cell)
_PATROL_CELL = "c0"


class DisplayManager:
    """
    终端显示管理器。

    通过 GlobalSettings.verbose 控制输出级别:
        0 = QUIET   — 完全静默，适用于生产模式
        1 = NORMAL  — 每轮态势摘要 + 求解结果 + 分配方案
        2 = VERBOSE — NORMAL + 参数矩阵 + 变量/约束数量 + 求解器日志
    """

    def __init__(self, settings: GlobalSettings):
        """根据 settings.verbose 初始化静默标志。"""
        self.settings = settings
        self._v = settings.verbose

    # ========================================================================
    # 公共显示方法
    # ========================================================================

    def show_snapshot(self, snap) -> None:
        """
        显示态势快照总览：平台列表、目标列表、栅格信息。
        仅在 verbose >= 1 时输出。
        """
        if self._v < 1:
            return
        print()
        print(self._bar("=", 72))
        print(f"  第 {snap.cycle_id} 轮态势快照 | 时间戳: {snap.timestamp:.1f} | "
              f"AOI: {','.join(snap.commander_AOI) if snap.commander_AOI else '全部'}")
        print(self._bar("=", 72))
        self._show_platforms(snap)
        self._show_targets(snap)
        self._show_grids(snap)

    def show_parameters(self, params) -> None:
        """
        显示参数矩阵维度与关键数值（天气增益、距离矩阵）。
        仅在 verbose >= 2 时输出。

        Args:
            params: ParameterBuilder 实例
        """
        if self._v < 2:
            return
        print()
        print(self._bar("-", 64))
        print(f"  [参数详情] verbose={self._v}")
        print(f"  [维度] N_U={params.N_U}  N_H={params.N_H}  N_G={params.N_G}  "
              f"N_C={params.N_C}  N_S={params.N_S}  N_W={params.N_W}")
        var_count = (
            params.N_U * params.N_S * params.N_C   # x
            + params.N_U * params.N_S               # ell
            + params.N_C * 2                        # q + xi
            + (params.N_H * params.N_G * 2 if params.N_G > 0 else 0)  # z + r_L + r_W
            + (params.N_H * params.N_W * params.N_G if params.N_G > 0 else 0)  # y
            + (params.N_G * 2 if params.N_G > 0 else 0)  # k + u
        )
        print(f"  [变量] 估约 {var_count} 个")

        self._show_beta_matrix(params)
        self._show_distance_matrix(params)
        self._show_weapon_range_info(params)

    def show_model_built(self, idx, elapsed_ms: float) -> None:
        """
        显示 MILP 模型构建完成信息。
        仅在 verbose >= 2 时输出。

        Args:
            idx: VariableIndices 实例
            elapsed_ms: 模型构建耗时 (ms)
        """
        if self._v < 2:
            return
        n_u, n_h, n_g, n_c, n_s, n_w = (
            idx.N_U, idx.N_H, idx.N_G, idx.N_C, idx.N_S, idx.N_W
        )
        x_count = n_u * n_s * n_c
        z_count = n_h * n_g
        y_count = n_h * n_w * n_g if n_g > 0 else 0
        total = x_count + n_u * n_s + n_c * 2 + z_count + y_count
        if n_g > 0:
            total += n_g * 2 + n_h * n_g * 2  # k + u + r_L + r_W
        print(f"  [模型] 构建完成 | 变量: ~{total} | 耗时: {self._fmt_ms(elapsed_ms)}")

    def show_solve_result(self, plan, elapsed_ms: float) -> None:
        """
        显示求解状态、目标值、MIP Gap、耗时。
        仅在 verbose >= 1 时输出。

        Args:
            plan: AllocationPlan 实例
            elapsed_ms: 从 solve() 进入到此处的总耗时 (ms)
        """
        if self._v < 1:
            return
        status_icon = "OK" if plan.status == "OPTIMAL" else ("WARN" if plan.status in ("FEASIBLE", "TIME_LIMIT") else "FAIL")
        print()
        print(self._bar("-", 64))
        print(f"  [{status_icon}] 求解状态: {plan.status} | 目标值: {plan.objective:.2f} | "
              f"MIP Gap: {plan.mip_gap:.4f}")
        print(f"  [T] 求解耗时: {self._fmt_ms(plan.solve_time_ms)}"
              f"{' (总: ' + self._fmt_ms(elapsed_ms) + ')' if elapsed_ms != plan.solve_time_ms else ''}"
              f" | 求解器: {plan.solver_used}")
        print(self._bar("-", 64))

    def show_assignments(self, plan) -> None:
        """
        以表格形式显示侦察和打击分配方案。
        仅在 verbose >= 1 时输出。
        """
        if self._v < 1:
            return
        print()

        # 侦察分配
        if plan.recon_assignments:
            print(f"  侦察分配 ({len(plan.recon_assignments)} 条):")
            headers = ["平台", "传感器", "栅格", "角色"]
            rows = [[ra.pid, ra.sensor_used, ra.cell, ra.role]
                    for ra in plan.recon_assignments]
            print(self._format_table(headers, rows, [8, 8, 8, 12]))
        else:
            print(f"  侦察分配: 无")

        # 打击分配
        if plan.strike_assignments:
            print(f"  打击分配 ({len(plan.strike_assignments)} 条):")
            headers = ["平台", "目标", "弹药×数量", "角色", "说明"]
            rows = []
            for sa in plan.strike_assignments:
                weapon_str = f"{sa.munition}x{sa.qty}" if sa.munition else "-"
                note = ""
                if "_support" in sa.role:
                    note = "提供平台数"
                elif sa.role == "lead":
                    note = "主攻"
                elif sa.role == "wing":
                    note = "僚机"
                rows.append([sa.pid, sa.target, weapon_str, sa.role.replace("_support", "支援"), note])
            print(self._format_table(headers, rows, [8, 8, 12, 10, 14]))
        else:
            print(f"  打击分配: 无")
        print()

    def show_cycle_summary(self, cycle_id: int, plan) -> None:
        """
        单行本轮摘要。
        仅在 verbose >= 1 时输出。
        """
        if self._v < 1:
            return
        status_icon = "OK" if plan.status == "OPTIMAL" else "WARN"
        print(f"  [{status_icon}] 第{cycle_id}轮摘要 | "
              f"{plan.status} | 目标值: {plan.objective:.2f} | "
              f"侦察: {len(plan.recon_assignments)} | "
              f"打击: {len(plan.strike_assignments)} | "
              f"耗时: {self._fmt_ms(plan.solve_time_ms)}")
        print(self._bar("=", 72))

    # ========================================================================
    # 内部显示辅助
    # ========================================================================

    def _show_platforms(self, snap) -> None:
        """显示平台列表表格。"""
        uavs = [p for p in snap.platforms if p.type == "UAV"]
        helis = [p for p in snap.platforms if p.type == "HELI"]
        print(f"  平台 ({len(snap.platforms)}): {len(uavs)} UAV + {len(helis)} HELI")
        headers = ["PID", "类型", "位置 (km)", "高(km)", "传感器", "弹药"]
        rows = []
        for p in snap.platforms:
            pos = f"({p.pos[0]:.0f}, {p.pos[1]:.0f})"
            sensors = ",".join(p.sensors_mounted[:3])
            if p.type == "HELI":
                ammo = " ".join(f"{k}:{v}" for k, v in p.munitions.items())
            else:
                ammo = "-"
            rows.append([p.pid, p.type, pos, f"{p.alt:.1f}", sensors, ammo])
        print(self._format_table(headers, rows, [6, 6, 14, 8, 16, 24]))

    def _show_targets(self, snap) -> None:
        """显示目标列表表格。"""
        active = [t for t in snap.targets if t.alive and t.confirmed]
        total = len(snap.targets)
        if total == 0:
            print(f"  目标: 无")
            return
        print(f"  目标 ({total} | 活跃: {len(active)}):")
        headers = ["TID", "类型", "位置 (km)", "确认/存活", "价值", "威胁"]
        rows = []
        for t in snap.targets:
            pos = f"({t.pos_est[0]:.0f}, {t.pos_est[1]:.0f})"
            conf = "Y" if t.confirmed else "N"
            alive = "Y" if t.alive else "N"
            rows.append([t.tid, t.type, pos, f"{conf}/{alive}",
                         f"{t.value:.2f}", f"{t.threat:.2f}"])
        print(self._format_table(headers, rows, [6, 8, 14, 10, 8, 8]))

    def _show_grids(self, snap) -> None:
        """显示栅格列表表格。"""
        if not snap.grids:
            return
        print(f"  栅格 ({len(snap.grids)}):")
        headers = ["Cell", "中心 (km)", "天气 w", "先验 ρ", "已覆盖"]
        rows = []
        for g in snap.grids:
            pos = f"({g.center[0]:.0f}, {g.center[1]:.0f})"
            covered = "Y" if g.covered else "-"
            rows.append([g.cell_id, pos, f"{g.weather_w:.2f}",
                         f"{g.target_prior:.2f}", covered])
        print(self._format_table(headers, rows, [8, 14, 10, 10, 8]))

    def _show_beta_matrix(self, params) -> None:
        """显示天气增益矩阵 β_{s,c}。"""
        if params.beta_sc.size == 0:
            return
        print(f"  [天气增益 β ({params.N_S}×{params.N_C})]:")
        header = "        " + "  ".join(f"{params.cell_ids[c]:>6s}" for c in range(params.N_C))
        print(header)
        for si, s_name in enumerate(params.sensor_names):
            vals = "  ".join(f"{params.beta_sc[si, c]:6.2f}" for c in range(params.N_C))
            print(f"    {s_name:4s} {vals}")

    def _show_distance_matrix(self, params) -> None:
        """显示直升机-目标距离矩阵 D_hg。"""
        if params.D_hg.size == 0:
            return
        print(f"  [直升机-目标距离 (km)  ({params.N_H}×{params.N_G})]:")
        header = "        " + "  ".join(f" {params.active_targets[g].tid if g < len(params.active_targets) else f'g{g+1}':>6s}" for g in range(params.N_G))
        print(header)
        for h in range(params.N_H):
            pid = params.heli_platforms[h].pid
            vals = "  ".join(f"{params.D_hg[h, g]:6.2f}" for g in range(params.N_G))
            print(f"    {pid:4s} {vals}")

    def _show_weapon_range_info(self, params) -> None:
        """显示武器射程可行性汇总。"""
        if params.weapon_range_feasible.size == 0:
            return
        infeasible_count = int(np.sum(params.weapon_range_feasible < 0.5))
        total_count = int(np.prod(params.weapon_range_feasible.shape))
        if infeasible_count > 0:
            print(f"  [射程] {infeasible_count}/{total_count} 个 (直升机,武器,目标) 组合超出射程，已裁剪")
        else:
            print(f"  [射程] 全部 {total_count} 个组合在射程内")

    # ========================================================================
    # 格式化工具方法
    # ========================================================================

    def _format_table(self, headers: list, rows: list, col_widths: list) -> str:
        """
        构造 ASCII 表格字符串。

        Args:
            headers: 表头列表
            rows: 数据行列表，每行是等长列表
            col_widths: 各列宽度列表

        Returns:
            包含换行的表格字符串
        """
        def _row(cells, widths):
            parts = [f" {str(cells[i]):<{widths[i]}} " for i in range(len(cells))]
            return "|" + "|".join(parts) + "|"

        top = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        bot = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

        lines = [top, _row(headers, col_widths), sep]
        for row in rows:
            lines.append(_row(row, col_widths))
        lines.append(bot)
        return "\n".join(lines)

    def _format_timing(self, label: str, ms: float) -> str:
        """格式化带标签的计时信息。"""
        return f"  [{label}] {self._fmt_ms(ms)}"

    def _bar(self, char: str = "-", width: int = 72) -> str:
        """生成分隔线。"""
        return char * width

    @staticmethod
    def _fmt_ms(ms: float) -> str:
        """人性化毫秒显示。"""
        if ms < 1.0:
            return f"{ms * 1000:.0f} μs"
        elif ms < 1000:
            return f"{ms:.1f} ms"
        else:
            return f"{ms / 1000:.2f} s"
