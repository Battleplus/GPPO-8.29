"""
多 AOI 任务控制模块：AOI 排序与原 MILP 之间的接口层。

职责：
  1. 解析多 AOI 输入（aois / platforms / targets / sensor_params）
  2. 判断是否需要排序（首次调用 vs 续调）
  3. 根据底层反馈更新 AOI 状态
  4. 构造当前 AOI 的单区域 SituationSnapshot
  5. 调用原 MILP（TaskAllocator.solve）
  6. 封装输出（aoi_route_state + current_aoi_plan）

注意：
  - 不修改 MILP 内部，只调用 TaskAllocator.solve(snapshot)
  - 每次 MILP 仍只处理一个 AOI
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .aoi_state import AoiInfo, AOIRouteState, ExecutionFeedback
from .aoi_router import AOIRouter
from core.snapshot import (
    SituationSnapshot,
    GridInfo,
    TargetInfo,
    PlatformInfo,
    SensorParams,
    generate_aoi_grids,
)
from core.allocation import AllocationPlan


# ── 解析辅助函数 ──────────────────────────────────────────

def _parse_aois(aois_raw: List[Dict]) -> List[AoiInfo]:
    """从 JSON 列表解析 AoiInfo 对象。"""
    result = []
    for a in aois_raw:
        result.append(AoiInfo(
            id=a["id"],
            row=int(a["row"]),
            col=int(a["col"]),
            priority=float(a.get("priority", 1.0)),
            target_prior=float(a.get("target_prior", 0.25)),
            target_value=float(a.get("target_value", 0.5)),
            target_threat=float(a.get("target_threat", 0.5)),
        ))
    return result


def _parse_platforms(plat_cfg: Any, staging_pos: np.ndarray) -> List[PlatformInfo]:
    """Parse aggregate platform config or the per-platform Isaac format."""
    platforms: List[PlatformInfo] = []
    if isinstance(plat_cfg, list):
        for item in plat_cfg:
            ptype = str(item["type"]).upper()
            platforms.append(PlatformInfo(
                pid=str(item["pid"]),
                type=ptype,
                pos=np.array(item.get("pos", staging_pos), dtype=np.float64),
                alt=float(item.get("alt", 2.0 if ptype == "UAV" else 3.0)),
                lost=bool(item.get("lost", False)),
                sensors_mounted=list(item.get(
                    "sensors_mounted",
                    ["EO", "SAR", "ESM"] if ptype == "UAV" else ["MMW", "EOIR"],
                )),
                munitions=dict(item.get("munitions", {})),
            ))
        return platforms

    for ptype, cfg in plat_cfg.items():
        count = cfg["count"]
        sensors = cfg.get(
            "sensors",
            ["EO", "SAR", "ESM"] if ptype == "UAV" else ["MMW", "EOIR"],
        )
        munitions = cfg.get(
            "munitions",
            {"HF": 0, "RKT": 0, "GUN": 0} if ptype == "UAV"
            else {"HF": 16, "RKT": 76, "GUN": 1200},
        )
        alt = cfg.get("alt", 2.0 if ptype == "UAV" else 3.0)
        prefix = "U" if ptype == "UAV" else "H"
        pos_raw = cfg.get("pos", staging_pos.tolist())
        for i in range(1, count + 1):
            platforms.append(PlatformInfo(
                pid=f"{prefix}{i}",
                type=ptype,
                pos=np.array(pos_raw, dtype=np.float64).copy(),
                alt=alt,
                lost=False,
                sensors_mounted=list(sensors),
                munitions=dict(munitions),
            ))
    return platforms


def _parse_targets(targets_raw: List[Dict]) -> List[TargetInfo]:
    """从 targets 列表解析 TargetInfo 对象。"""
    targets: List[TargetInfo] = []
    for t in targets_raw:
        pos_cov = np.array(
            t.get("pos_cov", [[0.1, 0], [0, 0.1]]), dtype=np.float64
        )
        velocity = np.array(t.get("velocity", [0.0, 0.0]), dtype=np.float64)
        targets.append(TargetInfo(
            tid=t["tid"],
            type=t["type"],
            pos_est=np.array(t["pos"], dtype=np.float64),
            pos_cov=pos_cov,
            velocity_est=velocity,
            confirmed=t.get("confirmed", True),
            alive=t.get("alive", True),
            value=float(t.get("value", 0.8)),
            threat=float(t.get("threat", 0.5)),
        ))
    return targets


def _parse_sensor_params(sp_raw: Optional[List[Dict]]) -> List[SensorParams]:
    """从 sensor_params 列表解析，若为空则返回默认三传感器参数。"""
    if not sp_raw:
        return [
            SensorParams(name="EO",  P0=0.85, R=15.0,  weather_sensitive=True),
            SensorParams(name="SAR", P0=0.90, R=50.0,  weather_sensitive=False),
            SensorParams(name="ESM", P0=0.80, R=100.0, weather_sensitive=False),
        ]
    return [
        SensorParams(
            name=s["name"],
            P0=float(s.get("P0", 0.85)),
            R=float(s.get("R", 50.0)),
            weather_sensitive=bool(s.get("weather_sensitive", True)),
        )
        for s in sp_raw
    ]


def _filter_targets_in_aoi(
    targets: List[TargetInfo],
    aoi_row: int,
    aoi_col: int,
    grid_size_km: float = 50.0,
) -> List[TargetInfo]:
    """
    筛选坐标落在指定 AOI 边界内的目标。

    AOI (row, col) 的 x 范围：[(col-1)*gs, col*gs]
    AOI (row, col) 的 y 范围：[(row-1)*gs, row*gs]
    """
    x_min = (aoi_col - 1) * grid_size_km
    x_max = aoi_col * grid_size_km
    y_min = (aoi_row - 1) * grid_size_km
    y_max = aoi_row * grid_size_km
    return [
        t for t in targets
        if x_min <= t.pos_est[0] <= x_max
        and y_min <= t.pos_est[1] <= y_max
    ]


def _build_snapshot(
    aoi_info: AoiInfo,
    platforms: List[PlatformInfo],
    targets_all: List[TargetInfo],
    sensor_params: List[SensorParams],
    staging_pos: np.ndarray,
    cycle_id: int,
    grid_weather: Optional[Dict[str, float]] = None,
    grid_size_km: float = 50.0,
) -> SituationSnapshot:
    """
    为指定 AOI 构造单区域 SituationSnapshot，供原 MILP 直接消费。

    目标筛选：只保留坐标落在该 AOI 边界内的目标。
    栅格：调用 generate_aoi_grids 生成 c0~c4 五个栅格。
    """
    grids = generate_aoi_grids(
        aoi_row=aoi_info.row,
        aoi_col=aoi_info.col,
        target_prior=aoi_info.target_prior,
    )

    # 覆盖天气参数（可选）
    if grid_weather:
        for g in grids:
            if g.cell_id in grid_weather:
                g.weather_w = grid_weather[g.cell_id]

    targets_in_aoi = _filter_targets_in_aoi(
        targets_all, aoi_info.row, aoi_info.col, grid_size_km
    )

    n_heli = sum(1 for p in platforms if p.type == "HELI")
    n_tgt = len(targets_in_aoi)
    los = np.ones((n_heli, max(n_tgt, 1)))
    occ = np.ones((n_heli, max(n_tgt, 1)))

    return SituationSnapshot(
        cycle_id=cycle_id,
        timestamp=float(cycle_id),
        grids=grids,
        targets=targets_in_aoi,
        platforms=platforms,
        sensor_params=sensor_params,
        commander_AOI=[aoi_info.id],
        staging_position=staging_pos,
        los_matrix=los if n_tgt > 0 else None,
        occlusion_matrix=occ if n_tgt > 0 else None,
    )


# ── 主控制器 ─────────────────────────────────────────────

class MultiAOIController:
    """
    多 AOI 任务控制器。

    使用方法::

        from task_interface import TaskAllocator
        from aoi import MultiAOIController

        allocator = TaskAllocator(solver="cbc", verbose=1)
        controller = MultiAOIController(allocator)

        # 首次调用（aoi_route_state=null）
        result = controller.run(input_data)

        # 后续调用（带回 aoi_route_state 与 execution_feedback）
        input_data["aoi_route_state"] = result["aoi_route_state"]
        input_data["execution_feedback"] = {...}
        result = controller.run(input_data)
    """

    def __init__(
        self,
        allocator,               # TaskAllocator 实例
        grid_size_km: float = 50.0,
    ):
        self.allocator = allocator
        self.router = AOIRouter(grid_size_km=grid_size_km)
        self.grid_size_km = grid_size_km
        self._aoi_map: Dict[str, AoiInfo] = {}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        多 AOI 任务分配主调用接口。

        输入字段:
            aois                — 候选 AOI 列表（必填）
            platforms           — 平台配置 dict
            targets             — 目标列表，confirmed 字段由态势理解维护
            sensor_params       — 传感器参数（可选）
            staging_position    — 出发点坐标（可选，默认 [150, -50]）
            cycle_id            — 当前分配轮次（可选）
            grid_weather        — 各子区天气覆盖（可选）
            aoi_route_state     — 上次返回的 AOI 执行状态（上层持久保存带回，首次传 null）
            execution_feedback  — 底层规划返回的执行反馈（可选）

        输出字段:
            status              — "AOI_PLAN_READY" | "ALL_AOI_FINISHED"
            aoi_route_state     — 最新 AOI 执行状态（上层必须保存并带回）
            current_aoi_plan    — 当前 AOI 任务分配方案
                ├── aoi         — AOI ID
                ├── tasks       — 任务列表（先侦察后打击）
                ├── two_phase   — 是否触发两阶段（侦察→反馈→打击）
                ├── solve_status / objective / solve_time_ms
        """
        # ── Step 1: 解析 AOI → 建立 id->info 映射 ──────────
        aois = _parse_aois(input_data["aois"])
        self._aoi_map = {a.id: a for a in aois}
        # 提前解析目标，供 AOI 排序时判断各区域是否已有确认目标
        targets_all = _parse_targets(input_data.get("targets", []))

        # ── Step 2: 确定 AOI 执行状态（首次排序 or 续调推进）─
        staging_pos = np.array(
            input_data.get("staging_position", [150.0, -50.0]), dtype=np.float64
        )
        state_raw = input_data.get("aoi_route_state")
        aoi_just_advanced = False

        if state_raw is None:
            state = self.router.sort(aois, start_pos=staging_pos, targets=targets_all)
        else:
            state = AOIRouteState.from_dict(state_raw)
            feedback_raw = input_data.get("execution_feedback")
            if feedback_raw is not None:
                feedback = ExecutionFeedback.from_dict(feedback_raw)
                if (
                    feedback.aoi_id == state.current_aoi
                    and feedback.aoi_status == "FINISHED"
                ):
                    # AOI 推进：将出发点更新为刚完成 AOI 的中心
                    finished_info = self._aoi_map.get(feedback.aoi_id)
                    if finished_info:
                        cx = (finished_info.col - 0.5) * self.grid_size_km
                        cy = (finished_info.row - 0.5) * self.grid_size_km
                        staging_pos = np.array([cx, cy], dtype=np.float64)
                    state.advance()
                    aoi_just_advanced = True

        # ── Step 3: 全部完成判断 ───────────────────────────
        if state.is_finished():
            return {
                "status": "ALL_AOI_FINISHED",
                "aoi_route_state": state.to_dict(),
                "current_aoi_plan": None,
            }

        # ── Step 4: 解析平台（平台从当前 staging 出发） ─
        cycle_id = int(input_data.get("cycle_id", 0))
        platforms = _parse_platforms(input_data.get("platforms", {}), staging_pos)
        # 若 AOI 刚推进，覆盖所有平台位置为上一步 AOI 中心
        if aoi_just_advanced:
            for p in platforms:
                p.pos = staging_pos.copy()
        sensor_params = _parse_sensor_params(input_data.get("sensor_params"))
        grid_weather = input_data.get("grid_weather")

        # ── Step 5: 构造当前 AOI 的 Snapshot → MILP ─────────
        current_aoi_id = state.current_aoi
        current_aoi_info = self._aoi_map[current_aoi_id]

        snapshot = _build_snapshot(
            aoi_info=current_aoi_info,
            platforms=platforms,
            targets_all=targets_all,
            sensor_params=sensor_params,
            staging_pos=staging_pos,
            cycle_id=cycle_id,
            grid_weather=grid_weather,
            grid_size_km=self.grid_size_km,
        )
        plan: AllocationPlan = self.allocator.solve(snapshot)

        # ── Step 6: 构造输出 ──────────────────────────────
        tasks = self._format_tasks(plan, current_aoi_id)

        return {
            "status": "AOI_PLAN_READY",
            "aoi_route_state": state.to_dict(),
            "current_aoi_plan": {
                "aoi": current_aoi_id,
                "tasks": tasks,
                "solve_status": plan.status,
                "objective": plan.objective,
                "solve_time_ms": plan.solve_time_ms,
                "mounted_sensors": dict(plan.mounted_sensors),
            },
        }

    @staticmethod
    def _format_tasks(plan: AllocationPlan, aoi_id: str) -> List[Dict[str, Any]]:
        """将 AllocationPlan 转为精简任务清单（与 make_execution_order 格式兼容）。"""
        tasks = []
        for ra in plan.recon_assignments:
            tasks.append({
                "platform":  ra.pid,
                "task_type": "recon",
                "aoi":       aoi_id,
                "cell":      ra.cell,
                "sensor":    ra.sensor_used,
                "sensors_mounted": ra.sensors_mounted,
                "role":      ra.role,
            })
        for sa in plan.strike_assignments:
            tasks.append({
                "platform":  sa.pid,
                "task_type": "strike",
                "aoi":       aoi_id,
                "target":    sa.target,
                "munition":  sa.munition,
                "qty":       sa.qty,
                "role":      sa.role,
                "sensors_mounted": plan.mounted_sensors.get(sa.pid, []),
            })
        return tasks
