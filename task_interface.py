"""
任务分配模块 —— 对外统一接口。

本模块是无人/有人机协同侦察-打击任务分配系统的唯一入口。
外部调用者只需导入本文件，无需了解内部子模块结构。

=== 快速开始 ===

    from task_interface import TaskAllocator, make_snapshot

    # 1. 创建分配器（全局复用）
    allocator = TaskAllocator()

    # 2. 构造态势快照
    snapshot = make_snapshot(
        cycle_id=0,
        grids=[...],
        targets=[...],
        platforms=[...],
        sensor_params=[...],
    )

    # 3. 求解
    plan = allocator.solve(snapshot)

    # 4. 读取结果
    for ra in plan.recon_assignments:
        print(ra.pid, ra.sensor_used, ra.cell)
    for sa in plan.strike_assignments:
        print(sa.pid, sa.target, sa.munition, sa.qty)

=== 数据结构速查 ===

    输入:
        GridInfo       — 栅格（侦察子区），含天气、先验概率
        TargetInfo     — 目标，含类型(RADAR/CP/AV)、位置、价值、威胁
        PlatformInfo   — 平台(UAV/HELI)，含位置、可用传感器、弹药
        SensorParams   — 传感器固有参数(P0, 作用距离, 天气敏感性)
        SituationSnapshot — 以上四者的聚合容器

    输出:
        ReconAssignment  — 单条侦察分配 (平台→传感器→栅格)
        StrikeAssignment — 单条打击分配 (平台→目标→弹药×数量)
        AllocationPlan   — 完整分配方案 (含求解元数据)

    配置:
        TaskAllocator(solver="cbc", time_limit_s=3.0, verbose=0)
"""

import json
from pathlib import Path
import numpy as np
from typing import List, Optional, Dict

# ── 内部导入 ─────────────────────────────────────────
from config.settings import GlobalSettings, SolverType
from core.snapshot import (
    GridInfo,
    TargetInfo,
    PlatformInfo,
    SensorParams,
    SituationSnapshot,
    generate_aoi_grids,
)
from core.allocation import (
    ReconAssignment,
    StrikeAssignment,
    AllocationPlan,
)
from allocation.milp_allocator import MILPAllocator
from situation_input_interface import (
    SituationInput, build_situation_input, validate_situation_input,
    convert_to_snapshot as _convert_to_snapshot,
)
from execution_output_interface import (
    ExecutionOrder, build_execution_order as _build_execution_order,
    to_json as execution_order_to_json,
)

   
# ── 对外导出清单 ─────────────────────────────────────  使用from task_interface import *导入
__all__ = [
    # 主类
    "TaskAllocator",
    # 一站式求解
    "solve",
    "solve_file",
    # 输入数据结构
    "GridInfo",
    "TargetInfo",
    "PlatformInfo",
    "SensorParams",
    "SituationSnapshot",
    "SituationInput",
    # 输出数据结构
    "ReconAssignment",
    "StrikeAssignment",
    "AllocationPlan",
    "ExecutionOrder",
    # 工具函数
    "generate_aoi_grids",
    "make_snapshot",
    "make_grid",
    "make_target",
    "make_platform",
    "make_sensor_params",
    "load_snapshot_from_json",
    "load_all_scenarios",
    "make_execution_order",
    "save_execution_order",
    # 配置枚举
    "SolverType",
    # 解析辅助函数（供 aoi 子包内部使用）
    "_parse_platforms_from_dict",
    "_parse_targets_from_dict",
    "_parse_sensor_params_from_dict",
]

# ── 求解器中文名映射 ─────────────────────────────────
_SOLVER_ALIAS = {
    "cbc":     SolverType.CBC,
    "gurobi":  SolverType.GUROBI,
    "ortools": SolverType.ORTOOLS,
    "highs":   SolverType.HIGHS,
}


class TaskAllocator:
    """
    任务分配器 —— 对外唯一调用入口。

    封装了求解器选择、参数配置和 MILP 模型求解。
    单实例可跨仿真轮次复用（内置热启动）。

    Parameters
    ----------
    solver : str
        求解器名称: "cbc"(默认) / "gurobi" / "ortools" / "highs"
    time_limit_s : float
        单次求解时间上限 (秒), 默认 3.0
    mip_gap : float
        MIP 最优间隙阈值, 默认 1e-3
    verbose : int
        终端输出级别: 0=静默, 1=摘要, 2=详细
    """

    def __init__(
        self,
        solver: str = "cbc",
        time_limit_s: float = 3.0,
        mip_gap: float = 1e-3,
        verbose: int = 0,
    ):
        if solver not in _SOLVER_ALIAS:
            raise ValueError(
                f"未知求解器 '{solver}'，可选: {list(_SOLVER_ALIAS.keys())}"
            )
        self.settings = GlobalSettings(
            active_solver=_SOLVER_ALIAS[solver],
            solver_time_limit_s=time_limit_s,
            solver_mip_gap=mip_gap,
            verbose=verbose,
        )
        self._allocator = MILPAllocator(self.settings)

    def solve(self, snapshot: SituationSnapshot) -> AllocationPlan:
        """
        对给定态势快照求解任务分配方案。

        Parameters
        ----------
        snapshot : SituationSnapshot
            态势理解模块产出的态势快照

        Returns
        -------
        AllocationPlan
            包含侦察分配列表和打击分配列表
        """
        return self._allocator.solve(snapshot)

    @property
    def solver_name(self) -> str:
        """当前使用的求解器名称。"""
        return self._allocator._solver.solver_name


# ── 工厂函数 ──────────────────────────────────────────

def make_grid(
    cell_id: str,
    center_xy: tuple,
    width_km: float = 25.0,
    height_km: float = 25.0,
    weather_w: float = 0.2,
    terrain_level: int = 0,
    target_prior: float = 0.25,
    covered: bool = False,
) -> GridInfo:
    """快捷构造一个栅格。"""
    return GridInfo(
        cell_id=cell_id,
        center=np.array(center_xy, dtype=np.float64),
        width_km=width_km,
        height_km=height_km,
        weather_w=weather_w,
        terrain_level=terrain_level,
        target_prior=target_prior,
        covered=covered,
    )


def make_target(
    tid: str,
    target_type: str,
    pos_xy: tuple,
    confirmed: bool = True,
    alive: bool = True,
    value: float = 0.8,
    threat: float = 0.5,
    pos_cov: Optional[np.ndarray] = None,
    velocity: tuple = (0.0, 0.0),
) -> TargetInfo:
    """快捷构造一个目标。"""
    if pos_cov is None:
        pos_cov = np.eye(2) * 0.1
    return TargetInfo(
        tid=tid,
        type=target_type,
        pos_est=np.array(pos_xy, dtype=np.float64),
        pos_cov=pos_cov,
        velocity_est=np.array(velocity, dtype=np.float64),
        confirmed=confirmed,
        alive=alive,
        value=value,
        threat=threat,
    )


def make_platform(
    pid: str,
    platform_type: str,
    pos_xy: tuple,
    sensors_mounted: Optional[List[str]] = None,
    munitions: Optional[Dict[str, int]] = None,
    alt: float = 2.0,
    lost: bool = False,
) -> PlatformInfo:
    """快捷构造一个平台。sensors_mounted=None 时由算法自动分配传感器。"""
    if platform_type == "UAV":
        munitions = munitions or {"HF": 0, "RKT": 0, "GUN": 0}
        alt = 2.0
    elif platform_type == "HELI":
        munitions = munitions or {"HF": 16, "RKT": 76, "GUN": 1200}
        alt = 3.0
    else:
        raise ValueError(f"未知平台类型 '{platform_type}'，应为 UAV 或 HELI")
    return PlatformInfo(
        pid=pid,
        type=platform_type,
        pos=np.array(pos_xy, dtype=np.float64),
        alt=alt,
        lost=lost,
        sensors_mounted=sensors_mounted or [],
        munitions=munitions,
    )


def make_sensor_params(
    name: str,
    P0: float = 0.85,
    R: float = 50.0,
    weather_sensitive: bool = True,
) -> SensorParams:
    """快捷构造一个传感器参数条目。"""
    return SensorParams(
        name=name,
        P0=P0,
        R=R,
        weather_sensitive=weather_sensitive,
    )


def make_snapshot(
    cycle_id: int = 0,
    timestamp: float = 0.0,
    grids: Optional[List[GridInfo]] = None,
    targets: Optional[List[TargetInfo]] = None,
    platforms: Optional[List[PlatformInfo]] = None,
    sensor_params: Optional[List[SensorParams]] = None,
    commander_AOI: Optional[List[str]] = None,
    staging_position: Optional[np.ndarray] = None,
    los_matrix: Optional[np.ndarray] = None,
    occlusion_matrix: Optional[np.ndarray] = None,
) -> SituationSnapshot:
    """
    快捷构造一个完整的态势快照。

    除 grids/platforms 外，其余字段均有合理默认值。

    Parameters
    ----------
    cycle_id : int
        当前分配轮次编号
    timestamp : float
        仿真时间戳
    grids : list of GridInfo
        栅格列表（必填）
    targets : list of TargetInfo
        目标列表，默认空
    platforms : list of PlatformInfo
        平台列表（必填）
    sensor_params : list of SensorParams
        传感器参数，默认 [EO, SAR, ESM] 标准参数
    commander_AOI : list of str
        指挥官关注的 AOI 栅格列表
    staging_position : np.ndarray
        集结区坐标 (2,)，默认 None 表示已在任务区内
    los_matrix : np.ndarray
        直升机-目标通视矩阵, shape (N_H, N_G)，默认全 1
    occlusion_matrix : np.ndarray
        遮挡衰减矩阵, shape (N_H, N_G)，默认全 1

    Returns
    -------
    SituationSnapshot
    """
    if sensor_params is None:
        sensor_params = [
            make_sensor_params("EO",  P0=0.85, R=15.0,  weather_sensitive=True),
            make_sensor_params("SAR", P0=0.90, R=50.0,  weather_sensitive=False),
            make_sensor_params("ESM", P0=0.80, R=100.0, weather_sensitive=False),
        ]
    if targets is None:
        targets = []
    if commander_AOI is None:
        commander_AOI = []

    N_H = sum(1 for p in platforms if p.type == "HELI")
    N_G = len(targets)
    if los_matrix is None and N_H > 0 and N_G > 0:
        los_matrix = np.ones((N_H, N_G))
    if occlusion_matrix is None and N_H > 0 and N_G > 0:
        occlusion_matrix = np.ones((N_H, N_G))

    return SituationSnapshot(
        cycle_id=cycle_id,
        timestamp=timestamp,
        grids=grids or [],
        targets=targets,
        platforms=platforms or [],
        sensor_params=sensor_params,
        commander_AOI=commander_AOI,
        staging_position=staging_position,
        los_matrix=los_matrix,
        occlusion_matrix=occlusion_matrix,
    )


# ── JSON 解析辅助函数（供内部及 aoi 子包共用）────────────

def _parse_platforms_from_dict(
    plat_cfg: Dict,
    default_pos: List,
) -> List[PlatformInfo]:
    """
    从 platforms 配置字典解析 PlatformInfo 列表。

    Args:
        plat_cfg:    JSON 中的 "platforms" 字段值（dict）
        default_pos: 当平台无显式 pos 时使用的默认坐标

    Returns:
        PlatformInfo 列表
    """
    platforms: List[PlatformInfo] = []
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
        pos_raw = cfg.get("pos", default_pos)
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


def _parse_targets_from_dict(targets_raw: list) -> List[TargetInfo]:
    """
    从 targets 列表解析 TargetInfo 列表。

    Args:
        targets_raw: JSON 中的 "targets" 字段值（list）

    Returns:
        TargetInfo 列表
    """
    targets: List[TargetInfo] = []
    for t_cfg in targets_raw:
        pos_cov = np.array(
            t_cfg.get("pos_cov", [[0.1, 0], [0, 0.1]]), dtype=np.float64
        )
        velocity = np.array(t_cfg.get("velocity", [0.0, 0.0]), dtype=np.float64)
        targets.append(TargetInfo(
            tid=t_cfg["tid"],
            type=t_cfg["type"],
            pos_est=np.array(t_cfg["pos"], dtype=np.float64),
            pos_cov=pos_cov,
            velocity_est=velocity,
            confirmed=t_cfg.get("confirmed", True),
            alive=t_cfg.get("alive", True),
            value=float(t_cfg.get("value", 0.8)),
            threat=float(t_cfg.get("threat", 0.5)),
        ))
    return targets


def _parse_sensor_params_from_dict(sp_raw: Optional[list]) -> List[SensorParams]:
    """
    从 sensor_params 列表解析 SensorParams，若为 None 则返回默认三传感器参数。

    Args:
        sp_raw: JSON 中的 "sensor_params" 字段值，或 None

    Returns:
        SensorParams 列表
    """
    if sp_raw is None:
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


# ── JSON 场景加载器 ────────────────────────────────────

def load_snapshot_from_json(filepath: str) -> SituationSnapshot:
    """
    从 JSON 文件加载态势快照，适合多场景批量测试。

    JSON 格式示例::

        {
          "scenario_name": "默认场景",
          "aoi": {"row": 3, "col": 4},
          "staging_position": [150, -50],
          "commander_AOI": ["A_3_4"],
          "grid_weather": {"c0": 0.2, "c1": 0.15, ...},
          "platforms": {
            "UAV": {"count": 5, "sensors": ["EO","SAR","ESM"],
                    "munitions": {"HF":0,"RKT":0,"GUN":0}},
            "HELI": {"count": 2, "sensors": ["MMW","EOIR"],
                     "munitions": {"HF":16,"RKT":76,"GUN":1200}}
          },
          "targets": [
            {"tid":"g1","type":"RADAR","pos":[270,260],"value":1.0,"threat":0.9}
          ],
          "sensor_params": [...]  // 可选，有默认值
        }

    Parameters
    ----------
    filepath : str
        JSON 场景文件路径

    Returns
    -------
    SituationSnapshot
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── 栅格 ──
    aoi = data["aoi"]
    grids = generate_aoi_grids(aoi_row=aoi["row"], aoi_col=aoi["col"])

    weather_overrides = data.get("grid_weather", {})
    for cell_id, w in weather_overrides.items():
        for g in grids:
            if g.cell_id == cell_id:
                g.weather_w = w
                break

    terrain_overrides = data.get("grid_terrain", {})
    for cell_id, tl in terrain_overrides.items():
        for g in grids:
            if g.cell_id == cell_id:
                g.terrain_level = int(tl)
                break

    # ── 集结区 ──
    staging = np.array(data["staging_position"], dtype=np.float64)

    # ── 平台 ──
    platforms = _parse_platforms_from_dict(data["platforms"], data["staging_position"])

    # ── 目标 ──
    targets = _parse_targets_from_dict(data.get("targets", []))

    # ── 传感器参数 ──
    sensor_params = _parse_sensor_params_from_dict(data.get("sensor_params"))

    # ── LOS / Occlusion ──
    N_H = sum(1 for p in platforms if p.type == "HELI")
    N_G = len(targets)
    los = np.ones((N_H, max(N_G, 1)))
    occ = np.ones((N_H, max(N_G, 1)))

    commander_aoi = data.get("commander_AOI", [])

    return SituationSnapshot(
        cycle_id=0,
        timestamp=0.0,
        grids=grids,
        targets=targets,
        platforms=platforms,
        sensor_params=sensor_params,
        commander_AOI=commander_aoi,
        staging_position=staging,
        los_matrix=los if N_G > 0 else None,
        occlusion_matrix=occ if N_G > 0 else None,
    )


def load_all_scenarios(directory: str = "scenarios") -> List[SituationSnapshot]:
    """
    加载指定目录下所有 JSON 场景文件。

    Parameters
    ----------
    directory : str
        场景文件目录路径（相对于工作目录或绝对路径）

    Returns
    -------
    List[SituationSnapshot]
    """
    dir_path = Path(directory)
    if not dir_path.is_absolute():
        dir_path = Path.cwd() / dir_path
    if not dir_path.is_dir():
        raise FileNotFoundError(f"场景目录不存在: {dir_path}")

    snapshots = []
    for fpath in sorted(dir_path.glob("*.json")):
        # 跳过非单 AOI 格式文件（如 multi_aoi_example.json）
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "aoi" not in data:
            continue
        snap = load_snapshot_from_json(str(fpath))
        snapshots.append(snap)
    return snapshots


# ── 执行层接口：精简任务清单 ────────────────────────────

def make_execution_order(plan: AllocationPlan, snapshot: SituationSnapshot) -> dict:
    """
    将求解结果转为下游执行单元所需的精简任务清单。

    每条任务对应一个平台的一次行动，执行单元遍历 tasks 即可。

    Parameters
    ----------
    plan : AllocationPlan
        求解器输出的完整分配方案
    snapshot : SituationSnapshot
        当前态势快照（用于补充 AOI、时间戳等上下文）

    Returns
    -------
    dict
        {
            "cycle_id": int,
            "timestamp": float,
            "status": str,
            "tasks": [
                {
                    "platform": str,
                    "task_type": "recon" | "strike",
                    "aoi": str,            // 仅 recon
                    "cell": str,           // 仅 recon
                    "sensor": str,         // 仅 recon
                    "target": str,         // 仅 strike
                    "munition": str,       // 仅 strike
                    "qty": int,            // 仅 strike
                    "role": str
                }
            ]
        }
    """
    aoi = snapshot.commander_AOI[0] if snapshot.commander_AOI else ""

    tasks = []
    for ra in plan.recon_assignments:
        tasks.append({
            "platform": ra.pid,
            "task_type": "recon",
            "aoi": aoi,
            "sensors_mounted": ra.sensors_mounted,
            "sensor_used": ra.sensor_used,
            "cell": ra.cell,
            "role": ra.role,
        })

    for sa in plan.strike_assignments:
        tasks.append({
            "platform": sa.pid,
            "task_type": "strike",
            "aoi": aoi,
            "sensors_mounted": plan.mounted_sensors.get(sa.pid, []),
            "target": sa.target,
            "munition": sa.munition,
            "qty": sa.qty,
            "role": sa.role,
        })

    return {
        "cycle_id": plan.cycle_id,
        "timestamp": snapshot.timestamp,
        "status": plan.status,
        "tasks": tasks,
    }


def save_execution_order(plan: AllocationPlan, snapshot: SituationSnapshot,
                         filepath: str, indent: int = 2) -> None:
    """
    将 make_execution_order 的结果写入 JSON 文件。

    Parameters
    ----------
    plan : AllocationPlan
    snapshot : SituationSnapshot
    filepath : str
        输出 JSON 文件路径
    indent : int
        JSON 缩进空格数
    """
    import os
    out_dir = os.path.dirname(filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(make_execution_order(plan, snapshot), f,
                  ensure_ascii=False, indent=indent)


# ── 一站式求解函数 ─────────────────────────────────────

def solve(
    input_data,
    solver: str = "cbc",
    time_limit_s: float = 3.0,
    verbose: int = 0,
) -> ExecutionOrder:
    """
    一站式求解函数 —— 态势理解模块（大脑）调用的唯一入口。

    输入态势数据，自动完成：校验 → 转换 → MILP 求解 → 执行清单构建。

    Parameters
    ----------
    input_data : dict | SituationInput
        支持两种形式:

        1. dict（推荐，与 JSON 场景格式兼容）:
           {
               "aoi": {"row": 3, "col": 4},
               "targets": [
                   {"tid": "g1", "type": "RADAR", "pos": [162, 112],
                    "value": 0.97, "threat": 0.91},
               ],
               "platforms_uav": 5,          # 可选，默认 5
               "platforms_heli": 2,          # 可选，默认 2
               "staging_position": [150, -50],  # 可选
               "grid_weather": {"c3": 0.85},    # 可选
               "grid_terrain": {"c2": 1},        # 可选
               "commander_AOI": ["A_3_4"],       # 可选
           }

        2. SituationInput 对象（完全控制）:
           直接传入已构造好的 SituationInput dataclass 实例。

    solver : str
        求解器名称: "cbc"(默认) / "gurobi" / "ortools" / "highs"
    time_limit_s : float
        单次求解时间上限 (秒)，默认 3.0
    verbose : int
        终端输出级别: 0=静默, 1=摘要, 2=详细

    Returns
    -------
    ExecutionOrder
        含扁平任务列表的执行清单:
        - eo.tasks → List[ReconTask | StrikeTask]
        - eo.summary() → 单行摘要
        - eo.validate() → 校验错误列表

    Raises
    ------
    ValueError
        输入校验失败时抛出，包含所有错误信息

    Example
    -------
        >>> from task_interface import solve

        >>> # 最简调用
        >>> result = solve({
        ...     "aoi": {"row": 3, "col": 4},
        ...     "targets": [
        ...         {"tid": "g1", "type": "RADAR", "pos": [162, 112],
        ...          "value": 0.97, "threat": 0.91},
        ...     ],
        ... })

        >>> # 遍历执行
        >>> for task in result.tasks:
        ...     if task.task_type == "recon":
        ...         print(f"{task.platform}: {task.sensor_used} → {task.cell}")
        ...     else:
        ...         print(f"{task.platform}: {task.munition}×{task.qty} → {task.target}")

        >>> # 获取 JSON 字符串（跨进程传输）
        >>> json_str = result.to_json()
    """
    # ── Step 1: 输入解析 ──
    if isinstance(input_data, SituationInput):
        si = input_data
    elif isinstance(input_data, dict):
        data = input_data
        # 提取 aoi 信息
        if "aoi" in data:
            aoi_row = data["aoi"]["row"]
            aoi_col = data["aoi"]["col"]
        else:
            aoi_row = data.get("aoi_row", 3)
            aoi_col = data.get("aoi_col", 4)

        staging = tuple(data.get("staging_position", (150.0, -50.0)))

        # 解析 platforms：支持列表格式和旧的 count 格式
        platforms_raw = data.get("platforms", None)
        if isinstance(platforms_raw, list):
            # 新格式：逐架列出 [{"pid":"U1","type":"UAV",...}, ...]
            from situation_input_interface import build_platform
            _platform_objs = []
            for p in platforms_raw:
                _platform_objs.append(build_platform(
                    pid=p["pid"], type_=p["type"],
                    pos=tuple(p["pos"]) if isinstance(p["pos"], list) else p["pos"],
                    sensors_mounted=p.get("sensors_mounted") or p.get("sensors"),
                    munitions=p.get("munitions"),
                    alt=p.get("alt"),
                    lost=p.get("lost", False),
                ))
            si = build_situation_input(
                aoi_row=aoi_row,
                aoi_col=aoi_col,
                targets=data.get("targets", []),
                platforms_uav=0,   # 下面用 custom_platforms 覆盖
                platforms_heli=0,
                staging_position=staging,
                weather_overrides=data.get("grid_weather"),
                terrain_overrides=data.get("grid_terrain"),
                commander_aoi=data.get("commander_AOI", data.get("commander_aoi")),
            )
            si.platforms = _platform_objs
        else:
            # 旧格式：platforms_uav / platforms_heli 整数字段
            si = build_situation_input(
                aoi_row=aoi_row,
                aoi_col=aoi_col,
                targets=data.get("targets", []),
                platforms_uav=data.get("platforms_uav", 5),
                platforms_heli=data.get("platforms_heli", 2),
                staging_position=staging,
                weather_overrides=data.get("grid_weather"),
                terrain_overrides=data.get("grid_terrain"),
                commander_aoi=data.get("commander_AOI", data.get("commander_aoi")),
            )
    else:
        raise TypeError(
            f"input_data 应为 dict 或 SituationInput，收到 {type(input_data).__name__}"
        )

    # ── Step 2: 校验 ──
    errors = validate_situation_input(si)
    if errors:
        raise ValueError("态势输入校验失败:\n  " + "\n  ".join(errors))

    # ── Step 3: 转换 → 求解 ──
    snap = _convert_to_snapshot(si)
    allocator = TaskAllocator(
        solver=solver, time_limit_s=time_limit_s, verbose=verbose,
    )
    plan = allocator.solve(snap)

    # ── Step 4: 构建执行清单 ──
    return _build_execution_order(plan, snap)


# ── 文件输入输出式调用 ─────────────────────────────────

def solve_file(
    input_path: str,
    output_path: Optional[str] = None,
    solver: str = "cbc",
    time_limit_s: float = 3.0,
    verbose: int = 0,
) -> ExecutionOrder:
    """
    从 JSON 文件读取态势输入，求解，可选写入 JSON 输出文件。

    这是最简单的调用方式：大脑只需准备好输入 JSON 文件路径即可。

    Parameters
    ----------
    input_path : str
        输入 JSON 文件路径，格式参考 templates/input_template.json
    output_path : str, optional
        输出 JSON 文件路径。不填则仅返回 ExecutionOrder 对象，不写文件。
    solver : str
        求解器名称: cbc / gurobi / ortools / highs
    time_limit_s : float
        单次求解时间上限 (秒)
    verbose : int
        0=静默, 1=摘要, 2=详细

    Returns
    -------
    ExecutionOrder

    Example
    -------
        >>> from task_interface import solve_file

        >>> # 最简用法：指定输入文件，自动写入输出文件
        >>> result = solve_file("input.json", "output.json")

        >>> # 仅读输入，不写文件
        >>> result = solve_file("input.json")

        >>> # 遍历结果
        >>> for task in result.tasks:
        ...     print(task.platform, task.task_type)
    """
    # ── 读取输入 JSON ──
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── 解析 platforms 字段（仅对旧的 dict 格式做 count 提取）──
    if isinstance(data.get("platforms"), dict) and "platforms_uav" not in data:
        data["platforms_uav"] = data["platforms"].get("UAV", {}).get("count", 5)
    if isinstance(data.get("platforms"), dict) and "platforms_heli" not in data:
        data["platforms_heli"] = data["platforms"].get("HELI", {}).get("count", 2)

    # ── 调用核心求解 ──
    result = solve(
        data,
        solver=solver,
        time_limit_s=time_limit_s,
        verbose=verbose,
    )

    # ── 写入输出 JSON ──
    if output_path:
        import os
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(execution_order_to_json(result))

    return result
