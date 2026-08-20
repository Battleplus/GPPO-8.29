"""
态势输入接口 —— 态势理解模块 → 任务分配模块 的标准化数据契约。

================================================================================
使用方：态势理解模块（大脑）的开发者
作用：  按此接口格式产出态势快照，传给任务分配模块求解
================================================================================

=== 快速模板 ===

    from situation_input_interface import (
        SituationInput, GridSpec, TargetSpec, PlatformSpec, SensorSpec,
        build_situation_input, validate_situation_input, to_json, from_json,
    )

    # 方式 A：工厂函数构建（推荐）
    si = build_situation_input(
        aoi_row=3, aoi_col=4,
        targets=[
            {"tid": "g1", "type": "RADAR", "pos": [162, 112],
             "confirmed": True, "value": 0.97, "threat": 0.91},
        ],
        platforms_uav=5,
        platforms_heli=2,
    )

    # 方式 B：手动逐字段构造（完全控制）
    si = SituationInput(
        aoi_id="A_3_4",
        grids=[
            GridSpec(cell_id="c0", center=(175, 125), size=(50, 50), weather_w=0.2),
            GridSpec(cell_id="c1", center=(162, 112), size=(25, 25), weather_w=0.15),
            ...
        ],
        targets=[...],
        platforms=[...],
        sensors=[...],
        staging_position=(150.0, -50.0),
    )

    # 校验（返回错误列表，空列表 = 通过）
    errors = validate_situation_input(si)
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")

    # 序列化（跨进程/跨语言传输）
    json_str = to_json(si)
    si2 = from_json(json_str)

    # 传入任务分配模块
    from task_interface import TaskAllocator, situation_input_to_snapshot
    snap = situation_input_to_snapshot(si)
    plan = TaskAllocator().solve(snap)

================================================================================
字段速查表
================================================================================

SituationInput（顶层容器）
├── aoi_id: str                          AOI 标识，如 "A_3_4"
├── staging_position: (float, float)     集结区坐标 (km)，默认 (150, -50)
├── grids: List[GridSpec]                5 个栅格 [c0, c1, c2, c3, c4]
├── targets: List[TargetSpec]            目标列表（仅 confirmed+alive 参与 MILP）
├── platforms: List[PlatformSpec]        平台列表
├── sensors: List[SensorSpec]            传感器参数（默认 EO/SAR/ESM）
├── commander_aoi: List[str]             指挥官关注 AOI（可选）
├── los_matrix: List[List[float]]        直升机-目标通视矩阵（可选, 默认全 1）
└── occlusion_matrix: List[List[float]]  遮挡衰减矩阵（可选, 默认全 1）

GridSpec（单个栅格）
├── cell_id: str                         标识: c0(巡逻区), c1~c4(子区)
├── center: (float, float)               中心坐标 (km)
├── size: (float, float)                 宽×高 (km)，c0 为 (50,50)，c1~c4 为 (25,25)
├── weather_w: float [0,1]              天气系数，≥0.80 时 EO 禁用
├── terrain_level: int {0,1,2}          0=平原, 1=丘陵, 2=山地
├── target_prior: float [0,1]           目标存在先验概率
└── covered: bool                        是否已被侦察覆盖

TargetSpec（单个目标）
├── tid: str                             唯一标识
├── type: str                            "RADAR" | "CP" | "AV"
├── pos: (float, float)                  位置估计 (km)
├── confirmed: bool                      是否已确认（False 不参与打击）
├── alive: bool                          是否存活（False 不参与打击）
├── value: float [0,1]                   归一化作戰价值
├── threat: float [0,1]                  威胁度
├── pos_cov: [[float*2]*2]              位置协方差（可选, 默认 diag(0.1)）
└── velocity: (float, float)             速度估计 (km/tick)（可选, 默认 (0,0)）

PlatformSpec（单个平台）
├── pid: str                             唯一标识，如 "U1", "H1"
├── type: str                            "UAV" | "HELI"
├── pos: (float, float)                  当前位置 (km)
├── sensors_mounted: List[str]           空/null=算法自动分配; 非空=锁定指定传感器
├── munitions: Dict[str, int]            UAV 全 0; HELI: {"HF":16,"RKT":76,"GUN":1200}
├── alt: float                           高度 (km)，UAV=2.0, HELI=3.0
└── lost: bool                           战损标志

SensorSpec（传感器参数）
├── name: str                            "EO" | "SAR" | "ESM" | "MMW" | "EOIR"
├── P0: float [0,1]                     理想条件基础探测概率
├── R: float >0                         最大作用距离 (km)
└── weather_sensitive: bool              是否受天气衰减影响
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Union
import numpy as np


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class GridSpec:
    """栅格/子区规格。

    约束:
      - cell_id 必须唯一，约定 c0=巡逻区, c1~c4=子区
      - c0 的 size 为 (50, 50)，c1~c4 的 size 为 (25, 25)
      - weather_w ∈ [0, 1]
      - terrain_level ∈ {0, 1, 2}
      - target_prior ∈ [0, 1]
    """
    cell_id: str
    center: Tuple[float, float]
    size: Tuple[float, float] = (25.0, 25.0)
    weather_w: float = 0.2
    terrain_level: int = 0
    target_prior: float = 0.25
    covered: bool = False


@dataclass
class TargetSpec:
    """目标规格。

    约束:
      - type 必须在 {"RADAR", "CP", "AV"} 中
      - value, threat ∈ [0, 1]
      - pos_cov 为 2×2 协方差矩阵（list of list）
      - velocity 为 [vx, vy] km/tick
      - 仅 confirmed=True AND alive=True 的目标参与 MILP 优化
    """
    tid: str
    type: str
    pos: Tuple[float, float]
    confirmed: bool = True
    alive: bool = True
    value: float = 0.8
    threat: float = 0.5
    pos_cov: List[List[float]] = field(default_factory=lambda: [[0.1, 0.0], [0.0, 0.1]])
    velocity: Tuple[float, float] = (0.0, 0.0)


@dataclass
class PlatformSpec:
    """平台规格。

    约束:
      - type 必须在 {"UAV", "HELI"} 中
      - sensors_mounted 为空列表或 null 时由算法自动分配传感器组合
      - sensors_mounted 非空时锁定，算法不可更改
      - UAV alt=2.0, HELI alt=3.0
    """
    pid: str
    type: str
    pos: Tuple[float, float]
    sensors_mounted: Optional[List[str]] = None
    munitions: Optional[Dict[str, int]] = None
    alt: float = 2.0
    lost: bool = False


@dataclass
class SensorSpec:
    """传感器固有参数规格。

    约束:
      - name 在 {"EO", "SAR", "ESM", "MMW", "EOIR"} 中
      - P0 ∈ [0, 1]
      - R > 0
    """
    name: str
    P0: float = 0.85
    R: float = 50.0
    weather_sensitive: bool = True


@dataclass
class SituationInput:
    """态势输入 —— 态势理解模块传给任务分配模块的完整数据契约。

    顶层容器，包含 AOI 标识、集结区、栅格、目标、平台、传感器参数。
    可选字段（los_matrix, occlusion_matrix）有合理默认值。

    维度约定:
      - grids: 固定 5 个（c0 + c1~c4），由 build_situation_input 自动生成
      - platforms: UAV 数量 1~10，HELI 数量 0~6
      - targets: 数量不限，超出平台处理能力的目标会被自然裁剪
      - sensors: 默认 EO/SAR/ESM 三传感器
    """
    aoi_id: str
    grids: List[GridSpec]
    targets: List[TargetSpec]
    platforms: List[PlatformSpec]
    sensors: List[SensorSpec] = field(default_factory=lambda: [
        SensorSpec(name="EO",  P0=0.85, R=15.0,  weather_sensitive=True),
        SensorSpec(name="SAR", P0=0.90, R=50.0,  weather_sensitive=False),
        SensorSpec(name="ESM", P0=0.80, R=100.0, weather_sensitive=False),
    ])
    staging_position: Tuple[float, float] = (150.0, -50.0)
    commander_aoi: List[str] = field(default_factory=list)
    los_matrix: Optional[List[List[float]]] = None
    occlusion_matrix: Optional[List[List[float]]] = None


# ============================================================================
# 工厂函数
# ============================================================================

def build_grid(
    cell_id: str,
    center: Tuple[float, float],
    width_km: float = 25.0,
    height_km: float = 25.0,
    weather_w: float = 0.2,
    terrain_level: int = 0,
    target_prior: float = 0.25,
    covered: bool = False,
) -> GridSpec:
    """快捷构造单个栅格规格。"""
    return GridSpec(
        cell_id=cell_id,
        center=center,
        size=(width_km, height_km),
        weather_w=weather_w,
        terrain_level=terrain_level,
        target_prior=target_prior,
        covered=covered,
    )


def build_target(
    tid: str,
    type_: str,
    pos: Tuple[float, float],
    confirmed: bool = True,
    alive: bool = True,
    value: float = 0.8,
    threat: float = 0.5,
    pos_cov: Optional[List[List[float]]] = None,
    velocity: Tuple[float, float] = (0.0, 0.0),
) -> TargetSpec:
    """快捷构造单个目标规格。"""
    return TargetSpec(
        tid=tid,
        type=type_,
        pos=pos,
        confirmed=confirmed,
        alive=alive,
        value=value,
        threat=threat,
        pos_cov=pos_cov or [[0.1, 0.0], [0.0, 0.1]],
        velocity=velocity,
    )


def build_platform(
    pid: str,
    type_: str,
    pos: Tuple[float, float],
    sensors_mounted: Optional[List[str]] = None,
    munitions: Optional[Dict[str, int]] = None,
    alt: Optional[float] = None,
    lost: bool = False,
) -> PlatformSpec:
    """快捷构造单个平台规格。

    sensors_mounted=None 时由算法自动分配传感器组合。
    sensors_mounted 非空时锁定为指定传感器。
    """
    if type_ == "UAV":
        munitions = munitions or {"HF": 0, "RKT": 0, "GUN": 0}
        alt = alt if alt is not None else 2.0
    elif type_ == "HELI":
        munitions = munitions or {"HF": 16, "RKT": 76, "GUN": 1200}
        alt = alt if alt is not None else 3.0
    else:
        raise ValueError(f"type 必须是 'UAV' 或 'HELI'，收到: {type_}")
    return PlatformSpec(
        pid=pid, type=type_, pos=pos,
        sensors_mounted=sensors_mounted, munitions=munitions, alt=alt, lost=lost,
    )


def build_situation_input(
    aoi_row: int,
    aoi_col: int,
    targets: List[Dict],
    platforms_uav: int = 5,
    platforms_heli: int = 2,
    staging_position: Tuple[float, float] = (150.0, -50.0),
    weather_w: float = 0.2,
    weather_overrides: Optional[Dict[str, float]] = None,
    terrain_overrides: Optional[Dict[str, int]] = None,
    target_prior: float = 0.25,
    sensors: Optional[List[SensorSpec]] = None,
    commander_aoi: Optional[List[str]] = None,
) -> SituationInput:
    """一站式构建态势输入（最常用入口）。

    自动生成 AOI 的 5 个栅格（c0 巡逻区 + c1~c4 子区），
    支持按 cell_id 覆盖天气和地形。

    Args:
        aoi_row: AOI 行号 (1..6, y 方向自上而下)
        aoi_col: AOI 列号 (1..6, x 方向自左而右)
        targets: 目标字典列表，每项含 tid/type/pos/value/threat/confirmed/alive
        platforms_uav: UAV 数量 (默认 5)
        platforms_heli: HELI 数量 (默认 2)
        staging_position: 集结区坐标 (km)
        weather_w: 默认天气系数
        weather_overrides: 按 cell_id 覆盖天气，如 {"c3": 0.85}
        terrain_overrides: 按 cell_id 覆盖地形，如 {"c2": 2}
        target_prior: 默认目标先验概率
        sensors: 传感器参数列表 (默认 EO/SAR/ESM)
        commander_aoi: 指挥官关注 AOI

    Returns:
        SituationInput，可直接传入 convert_to_snapshot() 转换后求解

    Example:
        >>> si = build_situation_input(
        ...     aoi_row=3, aoi_col=4,
        ...     targets=[
        ...         {"tid": "g1", "type": "RADAR", "pos": [162, 112],
        ...          "confirmed": True, "value": 0.97, "threat": 0.91},
        ...     ],
        ...     weather_overrides={"c3": 0.85},
        ... )
    """
    if not (1 <= aoi_row <= 6 and 1 <= aoi_col <= 6):
        raise ValueError(f"AOI 行列号需在 1..6，收到 row={aoi_row}, col={aoi_col}")

    weather_overrides = weather_overrides or {}
    terrain_overrides = terrain_overrides or {}

    aoi_id = f"A_{aoi_row}_{aoi_col}"
    x0 = (aoi_col - 1) * 50.0
    y0 = (aoi_row - 1) * 50.0
    cx_aoi = x0 + 25.0
    cy_aoi = y0 + 25.0
    half = 12.5

    # 5 个栅格
    grids = [
        GridSpec(cell_id="c0", center=(cx_aoi, cy_aoi), size=(50.0, 50.0),
                 weather_w=weather_w, terrain_level=0,
                 target_prior=target_prior * 0.4, covered=False),
        GridSpec(cell_id="c1", center=(x0 + half,     y0 + half),      size=(25.0, 25.0),
                 weather_w=weather_w, terrain_level=0,
                 target_prior=target_prior, covered=False),
        GridSpec(cell_id="c2", center=(x0 + 3 * half, y0 + half),      size=(25.0, 25.0),
                 weather_w=weather_w, terrain_level=0,
                 target_prior=target_prior, covered=False),
        GridSpec(cell_id="c3", center=(x0 + half,     y0 + 3 * half),  size=(25.0, 25.0),
                 weather_w=weather_w, terrain_level=0,
                 target_prior=target_prior, covered=False),
        GridSpec(cell_id="c4", center=(x0 + 3 * half, y0 + 3 * half),  size=(25.0, 25.0),
                 weather_w=weather_w, terrain_level=0,
                 target_prior=target_prior, covered=False),
    ]

    # 覆盖天气/地形
    for g in grids:
        if g.cell_id in weather_overrides:
            g.weather_w = float(weather_overrides[g.cell_id])
        if g.cell_id in terrain_overrides:
            g.terrain_level = int(terrain_overrides[g.cell_id])

    # 目标
    target_objs = []
    for t in targets:
        target_objs.append(build_target(
            tid=t["tid"], type_=t["type"],
            pos=tuple(t["pos"]),
            confirmed=t.get("confirmed", True),
            alive=t.get("alive", True),
            value=float(t.get("value", 0.8)),
            threat=float(t.get("threat", 0.5)),
            pos_cov=t.get("pos_cov"),
            velocity=tuple(t.get("velocity", (0.0, 0.0))),
        ))

    # 平台
    platform_objs = []
    for i in range(1, platforms_uav + 1):
        platform_objs.append(build_platform(f"U{i}", "UAV", staging_position))
    for i in range(1, platforms_heli + 1):
        platform_objs.append(build_platform(f"H{i}", "HELI", staging_position))

    return SituationInput(
        aoi_id=aoi_id,
        staging_position=staging_position,
        grids=grids,
        targets=target_objs,
        platforms=platform_objs,
        sensors=sensors or [
            SensorSpec(name="EO",  P0=0.85, R=15.0,  weather_sensitive=True),
            SensorSpec(name="SAR", P0=0.90, R=50.0,  weather_sensitive=False),
            SensorSpec(name="ESM", P0=0.80, R=100.0, weather_sensitive=False),
        ],
        commander_aoi=commander_aoi or [aoi_id],
    )


# ============================================================================
# 校验
# ============================================================================

def validate_situation_input(si: SituationInput) -> List[str]:
    """校验 SituationInput 的完整性与合法性。

    Returns:
        错误信息列表，空列表表示通过。
        调用方应检查返回值，非空时不应传给求解器。

    Example:
        >>> errors = validate_situation_input(si)
        >>> if errors:
        ...     raise ValueError("\\n".join(errors))
    """
    errs = []

    # --- 顶层 ---
    if not si.aoi_id:
        errs.append("aoi_id 为空")
    if not si.grids:
        errs.append("grids 为空，至少需要 5 个栅格 (c0~c4)")
    if not si.platforms:
        errs.append("platforms 为空，至少需要 1 个平台")

    # --- 栅格 ---
    cell_ids = set()
    for g in si.grids:
        if not g.cell_id:
            errs.append(f"GridSpec: cell_id 为空")
            continue
        if g.cell_id in cell_ids:
            errs.append(f"GridSpec: cell_id '{g.cell_id}' 重复")
        cell_ids.add(g.cell_id)
        if not (0.0 <= g.weather_w <= 1.0):
            errs.append(f"GridSpec {g.cell_id}: weather_w={g.weather_w} 不在 [0,1] 范围")
        if g.terrain_level not in (0, 1, 2):
            errs.append(f"GridSpec {g.cell_id}: terrain_level={g.terrain_level} 不在 {{0,1,2}}")
        if not (0.0 <= g.target_prior <= 1.0):
            errs.append(f"GridSpec {g.cell_id}: target_prior={g.target_prior} 不在 [0,1] 范围")
        if g.size[0] <= 0 or g.size[1] <= 0:
            errs.append(f"GridSpec {g.cell_id}: size={g.size} 宽度/高度必须 >0")

    # --- 目标 ---
    valid_types = {"RADAR", "CP", "AV"}
    target_ids = set()
    for t in si.targets:
        if not t.tid:
            errs.append("TargetSpec: tid 为空")
            continue
        if t.tid in target_ids:
            errs.append(f"TargetSpec: tid '{t.tid}' 重复")
        target_ids.add(t.tid)
        if t.type not in valid_types:
            errs.append(f"TargetSpec {t.tid}: type='{t.type}' 不在 {valid_types}")
        if not (0.0 <= t.value <= 1.0):
            errs.append(f"TargetSpec {t.tid}: value={t.value} 不在 [0,1] 范围")
        if not (0.0 <= t.threat <= 1.0):
            errs.append(f"TargetSpec {t.tid}: threat={t.threat} 不在 [0,1] 范围")

    # --- 平台 ---
    valid_ptypes = {"UAV", "HELI"}
    platform_ids = set()
    for p in si.platforms:
        if not p.pid:
            errs.append("PlatformSpec: pid 为空")
            continue
        if p.pid in platform_ids:
            errs.append(f"PlatformSpec: pid '{p.pid}' 重复")
        platform_ids.add(p.pid)
        if p.type not in valid_ptypes:
            errs.append(f"PlatformSpec {p.pid}: type='{p.type}' 不在 {valid_ptypes}")
        # sensors_mounted 为空=算法自动分配，非空=锁定指定传感器
        if p.munitions is None:
            errs.append(f"PlatformSpec {p.pid}: munitions 不能为 None")

    # --- 传感器 ---
    valid_sensors = {"EO", "SAR", "ESM", "MMW", "EOIR"}
    sensor_names = set()
    for s in si.sensors:
        if s.name not in valid_sensors:
            errs.append(f"SensorSpec: name='{s.name}' 不在 {valid_sensors}")
        if s.name in sensor_names:
            errs.append(f"SensorSpec: name='{s.name}' 重复")
        sensor_names.add(s.name)
        if not (0.0 <= s.P0 <= 1.0):
            errs.append(f"SensorSpec {s.name}: P0={s.P0} 不在 [0,1] 范围")
        if s.R <= 0:
            errs.append(f"SensorSpec {s.name}: R={s.R} 必须 >0")

    return errs


# ============================================================================
# JSON 序列化/反序列化
# ============================================================================

class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (GridSpec, TargetSpec, PlatformSpec, SensorSpec, SituationInput)):
            return asdict(obj)
        return super().default(obj)


def to_json(si: SituationInput, indent: int = 2) -> str:
    """将 SituationInput 序列化为 JSON 字符串。

    Example:
        >>> json_str = to_json(si)
        >>> with open("situation.json", "w") as f:
        ...     f.write(json_str)
    """
    return json.dumps(asdict(si), cls=_Encoder, ensure_ascii=False, indent=indent)


def from_json(data: Union[str, Dict]) -> SituationInput:
    """从 JSON 字符串或字典反序列化为 SituationInput。

    Example:
        >>> si = from_json('{"aoi_id": "A_3_4", ...}')
        >>> si = from_json({"aoi_id": "A_3_4", ...})
    """
    if isinstance(data, str):
        data = json.loads(data)

    grids = [GridSpec(**g) for g in data["grids"]]
    targets = [TargetSpec(**t) for t in data.get("targets", [])]
    platforms = [
        PlatformSpec(
            pid=p["pid"], type=p["type"], pos=tuple(p["pos"]),
            sensors_mounted=p.get("sensors_mounted") or p.get("sensors"),
            munitions=p.get("munitions"),
            alt=p.get("alt", 2.0), lost=p.get("lost", False),
        )
        for p in data.get("platforms", [])
    ]
    sensors = [SensorSpec(**s) for s in data.get("sensors", [])]
    if not sensors:
        sensors = [
            SensorSpec(name="EO",  P0=0.85, R=15.0,  weather_sensitive=True),
            SensorSpec(name="SAR", P0=0.90, R=50.0,  weather_sensitive=False),
            SensorSpec(name="ESM", P0=0.80, R=100.0, weather_sensitive=False),
        ]

    staging = tuple(data.get("staging_position", (150.0, -50.0)))
    commander = data.get("commander_aoi", data.get("commander_AOI", []))

    return SituationInput(
        aoi_id=data["aoi_id"],
        grids=grids,
        targets=targets,
        platforms=platforms,
        sensors=sensors,
        staging_position=staging,
        commander_aoi=commander,
    )


# ============================================================================
# 转换函数：SituationInput → 任务分配模块内部的 SituationSnapshot
# ============================================================================

def convert_to_snapshot(si: SituationInput, cycle_id: int = 0, timestamp: float = 0.0):
    """将标准化 SituationInput 转为任务分配模块内部使用的 SituationSnapshot。

    其他模块开发者无需关心此函数内部实现，只需知道：
    - 入参：SituationInput（标准化态势输入）
    - 出参：可直接传给 TaskAllocator.solve() 的对象

    Example:
        >>> si = build_situation_input(aoi_row=3, aoi_col=4, targets=[...])
        >>> errors = validate_situation_input(si)
        >>> if not errors:
        ...     snap = convert_to_snapshot(si)
        ...     plan = TaskAllocator().solve(snap)
    """
    from core.snapshot import (
        GridInfo, TargetInfo, PlatformInfo, SensorParams, SituationSnapshot,
    )

    grids = [
        GridInfo(
            cell_id=g.cell_id,
            center=np.array(g.center, dtype=np.float64),
            width_km=g.size[0],
            height_km=g.size[1],
            weather_w=g.weather_w,
            terrain_level=g.terrain_level,
            target_prior=g.target_prior,
            covered=g.covered,
        )
        for g in si.grids
    ]

    targets = [
        TargetInfo(
            tid=t.tid, type=t.type,
            pos_est=np.array(t.pos, dtype=np.float64),
            pos_cov=np.array(t.pos_cov, dtype=np.float64),
            velocity_est=np.array(t.velocity, dtype=np.float64),
            confirmed=t.confirmed, alive=t.alive,
            value=t.value, threat=t.threat,
        )
        for t in si.targets
    ]

    platforms = [
        PlatformInfo(
            pid=p.pid, type=p.type,
            pos=np.array(p.pos, dtype=np.float64),
            alt=p.alt, lost=p.lost,
            sensors_mounted=list(p.sensors_mounted) if p.sensors_mounted else [],
            munitions=dict(p.munitions),
        )
        for p in si.platforms
    ]

    sensor_params = [
        SensorParams(
            name=s.name, P0=s.P0, R=s.R,
            weather_sensitive=s.weather_sensitive,
        )
        for s in si.sensors
    ]

    n_heli = sum(1 for p in si.platforms if p.type == "HELI")
    n_tgt = len(targets)

    los = None
    occ = None
    if si.los_matrix is not None and n_heli > 0 and n_tgt > 0:
        los = np.array(si.los_matrix, dtype=np.float64)
    elif n_heli > 0 and n_tgt > 0:
        los = np.ones((n_heli, n_tgt))

    if si.occlusion_matrix is not None and n_heli > 0 and n_tgt > 0:
        occ = np.array(si.occlusion_matrix, dtype=np.float64)
    elif n_heli > 0 and n_tgt > 0:
        occ = np.ones((n_heli, n_tgt))

    return SituationSnapshot(
        cycle_id=cycle_id,
        timestamp=timestamp,
        grids=grids,
        targets=targets,
        platforms=platforms,
        sensor_params=sensor_params,
        commander_AOI=si.commander_aoi,
        staging_position=np.array(si.staging_position, dtype=np.float64),
        los_matrix=los,
        occlusion_matrix=occ,
    )
