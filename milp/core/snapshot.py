"""
态势快照数据结构 —— 任务分配模块的标准化输入接口。

职责:
  1. 定义 GridInfo / TargetInfo / PlatformInfo / SensorParams 数据类
  2. 定义 SituationSnapshot 顶层容器，桥接态势理解与任务分配模块
  3. 提供辅助查询方法（活跃目标、平台按类型筛选等）

设计原则:
  - 任务分配模块只消费此快照，不感知环境内部实现
  - 两个模块通过 SituationSnapshot / AllocationPlan 数据契约隔离

对外接口:
  - GridInfo           — 栅格信息
  - TargetInfo         — 目标信息
  - PlatformInfo       — 平台信息
  - SensorParams       — 传感器固有参数
  - SituationSnapshot  — 态势快照顶层容器

参考:
  设计方案 §3.1; 实施方案 §2.3
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class GridInfo:
    """栅格/子区信息，对应战场划分后的一个侦察单元。"""
    cell_id: str            # 栅格标识，如 "c1", "c0"（巡逻区）
    center: np.ndarray      # shape (2,) 栅格中心 [x, y] km
    width_km: float         # 栅格宽度 (km)
    height_km: float        # 栅格高度 (km)
    weather_w: float        # 天气系数 w_c ∈ [0, 1]，越高越不利
    terrain_level: int      # 地形等级: 0=平原, 1=丘陵, 2=山地
    target_prior: float     # 目标先验概率 ρ_c
    covered: bool           # 是否已被有效侦察覆盖


@dataclass
class TargetInfo:
    """动态目标信息，由传感器观测确认后纳入候选目标集 G。"""
    tid: str                # 目标唯一标识，如 "g1"
    type: str               # 目标类型: RADAR / CP / AV
    pos_est: np.ndarray     # shape (2,) 位置估计 [x, y] km
    pos_cov: np.ndarray     # shape (2,2) 位置协方差矩阵
    velocity_est: np.ndarray # shape (2,) 速度估计 [vx, vy] km/tick
    confirmed: bool         # 是否已被对应传感器确认（先侦后打门控）
    alive: bool             # 目标存活状态
    value: float            # 归一化作战价值 val_g ∈ [0, 1]
    threat: float           # 反击威胁强度 thr_g ∈ [0, 1]


@dataclass
class PlatformInfo:
    """平台（无人机/直升机）状态快照。"""
    pid: str                # 平台唯一标识，如 "U1", "H1"
    type: str               # 平台类型: UAV / HELI
    pos: np.ndarray         # shape (2,) 当前位置 [x, y] km
    alt: float              # 当前高度 (km)，UAV=2.0, HELI=3.0
    lost: bool              # 战损/故障标志
    sensors_mounted: List[str]   # 搭载传感器列表，空=算法自动分配
    munitions: dict         # 弹药余量 {"HF": n, "RKT": m, "GUN": k}


@dataclass
class SensorParams:
    """传感器固有物理参数，由环境模块提供，任务分配模块据此计算探测概率。"""
    name: str               # 传感器名称: EO / SAR / ESM / MMW / IR
    P0: float               # 理想条件下基础探测概率
    R: float                # 最大作用距离 (km)
    weather_sensitive: bool # 是否受天气衰减影响


@dataclass
class SituationSnapshot:
    """
    态势快照 —— 任务分配模块唯一输入。

    数据结构包含四类信息:
        - grids:    栅格/子区列表（含天气、先验概率）
        - targets:  动态候选目标集 G（仅含 alive 且 confirmed 的目标参与优化）
        - platforms: 可用平台状态
        - sensor_params: 传感器固有参数表

    可选字段:
        - los_matrix:       通视矩阵 V_{p,g} ∈ {0,1}
        - occlusion_matrix: 遮挡衰减 η_{p,g} ∈ [0,1]

    参考:
        设计方案 §3.1
    """
    cycle_id: int                           # 当前分配轮次
    timestamp: float                        # 仿真时间戳
    grids: List[GridInfo]                   # 栅格列表
    targets: List[TargetInfo]               # 目标列表（含未确认/已摧毁）
    platforms: List[PlatformInfo]           # 平台列表
    sensor_params: List[SensorParams]       # 传感器参数表
    commander_AOI: List[str]                # 指挥官选定的 AOI
    staging_position: Optional[np.ndarray] = None # (2,) 集结区坐标 (km)，平台出发位置
    los_matrix: Optional[np.ndarray] = None      # V_{p,g} ∈ {0,1}
    occlusion_matrix: Optional[np.ndarray] = None # η_{p,g} ∈ [0,1]

    def get_active_targets(self) -> List[TargetInfo]:
        """返回存活且已确认的目标（即参与 MILP 优化的候选集 G）。"""
        return [t for t in self.targets if t.alive and t.confirmed]

    def get_platform_by_type(self, ptype: str) -> List[PlatformInfo]:
        """按平台类型筛选（"UAV" / "HELI"）。"""
        return [p for p in self.platforms if p.type == ptype]

    def get_uav_platforms(self) -> List[PlatformInfo]:
        """返回所有未战损的 UAV 平台。"""
        return [p for p in self.platforms if p.type == "UAV" and not p.lost]

    def get_heli_platforms(self) -> List[PlatformInfo]:
        """返回所有未战损的 HELI 平台。"""
        return [p for p in self.platforms if p.type == "HELI" and not p.lost]


def generate_aoi_grids(aoi_row: int, aoi_col: int,
                       weather_w: float = 0.2,
                       target_prior: float = 0.25,
                       terrain_levels: dict = None) -> List[GridInfo]:
    """
    按 AOI 编号生成该区域的全部栅格 (c0 巡逻区 + c1~c4 子区)。

    300×300 km 任务区被划分为 6×6=36 个 AOI，每个 50×50 km。
    每个 AOI 内含 4 个 25×25 km 侦察子区。

    Args:
        aoi_row: AOI 行号 (1..6, y 方向自上而下)
        aoi_col: AOI 列号 (1..6, x 方向自左而右)
        weather_w: 天气系数默认值
        target_prior: 目标先验概率默认值

    Returns:
        [c0, c1, c2, c3, c4] 五个 GridInfo
    """
    if not (1 <= aoi_row <= 6 and 1 <= aoi_col <= 6):
        raise ValueError(f"AOI 行列号需在 1..6 之间，收到 row={aoi_row}, col={aoi_col}")

    if terrain_levels is None:
        terrain_levels = {}

    x0 = (aoi_col - 1) * 50.0   # AOI 左边界 x
    y0 = (aoi_row - 1) * 50.0   # AOI 上边界 y
    cx_aoi = x0 + 25.0          # AOI 中心 x
    cy_aoi = y0 + 25.0          # AOI 中心 y

    sub_half = 12.5  # 子区半边长

    return [
        # c0: 巡逻区，覆盖整个 AOI
        GridInfo(
            cell_id="c0",
            center=np.array([cx_aoi, cy_aoi]),
            width_km=50.0, height_km=50.0,
            weather_w=weather_w, terrain_level=terrain_levels.get("c0", 0),
            target_prior=target_prior * 0.4, covered=False,
        ),
        # c1: NW 子区
        GridInfo(
            cell_id="c1",
            center=np.array([x0 + sub_half, y0 + sub_half]),
            width_km=25.0, height_km=25.0,
            weather_w=weather_w, terrain_level=terrain_levels.get("c1", 0),
            target_prior=target_prior, covered=False,
        ),
        # c2: NE 子区
        GridInfo(
            cell_id="c2",
            center=np.array([x0 + 3 * sub_half, y0 + sub_half]),
            width_km=25.0, height_km=25.0,
            weather_w=weather_w, terrain_level=terrain_levels.get("c2", 0),
            target_prior=target_prior, covered=False,
        ),
        # c3: SW 子区
        GridInfo(
            cell_id="c3",
            center=np.array([x0 + sub_half, y0 + 3 * sub_half]),
            width_km=25.0, height_km=25.0,
            weather_w=weather_w, terrain_level=terrain_levels.get("c3", 0),
            target_prior=target_prior, covered=False,
        ),
        # c4: SE 子区
        GridInfo(
            cell_id="c4",
            center=np.array([x0 + 3 * sub_half, y0 + 3 * sub_half]),
            width_km=25.0, height_km=25.0,
            weather_w=weather_w, terrain_level=terrain_levels.get("c4", 0),
            target_prior=target_prior, covered=False,
        ),
    ]
