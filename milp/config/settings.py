"""
全局配置中心 —— 战场参数、平台数量、求解器选择、目标函数权重等所有可调参数。

职责:
  1. 定义 SolverType 枚举（GUROBI / CBC / ORTOOLS / HIGHS）
  2. 通过 GlobalSettings dataclass 集中管理全部配置项
  3. 提供 validate_solver() 校验求解器白名单

对外接口:
  - SolverType   — 求解器类型枚举
  - GlobalSettings — 全局配置 dataclass

使用示例:
    >>> from config.settings import GlobalSettings, SolverType
    >>> s = GlobalSettings(active_solver=SolverType.CBC, verbose=1)
    >>> s.validate_solver()
"""

from dataclasses import dataclass, field
from typing import List
from enum import Enum


class SolverType(Enum):
    """
    求解器类型枚举。

    值:
        GUROBI:  Gurobi 商业求解器（学术免费）
        CBC:     COIN-OR CBC 开源求解器（python-mip 后端）
        ORTOOLS: Google OR-Tools CP-SAT（Apache-2.0 开源）
        HIGHS:   HiGHS 开源求解器
    """
    GUROBI = "gurobi"
    CBC = "cbc"
    ORTOOLS = "ortools"
    HIGHS = "highs"


@dataclass
class GlobalSettings:
    """
    全局配置参数。

    属性:
        map_size_km: 战场边长 (km), 默认 300
        grid_size_km: 指挥级 AOI 边长 (km), 默认 50, 6×6=36 个
        subgrid_size_km: 侦察子区边长 (km), 默认 25, 每个 AOI 分 4 个
        num_uav: 彩虹-4 无人机数量, 默认 5
        num_heli: AH-64E 武装直升机数量, 默认 2
        total_platforms: 平台总数, 默认 7
        num_radar: 雷达目标数, 默认 1
        num_cp: 指挥所目标数, 默认 1
        num_av: 装甲车目标数, 默认 2
        active_solver: 当前激活的求解器（切换此字段即切换求解器）
        enabled_solvers: 求解器白名单
        verbose: 终端显示级别 (0=静默, 1=摘要, 2=详细), 默认 1

    参考:
        设计方案 §2, §5; 实施方案 §2.1
    """
    # ---------- 战场 ----------
    map_size_km: float = 300.0          # 战场边长 (km)
    grid_size_km: int = 50              # 指挥级 AOI 边长 (km)，6×6=36 个
    subgrid_size_km: int = 25           # 侦察子区边长 (km)，每个 AOI 分 4 个

    # ---------- 平台 ----------
    num_uav: int = 5                    # 彩虹-4 无人机数量
    num_heli: int = 2                   # AH-64E 阿帕奇数量
    total_platforms: int = 7            # 5 UAV + 2 HELI

    # ---------- 目标 ----------
    num_radar: int = 1                  # 雷达目标数量
    num_cp: int = 1                     # 指挥所目标数量
    num_av: int = 2                     # 装甲车目标数量
    min_target_spacing_km: float = 10.0 # 目标最小间距 (km)

    # ---------- 传感器 ----------
    uav_sensors: List[str] = None       # UAV 挂载传感器: ["EO","SAR","ESM"]
    heli_sensors: List[str] = None      # HELI 挂载传感器: ["MMW","EOIR"]

    # ---------- 弹药 ----------
    munition_types: List[str] = None    # 弹药类型: ["HF","RKT","GUN"]
    heli_initial_munitions: dict = None # 初始弹药: {"HF":16,"RKT":76,"GUN":1200}

    # ---------- 求解器（开关模式）----------
    active_solver: SolverType = SolverType.CBC       # 当前激活的求解器
    enabled_solvers: List[SolverType] = None          # 求解器白名单
    solver_time_limit_s: float = 3.0                  # 求解时间上限 (s)
    solver_mip_gap: float = 1e-3                      # MIP gap 阈值

    # ---------- 目标函数权重 ----------
    lambda_strike: float = 1000.0       # 打击收益权重 λ_S, 设计方案 §4.4
    lambda_recon: float = 1.0           # 侦察收益权重 λ_R
    lambda_risk: float = 0.1            # 风险惩罚权重 λ_T
    lambda_cost: float = 0.01           # 成本惩罚权重 λ_C
    big_m_xi: float = 100.0             # 覆盖松弛大 M 惩罚
    theta_max: float = 1.5              # 单平台威胁上限, 设计方案 §4.5 约束(12)

    # ---------- 集结区 ----------
    staging_center_xy: tuple = (150.0, -50.0)  # 集结区坐标 (km)，任务区南侧外 50 km

    # ---------- 传感器扫描时间约束 ----------
    uav_loiter_speed_kmh: float = 150.0      # UAV loiter 速度, 彩虹-4 ISR 巡航 (km/h)
    sensor_scan_time_max_min: float = 25.0   # 单栅格最大扫描时间 (分钟), 基于 OODA 决策周期
    mission_total_time_max_min: float = 120.0 # 转场 + 扫描总时间上限 (分钟)

    # ---------- 地形系数 ----------
    terrain_occ: tuple = (1.0, 0.7, 0.4)
    terrain_time: tuple = (1.0, 1.3, 1.8)
    terrain_dist: tuple = (1.0, 1.15, 1.3)
    terrain_shield: tuple = (0.0, 0.3, 0.6)
    k_shield: float = 0.5
    uav_max_range_km: float = 3000.0
    risk_budget_uav: float = 5.0
    lambda_distance: float = 0.005
    uav_available_time_min: float = 600.0

    # ---------- 传感器-目标类型兼容性 ----------
    sensor_target_compat: dict = None   # 传感器->{目标类型: 0/1} 兼容矩阵

    # ---------- 终端显示 ----------
    verbose: int = 1                    # 0=静默, 1=摘要, 2=详细（含求解器日志）

    def __post_init__(self):
        """初始化默认值：传感器列表、弹药列表、求解器白名单。"""
        if self.uav_sensors is None:
            self.uav_sensors = ["EO", "SAR", "ESM"]
        if self.heli_sensors is None:
            self.heli_sensors = ["MMW", "EOIR"]
        if self.munition_types is None:
            self.munition_types = ["HF", "RKT", "GUN"]
        if self.heli_initial_munitions is None:
            self.heli_initial_munitions = {"HF": 16, "RKT": 76, "GUN": 1200}
        if self.sensor_target_compat is None:
            self.sensor_target_compat = {
                "EO":  {"RADAR": 1.0, "CP": 1.0, "AV": 1.0},
                "SAR": {"RADAR": 1.0, "CP": 1.0, "AV": 1.0},
                "ESM": {"RADAR": 1.0, "CP": 0.0, "AV": 0.0},
            }
        if self.enabled_solvers is None:
            self.enabled_solvers = [
                SolverType.GUROBI,
                SolverType.CBC,
                SolverType.ORTOOLS,
                SolverType.HIGHS,
            ]

    def validate_solver(self):
        """
        启动时校验: active_solver 必须在 enabled_solvers 白名单内。

        Raises:
            ValueError: 若 active_solver 不在白名单中
        """
        if self.active_solver not in self.enabled_solvers:
            raise ValueError(
                f"active_solver={self.active_solver} 不在 enabled_solvers 白名单中。"
                f"可用列表：{self.enabled_solvers}"
            )
