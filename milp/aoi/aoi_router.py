"""
AOI 排序模块：通过枚举全排列选出综合得分最高的 AOI 执行顺序。

核心逻辑：
  1. 对每种 AOI 排列，计算路径综合得分 S(π)
  2. S(π) = 折扣后的 AOI 价值总和 - 路径距离惩罚
  3. 选取 S(π) 最大的排列，生成 AOIRouteState

适用规模：AOI 数量 ≤ 4（最多 4! = 24 种排列，直接枚举）。
"""

import math
from itertools import permutations
from typing import List, Optional, Tuple

import numpy as np

from .aoi_state import AoiInfo, AOIRouteState
from core.snapshot import TargetInfo

# ── 可调权重常量 ──────────────────────────────────────────
W_PRIORITY = 0.40       # 指挥员优先级权重
W_TARGET_VALUE = 0.20   # 目标平均价值权重
W_TARGET_THREAT = 0.20  # 目标平均威胁权重
W_TARGET_PRIOR = 0.20   # 目标存在先验概率权重

DISCOUNT_GAMMA = 0.90          # 时序折扣因子 γ
DISTANCE_PENALTY_ALPHA = 0.001 # 路径距离惩罚系数 α (1/km)


def _has_confirmed_target(aoi: AoiInfo, targets: List[TargetInfo],
                          grid_size_km: float = 50.0) -> bool:
    """检查 AOI 边界内是否存在已确认且存活的目标。"""
    x_min = (aoi.col - 1) * grid_size_km
    x_max = aoi.col * grid_size_km
    y_min = (aoi.row - 1) * grid_size_km
    y_max = aoi.row * grid_size_km
    for t in targets:
        if t.confirmed and t.alive:
            if x_min <= t.pos_est[0] <= x_max and y_min <= t.pos_est[1] <= y_max:
                return True
    return False


def _aoi_center(aoi: AoiInfo, grid_size_km: float = 50.0) -> np.ndarray:
    """计算 AOI 中心坐标（与 generate_aoi_grids 一致）。"""
    cx = (aoi.col - 0.5) * grid_size_km
    cy = (aoi.row - 0.5) * grid_size_km
    return np.array([cx, cy], dtype=np.float64)


def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    """两点欧氏距离（km）。"""
    return float(np.linalg.norm(p1 - p2))


def _aoi_value(aoi: AoiInfo) -> float:
    """
    计算单个 AOI 的综合价值。

    V(a) = w_p * p_a + w_v * v_a + w_t * t_a + w_ρ * ρ_a
    """
    return (
        W_PRIORITY      * aoi.priority
        + W_TARGET_VALUE  * aoi.target_value
        + W_TARGET_THREAT * aoi.target_threat
        + W_TARGET_PRIOR  * aoi.target_prior
    )


def _score_sequence(
    seq: Tuple[AoiInfo, ...],
    start_pos: np.ndarray,
    grid_size_km: float = 50.0,
) -> float:
    """
    计算给定 AOI 排列的综合得分。

    S(π) = Σ γ^(k-1) * V(a_k)  -  α * D(π)

    D(π) = d(start, a_1) + Σ d(a_k, a_{k+1})
    """
    centers = [_aoi_center(a, grid_size_km) for a in seq]

    # 路径总长度
    total_dist = _euclidean(start_pos, centers[0])
    for i in range(len(centers) - 1):
        total_dist += _euclidean(centers[i], centers[i + 1])

    # 折扣价值总和
    total_value = sum(
        (DISCOUNT_GAMMA ** k) * _aoi_value(seq[k])
        for k in range(len(seq))
    )

    return total_value - DISTANCE_PENALTY_ALPHA * total_dist


def _best_permutation(aois: List[AoiInfo], start_pos: np.ndarray,
                      grid_size_km: float = 50.0) -> List[AoiInfo]:
    """对给定 AOI 列表枚举全排列，返回得分最高的顺序（空列表直接返回）。"""
    if not aois:
        return []
    if len(aois) == 1:
        return list(aois)
    best_seq, best_score = None, float("-inf")
    for perm in permutations(aois):
        score = _score_sequence(perm, start_pos, grid_size_km)
        if score > best_score:
            best_score = score
            best_seq = list(perm)
    return best_seq


class AOIRouter:
    """
    AOI 排序器。

    首次调用时执行一次排序，后续复用已有排序结果（由 AOIRouteState 保存）。

    Args:
        grid_size_km: AOI 边长，默认 50 km
    """

    def __init__(self, grid_size_km: float = 50.0):
        self.grid_size_km = grid_size_km

    def sort(
        self,
        aois: List[AoiInfo],
        start_pos: Optional[np.ndarray] = None,
        targets: Optional[List[TargetInfo]] = None,
    ) -> AOIRouteState:
        """
        对多个 AOI 枚举全排列，返回得分最高的执行顺序。

        含有已确认目标的 AOI 优先排在前面，其余 AOI 按综合得分排序。

        Args:
            aois:      AOI 列表（2~4 个）
            start_pos: 出发点坐标 [x, y] km，默认集结区 (150, -50)
            targets:   目标列表，用于判断 AOI 内是否有已确认目标

        Returns:
            AOIRouteState（current_aoi_index=0，route_status="RUNNING"）
        """
        if not aois:
            raise ValueError("AOI 列表不能为空")

        if start_pos is None:
            start_pos = np.array([150.0, -50.0])

        # 分离：有已确认目标的 AOI 优先
        if targets:
            front = [a for a in aois if _has_confirmed_target(a, targets, self.grid_size_km)]
            back = [a for a in aois if a not in front]
        else:
            front, back = [], list(aois)

        # 排序前段（有已确认目标），从 start_pos 出发
        best_seq = list(_best_permutation(front, start_pos, self.grid_size_km))

        # 排序后段（无已确认目标），从前段终点出发
        if best_seq:
            back_start = _aoi_center(best_seq[-1], self.grid_size_km)
        else:
            back_start = start_pos
        best_seq += list(_best_permutation(back, back_start, self.grid_size_km))

        return AOIRouteState(
            aoi_sequence=[a.id for a in best_seq],
            current_aoi_index=0,
            route_status="RUNNING",
        )
