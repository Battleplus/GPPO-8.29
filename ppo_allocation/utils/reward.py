"""奖励函数模块。

计算每个决策步的即时奖励，引导 PPO 学习最优区域分配策略。

奖励 = 正向项 + 负向惩罚项，各项通过 config 中的权重控制相对重要性。

奖励组成：
  1. 区域合法分配奖励 (+W_REGION_ASSIGNED × 合法分配数)
  2. 未分配区域惩罚 (-W_UNASSIGNED × 未分配数)
  3. 传感器匹配奖励 (+W_SENSOR_MATCH × 匹配数)
  4. 距离代价        (-W_DISTANCE × 平均归一化距离)
  5. 负载均衡惩罚    (-W_BALANCE × 负载差距)
  6. 事件成功奖励    (+W_EVENT_SUCCESS，主导信号)
  7. 切换代价        (-W_SWITCH × 分配变更数)
  8. 非法动作惩罚    (-W_INVALID × 修复数)
  9. 提前终止奖励    (+W_TERMINAL_BONUS，所有区域合法稳定)
"""

import numpy as np
from typing import Dict, Any
from config import (
    NUM_UAVS,
    NUM_REGIONS,
    NO_UAV,
    NO_TARGET,
    AREA_SIZE,
    TaskType,
    SensorType,
    Weather,
    W_REGION_ASSIGNED,
    W_UNASSIGNED,
    W_DISTANCE,
    W_BALANCE,
    W_SENSOR_MATCH,
    W_EVENT_SUCCESS,
    W_SWITCH,
    W_INVALID,
    W_TERMINAL_BONUS,
    W_IDLE_UAV,
)
from utils.geometry import distance_xy


def is_legal_search_assignment(uav, region) -> bool:
    """验证无人机对区域的搜索分配是否合法。

    检查 5 项条件：
      1. 区域有分配（不是 NO_UAV）
      2. 无人机存活
      3. 无人机不在跟踪任务中
      4. 传感器未故障
      5. EO 传感器不遇到雨天

    Args:
        uav:    UAV 实体
        region: Region 实体

    Returns:
        bool: 分配是否合法
    """
    if region.assigned_uav == NO_UAV:
        return False
    if not uav.alive:
        return False
    if uav.task == TaskType.TRACK:
        return False
    if uav.sensor_failed:
        return False
    # if uav.sensor == SensorType.EO and region.weather == Weather.RAINY:  # 已注释：全部使用 SAR
    #     return False
    return True


def compute_reward(env, old_assignments: Dict[int, int], repaired_action,
                   invalid_count: int, event_success: bool, terminated: bool) -> float:
    """计算单步决策的奖励值。

    Args:
        env:             环境对象（包含当前状态）
        old_assignments: 决策前的区域分配映射 {rid: uid}
        repaired_action: 修复后实际执行的动作向量
        invalid_count:   被修复的非法动作数量
        event_success:   当前事件是否被 PPO 成功解决
        terminated:      是否触发提前终止（所有区域合法稳定）

    Returns:
        float: 累积奖励值
    """
    reward = 0.0

    # ===== 1. 合法分配奖励 + 未分配惩罚 + 传感器匹配奖励 =====
    legal_assigned_count = 0
    unassigned_count = 0
    sensor_match_count = 0

    for rid, region in env.regions.items():
        uid = region.assigned_uav
        if uid == NO_UAV:
            unassigned_count += 1
            continue

        uav = env.uavs[uid]
        if is_legal_search_assignment(uav, region):
            legal_assigned_count += 1
            sensor_match_count += 1
        else:
            unassigned_count += 1

    reward += W_REGION_ASSIGNED * legal_assigned_count
    reward -= W_UNASSIGNED * unassigned_count
    reward += W_SENSOR_MATCH * sensor_match_count

    # ===== 2. 距离代价（仅在分配发生变化的区域计算） =====
    distance_cost = 0.0
    changed_count = 0
    for rid, old_uid in old_assignments.items():
        new_uid = env.regions[rid].assigned_uav
        if old_uid != new_uid and new_uid != NO_UAV:
            uav = env.uavs[new_uid]
            region = env.regions[rid]
            distance_cost += distance_xy(uav.x, uav.y, region.center_x, region.center_y) / AREA_SIZE
            changed_count += 1

    if changed_count > 0:
        reward -= W_DISTANCE * (distance_cost / changed_count)

    # ===== 3. 搜索负载均衡惩罚 =====
    load_list = [
        len(uav.regions)
        for uav in env.uavs.values()
        if uav.alive and uav.task != TaskType.TRACK
    ]
    if len(load_list) > 1:
        reward -= W_BALANCE * (max(load_list) - min(load_list))

    # ===== 3.5 闲置无人机惩罚（鼓励把闲机派出去） =====
    idle_search = sum(
        1 for u in env.uavs.values()
        if u.alive and u.task == TaskType.IDLE and u.target_id == NO_TARGET
    )
    reward -= W_IDLE_UAV * idle_search

    # ===== 4. 事件成功解决奖励（主导信号） =====
    if event_success:
        reward += W_EVENT_SUCCESS

    # 事件未解决，对仍空缺的受影响区域加惩罚
    unresolved = 0
    for rid in env.current_event.affected_regions:
        if env.regions[rid].assigned_uav == NO_UAV:
            unresolved += 1
    if unresolved > 0:
        reward -= W_UNASSIGNED * unresolved

    # ===== 5. 切换代价 + 非法动作惩罚 =====
    switch_count = sum(
        1 for rid, old_uid in old_assignments.items()
        if env.regions[rid].assigned_uav != old_uid
    )
    reward -= W_SWITCH * switch_count
    reward -= W_INVALID * invalid_count

    # ===== 6. 提前终止奖励 =====
    # 所有区域合法稳定，成功完成 episode，给予大额正向奖励
    if terminated:
        reward += W_TERMINAL_BONUS

    return float(reward)
