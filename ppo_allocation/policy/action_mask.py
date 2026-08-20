"""动作掩码（Action Mask）模块。

为每个区域的 6 种动作生成合法性掩码，屏蔽非法动作，
使 MaskablePPO 在采样/推理时只考虑合法动作。

核心原则：
  - KEEP 始终合法（保持现状永远是一个选项）
  - U0~U3 仅当该无人机能合法搜索该区域时可用
  - NO_UAV 仅在没有任何合法无人机时可用（严格掩码策略）
  - TARGET_DESTROYED 事件特殊处理：只有被释放的无人机可分配到区域
"""

import numpy as np
from config import (
    NUM_REGIONS,
    NUM_UAVS,
    ActionCode,
    EventType,
    NO_UAV,
    TaskType,
    # SensorType,  # 已注释：不再检查传感器-天气兼容性
    # Weather,     # 已注释：不再检查传感器-天气兼容性
)


def _valid_uav_for_region(uav, region) -> bool:
    """判断无人机是否具备搜索某区域的基本能力。

    检查条件（全部满足才返回 True）：
      1. 无人机存活
      2. 无人机不在跟踪任务中（跟踪已锁定目标，不可分心搜索）
      3. 传感器未故障
      4. EO 传感器不遇到雨天（EO 在雨天失效，SAR 全天候可用）

    Args:
        uav:    UAV 实体对象
        region: Region 实体对象

    Returns:
        bool: 该无人机能否搜索此区域
    """
    if not uav.alive:
        return False
    if uav.task == TaskType.TRACK:
        return False
    if uav.sensor_failed:
        return False
    # if uav.sensor == SensorType.EO and region.weather == Weather.RAINY:  # 已注释：全部使用 SAR
    #     return False
    return True


def _can_released_uav_take_region(env, released_uav_id: int, rid: int) -> bool:
    """判断被释放的无人机（TARGET_DESTROYED 事件中）是否可以接管某区域。

    限制：
      1. 必须自身具备搜索该区域的能力
      2. 不能接管自己已经负责的区域
      3. 只能接管未分配区域，或负载 >1 的无人机所负责的区域

    Args:
        env:            环境对象
        released_uav_id: 被释放的无人机编号
        rid:             候选区域编号

    Returns:
        bool: 是否可以接管
    """
    if released_uav_id == NO_UAV:
        return False

    u_release = env.uavs[released_uav_id]
    region = env.regions[rid]

    if not _valid_uav_for_region(u_release, region):
        return False

    old_uid = region.assigned_uav

    # 未分配区域可以直接接管
    if old_uid == NO_UAV:
        return True

    # 不允许接管自己已负责的区域
    if old_uid == released_uav_id:
        return False

    old_uav = env.uavs[old_uid]

    # 允许从高负载无人机手中接管（负载 > 1 时才释放，避免过度重新分配）
    return len(old_uav.regions) > 1


def build_action_mask(env) -> np.ndarray:
    """构建当前状态下的全局动作掩码。

    返回扁平化的布尔数组，shape=(24,)，为 MultiDiscrete([6,6,6,6]) 的
    每维动作标注合法性。True 表示合法，False 表示被屏蔽。

    sb3-contrib 的 MaskablePPO 接收此掩码后在计算动作概率时
    将非法动作的 logits 置为 -inf，确保不会被采样。

    策略细节：
      - 非受影响区域：只允许 KEEP
      - 受影响区域：允许所有合法的 U0~U3 + KEEP
      - 若受影响区域无合法无人机：只允许 NO_UAV
      - TARGET_DESTROYED 事件：只有被释放无人机可分配到候选区域

    Args:
        env: UAVTaskAllocationEnv 环境对象

    Returns:
        np.ndarray: shape=(24,) 的布尔掩码
    """
    event = env.current_event
    # 初始化全 False 掩码 (4 区域 × 6 动作)
    masks = np.zeros((NUM_REGIONS, 6), dtype=bool)

    # ----- TARGET_DESTROYED 特殊处理 -----
    # 只有被释放的无人机有机会接管区域
    if event.event_type == EventType.TARGET_DESTROYED:
        released = event.released_uav
        candidate_regions = []

        for rid in range(NUM_REGIONS):
            if _can_released_uav_take_region(env, released, rid):
                candidate_regions.append(rid)

        for rid in range(NUM_REGIONS):
            # KEEP 始终可选
            masks[rid, ActionCode.KEEP] = True
            if rid in candidate_regions:
                # released+1 是被释放无人机对应的动作码（U0=1, U1=2, ...）
                masks[rid, released + 1] = True

        return masks.reshape(-1)

    # # ----- WEATHER_INVALID 特殊处理 -----
    # # 天气恶化导致 EO 失效，所有区域允许合法无人机接管（支持链式调度）
    # # 已注释：全部使用 SAR 传感器，天气不影响传感器有效性
    # if event.event_type == EventType.WEATHER_INVALID:
    #     for rid in range(NUM_REGIONS):
    #         region = env.regions[rid]
    #         masks[rid, ActionCode.KEEP] = True
    #         for uid, uav in env.uavs.items():
    #             if _valid_uav_for_region(uav, region):
    #                 masks[rid, uid + 1] = True
    #
    #     return masks.reshape(-1)

    # ----- 一般事件处理 -----
    affected = set(event.affected_regions)

    for rid in range(NUM_REGIONS):
        region = env.regions[rid]

        # 非受影响区域：只允许保持现状
        if rid not in affected:
            masks[rid, ActionCode.KEEP] = True
            continue

        # 受影响区域：收集所有合法的无人机动作
        legal_uav_codes = []
        for uid, uav in env.uavs.items():
            if _valid_uav_for_region(uav, region):
                # 动作码 = 无人机编号 + 1（KEEP=0, U0=1, U1=2, ...）
                legal_uav_codes.append(uid + 1)

        if legal_uav_codes:
            # 有合法无人机：允许这些无人机动作
            for code in legal_uav_codes:
                masks[rid, code] = True
        else:
            # 严格掩码策略：仅在没有任何合法无人机时允许 NO_UAV
            # 避免 PPO 在有合法选项时选择"放弃分配"
            masks[rid, ActionCode.NO_UAV] = True

    # 展平为 (24,) 一维向量
    return masks.reshape(-1)
