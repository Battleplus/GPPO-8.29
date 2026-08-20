"""动作修复模块。

PPO 输出的原始动作可能存在违反约束的情况（即使使用了 mask），
本模块对其进行修复，确保最终执行的联合动作满足所有业务约束。

修复步骤：
  1. 单区域修复：纠正违反 per-region mask 的动作
  2. 全局修复：处理跨区域冲突（如同一无人机被分配到多个区域）
"""

import numpy as np
from config import (
    NUM_REGIONS,
    ActionCode,
    EventType,
    NO_UAV,
)
from policy.action_mask import build_action_mask
from utils.geometry import distance_xy


def repair_action(env, action):
    """修复 PPO 原始动作中的违反约束项。

    两层修复：
      第一层：逐区域检查动作是否在 mask 内，不在则回退为 KEEP 或首个合法动作
      第二层：针对 TARGET_DESTROYED 事件，限制被释放无人机至多接管一个区域

    Args:
        env:    UAVTaskAllocationEnv 环境对象
        action: PPO 原始动作（4 个整数的 list/array）

    Returns:
        repaired_action: np.ndarray shape=(4,)，修复后的动作
        invalid_count:   被修复的动作项数量（用于奖励惩罚）
    """
    # 深拷贝，避免修改原始动作
    action = np.array(action, dtype=np.int64).copy()
    invalid_count = 0

    # 获取当前掩码用于合法性检查
    flat_mask = build_action_mask(env).reshape(NUM_REGIONS, 6)

    # ---------- 第一层：逐区域修复 ----------
    # 如果某个区域的动作不在 mask 合法列表中，强制回退
    for rid in range(NUM_REGIONS):
        code = int(action[rid])
        if not flat_mask[rid, code]:
            # 获取该区域的所有合法动作
            valid_codes = np.where(flat_mask[rid])[0]
            # 优先回退到 KEEP（保持现状代价最小）
            if ActionCode.KEEP in valid_codes:
                action[rid] = int(ActionCode.KEEP)
            else:
                action[rid] = int(valid_codes[0])
            invalid_count += 1

    # ---------- 第二层：TARGET_DESTROYED 跨区域修复 ----------
    # 被释放的无人机最多接管一个区域，多的回退为 KEEP
    if env.current_event.event_type == EventType.TARGET_DESTROYED:
        released = env.current_event.released_uav
        if released != NO_UAV:
            release_code = released + 1  # 被释放无人机的动作码
            take_regions = [rid for rid in range(NUM_REGIONS) if int(action[rid]) == release_code]

            if len(take_regions) > 1:
                # 选择最优的一个区域保留，其余回退为 KEEP
                keep_rid = _select_best_takeover_region(env, released, take_regions)
                for rid in take_regions:
                    if rid != keep_rid:
                        action[rid] = int(ActionCode.KEEP)
                        invalid_count += 1

    return action, invalid_count


def _select_best_takeover_region(env, released_uid: int, candidate_regions):
    """为被释放无人机选择最优接管区域。

    优先级排序（按元组升序）：
      1. 优先接管负载较高的原无人机所负责的区域（促进负载均衡）
      2. 距离越近越优先（减少移动代价）
      3. 区域编号小的优先（确定性打破平局）

    Args:
        env:              环境对象
        released_uid:      被释放的无人机编号
        candidate_regions: 候选区域编号列表

    Returns:
        int: 最优区域编号
    """
    released = env.uavs[released_uid]

    def score(rid):
        region = env.regions[rid]
        old_uid = region.assigned_uav
        # 原无人机的负载（越大越优先 → 取负值使升序排列正确）
        old_load = 0 if old_uid == NO_UAV else len(env.uavs[old_uid].regions)
        # 被释放无人机到此区域的距离
        dist = distance_xy(released.x, released.y, region.center_x, region.center_y)
        # 返回排序元组：(-负载, 距离, 区域id)
        return (-old_load, dist, rid)

    return sorted(candidate_regions, key=score)[0]
