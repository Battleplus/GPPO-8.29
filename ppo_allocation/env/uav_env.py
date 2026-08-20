"""无人机任务分配强化学习环境。

本模块实现了符合 Gymnasium 接口的无人机搜索任务局部重分配环境。
核心设计原则：
  - PPO 只负责搜索(SEARCH)区域的重分配
  - 目标追踪(TRACK)由规则逻辑处理，不作为 PPO 的动作输出
  - 突发事件驱动决策：每步生成一个事件，PPO 输出新的区域分配方案

标准交互流程:
  1. env.reset()          → 初始化场景，返回 obs + info(含 action_mask)
  2. env.step(action)     → 执行分配动作，返回 (obs, reward, terminated, truncated, info)
  3. env.action_masks()   → 获取当前合法动作掩码（供 MaskablePPO 使用）
  4. env.snapshot()       → 导出当前状态快照（用于可视化和导出）
"""

from __future__ import annotations

import copy
from typing import Dict, Tuple, Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    AREA_SIZE,
    NUM_UAVS,
    NUM_REGIONS,
    NUM_TARGETS,
    OBS_DIM,
    ACTION_N_VEC,
    ActionCode,
    Weather,
    SensorType,
    TaskType,
    TargetType,
    EventType,
    NO_UAV,
    NO_TARGET,
    REGION_CENTERS,
    MAX_DECISION_STEPS,
    RANDOM_SEED,
    EVENT_INTERVAL,
)
from env.uav import UAV
from env.region import Region
from env.target import Target
from env.event import Event
# from env.weather import sample_weather, flip_weather  # 已注释：SAR 全天候，天气不影响
from policy.action_mask import build_action_mask
from policy.action_repair import repair_action
from utils.reward import compute_reward


class UAVTaskAllocationEnv(gym.Env):
    """事件驱动的无人机搜索任务重分配环境。

    环境维护三类实体（无人机、区域、目标），每步决策产生一个突发事件，
    PPO 智能体输出对 4 个区域的新分配方案。

    动作空间: MultiDiscrete([6, 6, 6, 6])
        - 每个区域 6 种动作: KEEP / U0 / U1 / U2 / U3 / NO_UAV

    观测空间: Box(low=-1, high=1, shape=(OBS_DIM,))
        - 扁平化向量，包含所有无人机、区域、目标、事件的特征
    """

    # Gymnasium 元数据（当前不实现 render）
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_decision_steps: int = MAX_DECISION_STEPS,
        seed: Optional[int] = RANDOM_SEED,
        random_event_mode: bool = True,
    ):
        """初始化环境。

        Args:
            max_decision_steps: episode 最大决策步数，超出后 truncated=True
            seed:              随机种子（用于 numpy Generator）
            random_event_mode: 是否使用随机事件生成器；
                               训练时为 True；实际应用时可设为 False 并注入外部事件
        """
        super().__init__()
        self.max_decision_steps = max_decision_steps
        self.random_event_mode = random_event_mode
        # 使用 numpy 新式随机数生成器（推荐做法，比旧的 np.random 更可控）
        self.rng = np.random.default_rng(seed)

        # 定义观测空间：一维连续向量，范围 [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        # 定义动作空间：4 个离散动作，每个有 6 个选项
        self.action_space = spaces.MultiDiscrete(ACTION_N_VEC)

        # 实体字典：key 为编号，value 为实体对象
        self.uavs: Dict[int, UAV] = {}
        self.regions: Dict[int, Region] = {}
        self.targets: Dict[int, Target] = {}

        # 当前触发 PPO 决策的事件
        self.current_event: Event = Event(EventType.REGION_VACANCY, [0], description="init")

        # 当前决策步计数
        self.decision_step = 0

        # 距上次事件的步数（用于控制事件间隔）
        self.steps_since_event = 0

    # =========================
    # Gymnasium 标准 API
    # =========================

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境到初始状态。

        完成以下步骤：
          1. 初始化区域（含随机天气）
          2. 初始化无人机（位置在对应区域中心，随机传感器）
          3. 初始化目标（随机位置、随机区域）
          4. 执行初始分配（无人机 i 分配到区域 i）
          5. 生成首个决策事件

        Returns:
            obs:  初始观测向量
            info: 包含 action_mask 和当前 event 的字典
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.decision_step = 0
        self.steps_since_event = 0
        self._init_regions()
        self._init_uavs()
        self._init_targets()
        self._initial_assignment()
        self.current_event = self._generate_next_event()

        obs = self._get_obs()
        info = {
            "action_mask": self.action_masks(),
            "event": self.current_event,
        }
        return obs, info

    def step(self, action):
        """执行一步 PPO 决策。

        流程：
          1. 记录旧的分配状态
          2. 修复并执行 PPO 动作
          3. 计算奖励（含提前终止奖励）
          4. 检查是否所有区域均已合法分配 → 提前终止
          5. 每 EVENT_INTERVAL 步才生成新事件

        Args:
            action: 原始 PPO 动作，shape=(4,)，每个元素 ∈ {0..5}

        Returns:
            obs:        新的观测向量
            reward:     本次决策的奖励值
            terminated: 所有区域合法分配时提前终止
            truncated:  超出 max_decision_steps 而被截断
            info:       包含 mask、事件、修复信息等
        """
        self.decision_step += 1
        self.steps_since_event += 1

        # 快照旧分配状态，供奖励函数计算变化量
        old_assignments = {rid: region.assigned_uav for rid, region in self.regions.items()}

        # 修复非法动作并执行
        repaired_action, invalid_count = repair_action(self, action)
        self._execute_action(repaired_action)

        # 判断事件是否已被解决
        event_success = self._event_success()

        # 检查是否所有区域都已合法分配（提前终止条件）
        all_valid = self._all_regions_valid()

        # 判断终止与截断
        # 自动终止：没有存活且可搜的无人机时，无需继续
        can_search = any(
            u.alive and u.task != TaskType.TRACK
            for u in self.uavs.values()
        )
        no_uav_available = not can_search

        terminated = all_valid or no_uav_available
        truncated = (not terminated) and (self.decision_step >= self.max_decision_steps)

        # 计算奖励
        reward = compute_reward(self, old_assignments, repaired_action, invalid_count,
                                event_success, terminated)

        # 事件生成：每 EVENT_INTERVAL 步且未终止时才生成新事件
        if not terminated and not truncated:
            if self.steps_since_event >= EVENT_INTERVAL:
                self.current_event = self._generate_next_event()
                self.steps_since_event = 0

        obs = self._get_obs()
        no_uav_count = int(sum(1 for code in repaired_action if int(code) == int(ActionCode.NO_UAV)))

        info = {
            "action_mask": self.action_masks(),
            "event": self.current_event,
            "repaired_action": repaired_action,
            "invalid_count": invalid_count,
            "event_success": event_success,
            "no_uav_count": no_uav_count,
            "all_valid": all_valid,
            "snapshot": self.snapshot(),
        }
        return obs, reward, terminated, truncated, info

    def action_masks(self):
        """获取当前状态的合法动作掩码。

        供 MaskablePPO 调用，在计算动作概率时屏蔽非法动作。
        （标准 PPO 不支持 mask，因此必须使用 sb3-contrib 的 MaskablePPO。）

        Returns:
            np.ndarray: shape=(24,) 的扁平布尔数组，
                        为 True 表示该动作合法
        """
        return build_action_mask(self)

    # =========================
    # 初始化辅助方法
    # =========================

    def _init_regions(self):
        """初始化所有搜索区域。

        每个区域从预设的中心坐标创建，天气默认晴天。
        （SAR 全天候工作，天气不再影响传感器有效性）
        """
        self.regions = {}
        for rid in range(NUM_REGIONS):
            cx, cy = REGION_CENTERS[rid]
            self.regions[rid] = Region(
                rid=rid,
                center_x=cx,
                center_y=cy,
                weather=Weather.SUNNY,
                # weather=sample_weather(self.rng),  # 已注释：SAR 全天候
            )

    def _init_uavs(self):
        """初始化所有无人机。

        每架无人机初始位于对应编号区域的中心（U0→R0, U1→R1, ...）。
        全部使用 SAR 传感器（全天候，不受天气影响），初始分配始终合法。
        """
        self.uavs = {}
        for uid in range(NUM_UAVS):
            cx, cy = REGION_CENTERS[uid]
            # 统一使用 SAR 传感器
            sensor = SensorType.SAR
            # rid_weather = self.regions[uid].weather  # 已注释：SAR 全天候
            # if rid_weather == Weather.RAINY:
            #     sensor = SensorType.SAR
            # else:
            #     sensor = SensorType(int(self.rng.integers(0, 2)))
            self.uavs[uid] = UAV(
                uid=uid,
                x=float(cx),
                y=float(cy),
                sensor=sensor,
                alive=True,
                sensor_failed=False,
                task=TaskType.SEARCH,
                regions=set(),
                target_id=NO_TARGET,
            )

    def _init_targets(self):
        """初始化所有目标。

        前 2 个目标为 CAR（可移动），第 3 个为 COMMAND（固定）。
        目标位置在所属区域中心附近以高斯噪声随机放置，并限制在地图范围内。
        """
        self.targets = {}
        target_types = [TargetType.CAR, TargetType.CAR, TargetType.COMMAND]
        for tid in range(NUM_TARGETS):
            rid = int(self.rng.integers(0, NUM_REGIONS))
            cx, cy = REGION_CENTERS[rid]
            # 在区域中心附近以 N(0,4) 噪声随机放置
            x = float(np.clip(cx + self.rng.normal(0, 4), 0, AREA_SIZE))
            y = float(np.clip(cy + self.rng.normal(0, 4), 0, AREA_SIZE))
            self.targets[tid] = Target(
                tid=tid,
                target_type=target_types[tid],
                x=x,
                y=y,
                region=rid,
                movable=(target_types[tid] == TargetType.CAR),
            )

    def _initial_assignment(self):
        """执行初始区域分配。

        默认策略：无人机 Ui 分配到区域 Ri（一一对应）。
        由于 _init_uavs() 已确保传感器与区域天气兼容，
        初分配始终合法，无需清除。
        """
        for rid in range(NUM_REGIONS):
            uid = rid
            self._assign_region_to_uav(rid, uid)

    # =========================
    # 观测向量构建
    # =========================
    # 将所有实体和事件的特征拼接为一个一维浮点向量，
    # 每类特征独立编码，最后拼接为 OBS_DIM 维向量。

    def _get_obs(self) -> np.ndarray:
        """构建完整观测向量。

        Returns:
            np.ndarray: shape=(OBS_DIM,) 的归一化特征向量
        """
        features = []
        features.extend(self._encode_uavs())      # NUM_UAVS * UAV_FEATURE_DIM
        features.extend(self._encode_regions())   # NUM_REGIONS * REGION_FEATURE_DIM
        features.extend(self._encode_targets())   # NUM_TARGETS * TARGET_FEATURE_DIM
        features.extend(self._encode_event())     # EVENT_FEATURE_DIM
        obs = np.asarray(features, dtype=np.float32)
        assert obs.shape == (OBS_DIM,), f"Expected obs dim {OBS_DIM}, got {obs.shape}"
        return obs

    def _encode_uavs(self):
        """编码所有无人机特征。

        每架无人机输出 UAV_FEATURE_DIM=17 维特征：
          [0]:  alive (0/1)
          [1]:  x / AREA_SIZE（归一化 x）
          [2]:  y / AREA_SIZE（归一化 y）
          [3]:  is_EO (0/1)
          [4]:  is_SAR (0/1)
          [5]:  task_is_SEARCH (0/1)
          [6]:  task_is_TRACK (0/1)
          [7]:  task_is_IDLE (0/1)
          [8-11]:  区域 multi-hot（所属 4 个区域）
          [12-15]: 目标 one-hot（4 类：T0/T1/T2/无目标）
          [16]: 负载比例 assigned_count / NUM_REGIONS
        """
        vec = []
        for uid in range(NUM_UAVS):
            u = self.uavs[uid]
            # 区域 multi-hot 编码
            region_multi_hot = [1.0 if rid in u.regions else 0.0 for rid in range(NUM_REGIONS)]
            # 目标 one-hot 编码（索引 3 表示"无目标"）
            target_one_hot = [0.0] * 4
            if u.target_id in [0, 1, 2]:
                target_one_hot[u.target_id] = 1.0
            else:
                target_one_hot[3] = 1.0

            vec.extend([
                1.0 if u.alive else 0.0,
                u.x / AREA_SIZE,
                u.y / AREA_SIZE,
                1.0 if u.sensor == SensorType.EO else 0.0,
                1.0 if u.sensor == SensorType.SAR else 0.0,
                1.0 if u.task == TaskType.SEARCH else 0.0,
                1.0 if u.task == TaskType.TRACK else 0.0,
                1.0 if u.task == TaskType.IDLE else 0.0,
            ])
            vec.extend(region_multi_hot)
            vec.extend(target_one_hot)
            vec.append(len(u.regions) / NUM_REGIONS)
        return vec

    def _encode_regions(self):
        """编码所有区域特征。

        每个区域输出 REGION_FEATURE_DIM=10 维特征：
          [0]:  center_x / AREA_SIZE
          [1]:  center_y / AREA_SIZE
          [2]:  is_SUNNY (0/1)
          [3]:  is_RAINY (0/1)
          [4]:  is_assigned（是否有合法有效的搜索分配）
          [5-9]: 分配 one-hot（5 类：U0/U1/U2/U3/NO_UAV）
        """
        vec = []
        for rid in range(NUM_REGIONS):
            r = self.regions[rid]
            # is_assigned: 只有在分配且合法时才是 1
            is_assigned = 1.0 if r.assigned_uav != NO_UAV and self._valid_search_assign(r.assigned_uav, rid) else 0.0

            # 分配 one-hot：前 4 位对应 U0-U3，第 5 位表示无分配
            assigned_one_hot = [0.0] * 5
            if r.assigned_uav == NO_UAV:
                assigned_one_hot[4] = 1.0
            else:
                assigned_one_hot[r.assigned_uav] = 1.0

            vec.extend([
                r.center_x / AREA_SIZE,
                r.center_y / AREA_SIZE,
                1.0 if r.weather == Weather.SUNNY else 0.0,
                1.0 if r.weather == Weather.RAINY else 0.0,
                is_assigned,
            ])
            vec.extend(assigned_one_hot)
        return vec

    def _encode_targets(self):
        """编码所有目标特征。

        每个目标输出 TARGET_FEATURE_DIM=15 维特征。
        未被发现的目标位置信息置为 0（模拟部分可观测性）。
          [0]:  is_CAR (0/1)
          [1]:  is_COMMAND (0/1)
          [2]:  discovered (0/1)
          [3]:  tracked (0/1)
          [4]:  destroyed (0/1)
          [5]:  known_x / AREA_SIZE（未发现时为 0）
          [6]:  known_y / AREA_SIZE（未发现时为 0）
          [7-10]:  区域 one-hot（未发现时为全 0）
          [11-14]: 跟踪者 one-hot（4 架无人机）
        """
        vec = []
        for tid in range(NUM_TARGETS):
            t = self.targets[tid]

            # 未发现的目标：位置和区域未知（全零向量）
            if t.discovered:
                known_x = t.x / AREA_SIZE
                known_y = t.y / AREA_SIZE
                region_one_hot = [1.0 if rid == t.region else 0.0 for rid in range(NUM_REGIONS)]
            else:
                known_x = 0.0
                known_y = 0.0
                region_one_hot = [0.0] * NUM_REGIONS

            # 跟踪者 one-hot
            tracker_one_hot = [0.0] * NUM_UAVS
            if t.tracked and t.tracker_id != NO_UAV:
                tracker_one_hot[t.tracker_id] = 1.0

            vec.extend([
                1.0 if t.target_type == TargetType.CAR else 0.0,
                1.0 if t.target_type == TargetType.COMMAND else 0.0,
                1.0 if t.discovered else 0.0,
                1.0 if t.tracked else 0.0,
                1.0 if t.destroyed else 0.0,
                known_x,
                known_y,
            ])
            vec.extend(region_one_hot)
            vec.extend(tracker_one_hot)
        return vec

    def _encode_event(self):
        """编码当前事件特征。

        输出 EVENT_FEATURE_DIM=12 维特征：
          [0-3]:   事件类型 one-hot（4 类）
          [4-7]:   影响区域 multi-hot（4 个区域）
          [8-11]:  相关 UAV one-hot（被释放/损毁的无人机）
        """
        e = self.current_event
        type_one_hot = [0.0] * 4
        type_one_hot[int(e.event_type)] = 1.0

        region_multi_hot = [1.0 if rid in e.affected_regions else 0.0 for rid in range(NUM_REGIONS)]

        uav_one_hot = [0.0] * NUM_UAVS
        event_uav = e.released_uav if e.released_uav != NO_UAV else (
            e.damaged_uav if e.damaged_uav != NO_UAV else e.weather_disabled_uav)
        if event_uav != NO_UAV:
            uav_one_hot[event_uav] = 1.0

        return type_one_hot + region_multi_hot + uav_one_hot

    # =========================
    # 事件生成
    # =========================

    def _generate_next_event(self) -> Event:
        """生成下一个决策事件（训练用随机事件生成器）。

        从 5 种候选事件类型中随机采样并尝试生成，
        若 30 次内未生成有效事件，回退为简单区域空缺事件。

        实际应用时，可替换此方法：将完整仿真器产生的事件
        直接注入 env.current_event 即可。

        Returns:
            Event: 生成的事件对象
        """
        candidates = [
            self._event_uav_damage,
            # self._event_weather_invalid,  # 已注释：SAR 全天候，天气不影响
            self._event_target_discovered_causes_vacancy,
            self._event_target_destroyed,
        ]

        # 多次尝试以生成有意义的事件（避免因条件不满足返回 None）
        for _ in range(30):
            fn = candidates[int(self.rng.integers(0, len(candidates)))]
            event = fn()
            if event is not None:
                return event

        # 兜底：返回空事件（不修改状态）
        return Event(EventType.TARGET_DESTROYED, [], description="Fallback: no event")

    def _event_region_vacancy(self):
        """生成区域空缺事件。

        随机选择一个已被分配的区域，清除其分配，
        模拟无人机离开或被重新部署导致的空缺。

        Returns:
            Event 或 None（无可用区域时）
        """
        assigned_regions = [rid for rid, r in self.regions.items() if r.assigned_uav != NO_UAV]
        if not assigned_regions:
            return None
        rid = int(self.rng.choice(assigned_regions))
        self._clear_region_assignment(rid)
        self.regions[rid].need_reassign = True
        return Event(EventType.REGION_VACANCY, [rid], description=f"Region R{rid} has no search UAV")

    def _event_uav_damage(self):
        """生成无人机损毁事件。

        随机选择一架存活且有负责区域的无人机，
        将其标记为损毁，清除其所有区域分配。

        Returns:
            Event 或 None（无可损毁的无人机时）
        """
        candidates = [uid for uid, u in self.uavs.items() if u.alive and len(u.regions) > 0]
        if not candidates:
            return None
        uid = int(self.rng.choice(candidates))
        u = self.uavs[uid]
        affected = list(u.regions)
        # 损毁：标记不存活、任务置为空闲、清除目标
        u.alive = False
        u.task = TaskType.IDLE
        u.target_id = NO_TARGET
        for rid in affected:
            self._clear_region_assignment(rid)
            self.regions[rid].need_reassign = True
        return Event(EventType.UAV_DAMAGE, affected, damaged_uav=uid, description=f"U{uid} damaged")

    # def _event_weather_invalid(self):
    #     """生成天气恶化事件。
    #
    #     优先选择当前由 EO 传感器无人机负责的区域，
    #     将其天气切换为雨天（EO 失效），清除分配。
    #
    #     Returns:
    #         Event 或 None（无合适的 EO 区域时）
    #     """
    #     eo_regions = [
    #         rid for rid, r in self.regions.items()
    #         if r.assigned_uav != NO_UAV
    #         and self.uavs[r.assigned_uav].sensor == SensorType.EO
    #         and self.uavs[r.assigned_uav].alive
    #     ]
    #     if not eo_regions:
    #         return None
    #
    #     rid = int(self.rng.choice(eo_regions))
    #     old_uid = self.regions[rid].assigned_uav
    #     self.regions[rid].weather = Weather.RAINY
    #     self._clear_region_assignment(rid)
    #     self.regions[rid].need_reassign = True
    #     return Event(EventType.WEATHER_INVALID, [rid],
    #                  weather_disabled_uav=old_uid,
    #                  description=f"Weather invalid: R{rid} changed to rainy")

    def _event_target_discovered_causes_vacancy(self):
        """生成目标发现事件（规则触发，非 PPO 决策）。

        模拟一架搜索中的无人机发现了一个未发现的目标：
          1. 目标标记为已发现并进入跟踪状态
          2. 该无人机切换为 TRACK 任务，清空其搜索区域
          3. 被清空的搜索区域产生空缺，需要 PPO 重分配

        Returns:
            Event 或 None（无合适无人机或目标时）
        """
        searchers = [uid for uid, u in self.uavs.items() if u.alive and u.task == TaskType.SEARCH and len(u.regions) > 0]
        undiscovered_targets = [tid for tid, t in self.targets.items() if not t.discovered and not t.destroyed]
        valid_pairs = [(uid, tid) for uid in searchers for tid in undiscovered_targets
                       if self.targets[tid].region in self.uavs[uid].regions]
        if not valid_pairs:
            return None

        uid, tid = valid_pairs[int(self.rng.choice(len(valid_pairs)))]

        u = self.uavs[uid]
        t = self.targets[tid]

        affected = list(u.regions)

        # 规则：目标被发现 → 发现者进入跟踪模式
        t.discovered = True
        t.tracked = True
        t.tracker_id = uid

        u.regions.clear()
        u.task = TaskType.TRACK
        u.target_id = tid

        for rid in affected:
            self.regions[rid].assigned_uav = NO_UAV
            self.regions[rid].need_reassign = True

        return Event(
            EventType.REGION_VACANCY,
            affected,
            description=f"U{uid} discovered T{tid}; its search regions need reassignment",
        )

    def _event_target_destroyed(self):
        """生成目标被摧毁事件。

        随机选择一个正在被跟踪且未被摧毁的目标，
        标记为已摧毁，释放其跟踪无人机。

        Returns:
            Event 或 None（无可摧毁目标时）
        """
        tracked_targets = [tid for tid, t in self.targets.items() if t.tracked and not t.destroyed and t.tracker_id != NO_UAV]
        if not tracked_targets:
            return None

        tid = int(self.rng.choice(tracked_targets))
        t = self.targets[tid]
        released_uid = t.tracker_id

        # 目标摧毁：清除跟踪状态
        t.destroyed = True
        t.tracked = False
        t.tracker_id = NO_UAV

        # 释放跟踪无人机（恢复为空闲状态，可供重新分配）
        u = self.uavs[released_uid]
        if u.alive:
            u.task = TaskType.IDLE
            u.target_id = NO_TARGET

        return Event(
            EventType.TARGET_DESTROYED,
            affected_regions=[],
            released_uav=released_uid,
            description=f"T{tid} destroyed; U{released_uid} released",
        )

    # =========================
    # 动作执行
    # =========================

    def _execute_action(self, action):
        """执行修复后的联合动作到环境中。

        对每个区域的动作进行解释：
          - KEEP:   不做任何改变
          - NO_UAV: 清除该区域分配
          - U0~U3:  将区域分配给指定无人机

        Args:
            action: np.ndarray shape=(4,)，已修复的合法动作
        """
        for rid in range(NUM_REGIONS):
            code = int(action[rid])

            if code == int(ActionCode.KEEP):
                # 保持当前分配，不操作
                continue

            if code == int(ActionCode.NO_UAV):
                # 清除该区域分配
                self._clear_region_assignment(rid)
                self.regions[rid].need_reassign = True
                continue

            # U0~U3: 动作码 1~4 对应无人机 0~3
            new_uid = code - 1
            self._assign_region_to_uav(rid, new_uid)

        # 更新状态：无区域的搜索无人机设为 IDLE
        for u in self.uavs.values():
            if u.alive and u.task == TaskType.SEARCH and len(u.regions) == 0:
                u.task = TaskType.IDLE

    def _assign_region_to_uav(self, rid: int, uid: int):
        """将区域分配给指定无人机。

        自动处理：
          - 解除旧分配关系（从旧无人机的 regions 中移除）
          - 建立新分配关系
          - 将新无人机任务设置为 SEARCH

        Args:
            rid: 区域编号
            uid: 无人机编号
        """
        old_uid = self.regions[rid].assigned_uav

        # 从旧无人机中移除该区域
        if old_uid != NO_UAV and old_uid in self.uavs:
            self.uavs[old_uid].regions.discard(rid)

        # 建立新分配
        self.regions[rid].assigned_uav = uid
        self.regions[rid].need_reassign = False

        # 只有从空闲/被逼退状态重新上岗时才移动位置
        # 已有区域的无人机接管新区时原地不动（箭头指向新区即可）
        was_idle = (self.uavs[uid].task != TaskType.SEARCH or len(self.uavs[uid].regions) == 0)

        self.uavs[uid].regions.add(rid)
        self.uavs[uid].task = TaskType.SEARCH
        self.uavs[uid].target_id = NO_TARGET

        if was_idle:
            r = self.regions[rid]
            self.uavs[uid].x = r.center_x
            self.uavs[uid].y = r.center_y

    def _clear_region_assignment(self, rid: int):
        """清除区域的无人机分配。

        将区域置为未分配状态，并从对应无人机的区域集合中移除。
        如果该无人机没有其他搜索区域，将其任务设为 IDLE。

        Args:
            rid: 要清除分配的区域编号
        """
        old_uid = self.regions[rid].assigned_uav
        if old_uid != NO_UAV and old_uid in self.uavs:
            u = self.uavs[old_uid]
            u.regions.discard(rid)
            # 如果无人机位置还在被清掉的区域中心，移到它还有的另一区域
            if len(u.regions) > 0:
                r_cur = self.regions[rid]
                if abs(u.x - r_cur.center_x) < 1e-6 and abs(u.y - r_cur.center_y) < 1e-6:
                    another = next(iter(u.regions))
                    r_new = self.regions[another]
                    u.x = r_new.center_x
                    u.y = r_new.center_y
            # 如果无人机不再有搜索区域，设为空闲
            elif u.task == TaskType.SEARCH:
                u.task = TaskType.IDLE
        self.regions[rid].assigned_uav = NO_UAV

    def _valid_search_assign(self, uid: int, rid: int) -> bool:
        """判断无人机 uid 是否能合法搜索区域 rid。

        合法性条件（全部满足才视为合法）：
          1. uid 不是 NO_UAV
          2. 无人机存活
          3. 无人机不在跟踪任务中（跟踪中的无人机已锁定目标）
          4. 传感器未故障
          （SAR 全天候，不再检查天气-传感器兼容性）

        Args:
            uid: 无人机编号
            rid: 区域编号

        Returns:
            bool: 分配是否合法
        """
        if uid == NO_UAV:
            return False
        u = self.uavs[uid]
        r = self.regions[rid]
        if not u.alive:
            return False
        if u.task == TaskType.TRACK:
            return False
        if u.sensor_failed:
            return False
        # if u.sensor == SensorType.EO and r.weather == Weather.RAINY:  # 已注释：全部使用 SAR
        #     return False
        return True

    def _all_regions_valid(self) -> bool:
        """检查是否所有区域都已被合法分配。

        所有 NUM_REGIONS 个区域都有合法且存活无人机的搜索分配，
        且没有 need_reassign 标记，表示当前态势完全稳定。

        Returns:
            bool: 所有区域均已合法分配
        """
        for rid in range(NUM_REGIONS):
            region = self.regions[rid]
            uid = region.assigned_uav
            if uid == NO_UAV:
                return False
            if not self._valid_search_assign(uid, rid):
                return False
            if region.need_reassign:
                return False
        return True

    def _event_success(self) -> bool:
        """判断当前事件是否被成功解决。

        - TARGET_DESTROYED: 被释放的无人机已被重新分配到搜索任务
        - 其他事件类型: 所有受影响区域都已重新分配

        Returns:
            bool: 事件是否已成功解决
        """
        e = self.current_event
        if e.event_type == EventType.TARGET_DESTROYED:
            uid = e.released_uav
            return uid != NO_UAV and self.uavs[uid].task == TaskType.SEARCH and len(self.uavs[uid].regions) > 0

        return all(self.regions[rid].assigned_uav != NO_UAV for rid in e.affected_regions)

    # =========================
    # 快照 / 导出接口
    # =========================

    def snapshot(self) -> Dict[str, Any]:
        """导出当前环境完整状态快照。

        用于：
          - 可视化（before/after 对比）
          - 日志记录
          - 与外部仿真器对接

        Returns:
            Dict: 包含 regions、uavs、targets、event 四个键的字典
        """
        return {
            "regions": {
                str(rid): {
                    "center_x": r.center_x,
                    "center_y": r.center_y,
                    "weather": int(r.weather),
                    "assigned_uav": int(r.assigned_uav),
                    "need_reassign": bool(r.need_reassign),
                }
                for rid, r in self.regions.items()
            },
            "uavs": {
                str(uid): {
                    "x": u.x,
                    "y": u.y,
                    "sensor": int(u.sensor),
                    "alive": bool(u.alive),
                    "sensor_failed": bool(u.sensor_failed),
                    "task": int(u.task),
                    "regions": sorted(list(u.regions)),
                    "target_id": int(u.target_id),
                }
                for uid, u in self.uavs.items()
            },
            "targets": {
                str(tid): {
                    "x": t.x,
                    "y": t.y,
                    "region": int(t.region),
                    "target_type": int(t.target_type),
                    "movable": bool(t.movable),
                    "discovered": bool(t.discovered),
                    "tracked": bool(t.tracked),
                    "destroyed": bool(t.destroyed),
                    "tracker_id": int(t.tracker_id),
                }
                for tid, t in self.targets.items()
            },
            "event": {
                "event_type": int(self.current_event.event_type),
                "affected_regions": list(self.current_event.affected_regions),
                "released_uav": int(self.current_event.released_uav),
                "damaged_uav": int(self.current_event.damaged_uav),
                "weather_disabled_uav": int(self.current_event.weather_disabled_uav),
                "description": self.current_event.description,
            },
        }

    def export_assignment_json(self) -> Dict[str, Any]:
        """导出分配结果为标准 JSON 格式。

        供下游路径规划模块或其他外部系统使用。
        输出格式与 README 中定义的接口一致。

        Returns:
            Dict: 包含 region_assignments 和 uav_tasks 的字典
        """
        # 区域分配映射: {"R0": "U1", "R1": "U2", ...}
        region_assignments = {
            f"R{rid}": (None if r.assigned_uav == NO_UAV else f"U{r.assigned_uav}")
            for rid, r in self.regions.items()
        }

        # 无人机任务详情
        uav_tasks = {}
        for uid, u in self.uavs.items():
            task_name = "SEARCH" if u.task == TaskType.SEARCH else ("TRACK" if u.task == TaskType.TRACK else "IDLE")
            regions = [f"R{rid}" for rid in sorted(u.regions)]
            target_points = [
                [self.regions[rid].center_x, self.regions[rid].center_y]
                for rid in sorted(u.regions)
            ]
            uav_tasks[f"U{uid}"] = {
                "alive": bool(u.alive),
                "task": task_name,
                "sensor": "EO" if u.sensor == SensorType.EO else "SAR",
                "regions": regions,
                "target_id": None if u.target_id == NO_TARGET else f"T{u.target_id}",
                "target_points": target_points,
            }

        return {
            "region_assignments": region_assignments,
            "uav_tasks": uav_tasks,
        }
