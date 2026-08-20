"""全局配置模块。

定义所有场景参数、枚举类型、观测维度、奖励权重等核心常量。
所有模块均从此处导入配置，避免魔法数字分散在代码各处。
"""

from enum import IntEnum
from pathlib import Path
import math


# =========================
# 基础场景参数
# =========================

# 地图总边长（正方形区域，单位：km 或归一化单位）
AREA_SIZE = 50.0

# 无人机、区域、目标的数量
NUM_UAVS = 4
NUM_REGIONS = 4
NUM_TARGETS = 3

# 场景中对角线最大距离，用于归一化距离代价
MAX_DISTANCE = math.sqrt(AREA_SIZE ** 2 + AREA_SIZE ** 2)

# 每个 episode 的最大决策步数（PPO 每次决策对应一步），超过后被截断
MAX_DECISION_STEPS = 50

# 全局随机种子，保证实验可复现
RANDOM_SEED = 42

# 结果输出目录配置
RESULTS_DIR = Path("results")        # 总结果根目录
MODELS_DIR = RESULTS_DIR / "models"  # 模型保存目录
LOGS_DIR = RESULTS_DIR / "logs"      # TensorBoard 日志目录
FIGURES_DIR = RESULTS_DIR / "figures" # 图表保存目录
EVAL_DIR = RESULTS_DIR / "eval"      # 评估结果/JSON/GIF 输出目录


# =========================
# 观测空间维度定义
# =========================
# 所有特征拼接为一个一维向量输入 PPO 网络，
# 总维度 = UAV特征 + 区域特征 + 目标特征 + 事件特征

# 单架无人机特征维度：
#   alive(1) + x(1) + y(1) + is_EO(1) + is_SAR(1) +
#   task_SEARCH(1) + task_TRACK(1) + task_IDLE(1) +
#   区域multi-hot(4) + 目标one-hot(4) + 负载比例(1) = 17
UAV_FEATURE_DIM = 17

# 单个区域特征维度：
#   center_x(1) + center_y(1) + is_SUNNY(1) + is_RAINY(1) +
#   is_assigned(1) + 分配one-hot(5) = 10
REGION_FEATURE_DIM = 10

# 单个目标特征维度：
#   is_CAR(1) + is_COMMAND(1) + discovered(1) + tracked(1) + destroyed(1) +
#   known_x(1) + known_y(1) + 区域one-hot(4) + 跟踪者one-hot(4) = 15
TARGET_FEATURE_DIM = 15

# 事件特征维度：
#   事件类型one-hot(4) + 影响区域multi-hot(4) + 相关UAV one-hot(4) = 12
EVENT_FEATURE_DIM = 12

# 拼接后的总观测维度
OBS_DIM = (
    NUM_UAVS * UAV_FEATURE_DIM
    + NUM_REGIONS * REGION_FEATURE_DIM
    + NUM_TARGETS * TARGET_FEATURE_DIM
    + EVENT_FEATURE_DIM
)


# =========================
# 动作空间定义
# =========================
# 动作为 MultiDiscrete([6, 6, 6, 6])，表示每个区域 R0~R3 的分配动作
# 每个区域有 6 种离散选择：

class ActionCode(IntEnum):
    """区域动作编码。

    KEEP:   保持当前分配不变
    U0~U3:  将区域分配给对应编号的无人机
    NO_UAV: 清除区域分配（无人值守）
    """
    KEEP = 0
    U0 = 1
    U1 = 2
    U2 = 3
    U3 = 4
    NO_UAV = 5


# 每个区域的可选动作数量
N_ACTIONS_PER_REGION = 6

# MultiDiscrete 的 nvec 参数：[6, 6, 6, 6]
ACTION_N_VEC = [N_ACTIONS_PER_REGION] * NUM_REGIONS


# =========================
# 场景枚举类型
# =========================

class Weather(IntEnum):
    """天气类型。

    SUNNY: 晴天
    RAINY: 雨天
    注意：当前全部使用 SAR 传感器，SAR 全天候工作，天气不再影响传感器有效性。
    """
    SUNNY = 0
    RAINY = 1


class SensorType(IntEnum):
    """传感器类型。

    EO:  光电传感器（已弃用，全部改用 SAR）
    SAR: 合成孔径雷达（全天候，不受天气影响）
    """
    EO = 0
    SAR = 1


class TaskType(IntEnum):
    """无人机当前任务类型。

    SEARCH: 搜索任务（PPO 的分配范围）
    TRACK:  跟踪已发现目标（由规则触发，不由 PPO 决策）
    IDLE:   空闲（无任务可执行）
    """
    SEARCH = 0
    TRACK = 1
    IDLE = 2


class TargetType(IntEnum):
    """目标类型。

    CAR:     普通车辆（可移动）
    COMMAND: 指挥所（固定不动）
    """
    CAR = 0
    COMMAND = 1


class EventType(IntEnum):
    """突发事件类型，触发 PPO 局部重分配。

    REGION_VACANCY:     区域空缺（无人机离开、目标被发现转入追踪等）
    UAV_DAMAGE:         无人机损毁
    TARGET_DESTROYED:   目标被摧毁，追踪无人机被释放
    """
    REGION_VACANCY = 0
    # WEATHER_INVALID = 1   # 已注释：全部使用 SAR 传感器，天气不影响传感器有效性
    UAV_DAMAGE = 2
    TARGET_DESTROYED = 3


# 哨兵值：表示"无无人机"或"无目标"
NO_UAV = -1
NO_TARGET = -1


# =========================
# 区域几何划分
# =========================
# 将 50x50 地图等分为 4 个 25x25 的子区域：
#   R0: 左上  (x: 0~25,  y: 25~50)
#   R1: 右上  (x: 25~50, y: 25~50)
#   R2: 左下  (x: 0~25,  y: 0~25)
#   R3: 右下  (x: 25~50, y: 0~25)

REGION_BOUNDS = {
    0: (0.0, 25.0, 25.0, 50.0),   # (xmin, xmax, ymin, ymax)
    1: (25.0, 50.0, 25.0, 50.0),
    2: (0.0, 25.0, 0.0, 25.0),
    3: (25.0, 50.0, 0.0, 25.0),
}

# 从边界自动计算各区域中心点坐标
REGION_CENTERS = {
    rid: ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
    for rid, (xmin, xmax, ymin, ymax) in REGION_BOUNDS.items()
}

# 区域邻接关系（4-邻接，共享边才算相邻，对角不算）
#    R0(左上) — R1(右上)
#      |          |
#    R2(左下) — R3(右下)
REGION_ADJACENCY = {
    0: {1, 2},
    1: {0, 3},
    2: {0, 3},
    3: {1, 2},
}


# =========================
# 奖励函数权重
# =========================
# 总奖励 = 各项加权求和，权重控制各部分的重要性。
# 调整原则：让 PPO 以"成功解决事件"为主要目标，
# 避免结构性惩罚主导奖励信号。

W_REGION_ASSIGNED = 5.0    # 区域被合法分配的奖励（每个区域）
W_UNASSIGNED = 4.0         # 未分配区域的惩罚（每个区域），降低避免过度惩罚
W_DISTANCE = 10.0          # 距离代价权重（归一化到 AREA_SIZE=50）
W_BALANCE = 1.0            # 负载均衡惩罚权重
W_SENSOR_MATCH = 0.0       # 传感器与天气匹配奖励
W_IDLE_UAV = 2.0           # 闲置可分配无人机的轻微惩罚（每架）
W_EVENT_SUCCESS = 10.0     # 事件成功解决的奖励，大幅提高使其成为主导信号
W_SWITCH = 0.0             # 切换代价，设为0消除模型"怕动"心理
W_INVALID = 0.0            # 非法动作由 mask 屏蔽，不额外惩罚

# 额外奖励
W_ALL_VALID_BONUS = 0.0   # 所有区域均合法分配时的 episode 终止奖励
W_TERMINAL_BONUS = 0.0    # episode 提前终止奖励（所有区域合法 + 无 pending 事件）


# =========================
# 事件生成控制
# =========================

# 事件间隔：每隔 N 步才可能生成新事件，给 PPO 稳定期观察决策效果
EVENT_INTERVAL = 10


# =========================
# 训练阶段随机事件概率
# =========================
# 用于 _generate_next_event() 中的加权抽样。
# 训练时使用随机事件，实际应用时可由完整仿真器注入真实事件。

EVENT_PROBS = {
    EventType.REGION_VACANCY: 0.34,
    EventType.UAV_DAMAGE: 0.33,
    # EventType.WEATHER_INVALID: 0.25,  # 已注释：SAR 全天候，天气不影响传感器
    EventType.TARGET_DESTROYED: 0.33,
    # 目标发现事件（内部以 REGION_VACANCY 类型触发，
    # 因为发现后该无人机原来负责的搜索区域产生空缺）
}
