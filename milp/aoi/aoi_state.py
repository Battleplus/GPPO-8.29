"""
AOI 状态数据结构。

对外接口：
  - AoiInfo           — 单个 AOI 的静态描述
  - ExecutionFeedback — 底层规控返回的 AOI 执行反馈
  - AOIRouteState     — AOI 排序结果与当前执行进度
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class AoiInfo:
    """
    单个 AOI 的描述信息（由态势理解模块提供）。

    Attributes:
        id:            AOI 唯一标识，如 "A_5_6"
        row:           AOI 行号 (1..6)
        col:           AOI 列号 (1..6)
        priority:      指挥员优先级 ∈ [0, 1]，越高越重要
        target_prior:  AOI 内目标存在先验概率 ∈ [0, 1]
        target_value:  AOI 内目标平均价值 ∈ [0, 1]
        target_threat: AOI 内目标平均威胁度 ∈ [0, 1]
    """
    id: str
    row: int
    col: int
    priority: float = 1.0
    target_prior: float = 0.25
    target_value: float = 0.5
    target_threat: float = 0.5


@dataclass
class ExecutionFeedback:
    """
    底层规控对当前 AOI 的执行反馈。

    Attributes:
        aoi_id:            当前正在执行的 AOI 标识
        aoi_status:        执行状态，"RUNNING" | "FINISHED" | "ABORTED"
        coverage_rate:     AOI 侦察覆盖率 ∈ [0, 1]
        finished_tasks:    已完成任务 ID 列表
        detected_targets:  已探测目标 ID 列表
        destroyed_targets: 已摧毁目标 ID 列表
        elapsed_time:      已耗时（仿真时间单位）
    """
    aoi_id: str
    aoi_status: str  # "RUNNING" | "FINISHED" | "ABORTED"
    coverage_rate: float = 0.0
    finished_tasks: List[str] = field(default_factory=list)
    detected_targets: List[str] = field(default_factory=list)
    destroyed_targets: List[str] = field(default_factory=list)
    elapsed_time: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionFeedback":
        """从字典反序列化（最小必要字段：aoi_id, aoi_status）。"""
        return cls(
            aoi_id=d["aoi_id"],
            aoi_status=d["aoi_status"],
            coverage_rate=d.get("coverage_rate", 0.0),
            finished_tasks=d.get("finished_tasks", []),
            detected_targets=d.get("detected_targets", []),
            destroyed_targets=d.get("destroyed_targets", []),
            elapsed_time=d.get("elapsed_time", 0.0),
        )


@dataclass
class AOIRouteState:
    """
    AOI 排序结果与当前执行进度。

    由任务分配模块在首次调用时生成，之后由上层（态势理解 / 任务管理）
    持久保存，并在每次调用时原样带回。

    Attributes:
        aoi_sequence:       AOI 执行顺序，如 ["A_5_6", "A_3_4", "A_1_5"]
        current_aoi_index:  当前执行的 AOI 在序列中的下标（从 0 开始）
        route_status:       整体状态，"RUNNING" | "ALL_FINISHED"
    """
    aoi_sequence: List[str]
    current_aoi_index: int
    route_status: str  # "RUNNING" | "ALL_FINISHED"

    # ── 派生属性（只读） ──────────────────────────────────
    @property
    def current_aoi(self) -> Optional[str]:
        """当前应执行的 AOI ID；若全部完成则为 None。"""
        if self.current_aoi_index < len(self.aoi_sequence):
            return self.aoi_sequence[self.current_aoi_index]
        return None

    @property
    def next_aoi(self) -> Optional[str]:
        """下一个 AOI ID；若当前已是最后一个则为 None。"""
        next_idx = self.current_aoi_index + 1
        if next_idx < len(self.aoi_sequence):
            return self.aoi_sequence[next_idx]
        return None

    def is_finished(self) -> bool:
        """判断所有 AOI 是否已执行完毕。"""
        return self.current_aoi_index >= len(self.aoi_sequence)

    def advance(self) -> None:
        """将当前 AOI 标记为完成，索引前进 1。"""
        self.current_aoi_index += 1
        if self.is_finished():
            self.route_status = "ALL_FINISHED"

    # ── 序列化 / 反序列化 ────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "aoi_sequence": self.aoi_sequence,
            "current_aoi_index": self.current_aoi_index,
            "current_aoi": self.current_aoi,
            "next_aoi": self.next_aoi,
            "route_status": self.route_status,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AOIRouteState":
        return cls(
            aoi_sequence=d["aoi_sequence"],
            current_aoi_index=d["current_aoi_index"],
            route_status=d["route_status"],
        )
