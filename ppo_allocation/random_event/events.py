"""可重放的随机事件数据模型。

这里不复用旧 ``env.event.Event``：旧模型没有
``TARGET_DISCOVERED`` 类型，也没有事件时间、随机种子和图版本。
字符串枚举还保证事件带 JSON 与 Python 枚举整数编号解耦。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Mapping, Sequence


class RandomEventType(str, Enum):
    """论文实验使用的四类外生事件。"""

    UAV_DAMAGE = "UAV_DAMAGE"
    TARGET_DISCOVERED = "TARGET_DISCOVERED"
    TARGET_DESTROYED = "TARGET_DESTROYED"
    REGION_VACANCY = "REGION_VACANCY"


def _json_value(value: Any) -> Any:
    """递归转成可稳定 JSON 序列化的值。"""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported event payload value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class RandomEvent:
    """一条可完整回放的事件记录。

    ``occurred_at`` 是真实发生时间，``observed_at`` 是弱通信下系统
    实际观测到它的时间。``source_event`` 保留因果来源，例如目标
    发现后导致区域空缺，而不把事件类型偷换成区域空缺。
    """

    event_id: str
    event_type: RandomEventType
    occurred_at: float
    observed_at: float
    source_event: str
    affected_uavs: tuple[int, ...] = field(default_factory=tuple)
    affected_regions: tuple[int, ...] = field(default_factory=tuple)
    affected_targets: tuple[int, ...] = field(default_factory=tuple)
    severity: float = 1.0
    payload: Mapping[str, Any] = field(default_factory=dict)
    event_seed: int = 0
    state_version: int = 0

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if self.observed_at < self.occurred_at:
            raise ValueError("observed_at must be greater than or equal to occurred_at")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")
        if self.state_version < 0:
            raise ValueError("state_version must be non-negative")

        # frozen dataclass 中做一次规范化，避免调用方传入 list 影响稳定性。
        object.__setattr__(self, "event_type", RandomEventType(self.event_type))
        object.__setattr__(self, "affected_uavs", tuple(sorted(set(self.affected_uavs))))
        object.__setattr__(self, "affected_regions", tuple(sorted(set(self.affected_regions))))
        object.__setattr__(self, "affected_targets", tuple(sorted(set(self.affected_targets))))
        object.__setattr__(self, "payload", _json_value(dict(self.payload)))

    @property
    def type(self) -> RandomEventType:
        """与协议字段名 ``type`` 对齐的只读别名。"""

        return self.event_type

    def to_dict(self) -> dict[str, Any]:
        """转成字段顺序固定的协议字典。"""

        return {
            "event_id": self.event_id,
            "type": self.event_type.value,
            "occurred_at": self.occurred_at,
            "observed_at": self.observed_at,
            "source_event": self.source_event,
            "affected_uavs": list(self.affected_uavs),
            "affected_regions": list(self.affected_regions),
            "affected_targets": list(self.affected_targets),
            "severity": self.severity,
            "payload": _json_value(self.payload),
            "event_seed": self.event_seed,
            "state_version": self.state_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RandomEvent":
        """从协议字典恢复，同时兼容 ``event_type`` 键。"""

        event_type = data.get("type", data.get("event_type"))
        if event_type is None:
            raise ValueError("event record is missing 'type'")
        if isinstance(event_type, RandomEventType):
            event_type = event_type.value
        return cls(
            event_id=str(data["event_id"]),
            event_type=RandomEventType(str(event_type)),
            occurred_at=float(data["occurred_at"]),
            observed_at=float(data["observed_at"]),
            source_event=str(data.get("source_event", "")),
            affected_uavs=tuple(int(value) for value in data.get("affected_uavs", ())),
            affected_regions=tuple(int(value) for value in data.get("affected_regions", ())),
            affected_targets=tuple(int(value) for value in data.get("affected_targets", ())),
            severity=float(data.get("severity", 1.0)),
            payload=dict(data.get("payload", {})),
            event_seed=int(data.get("event_seed", 0)),
            state_version=int(data.get("state_version", 0)),
        )


@dataclass(frozen=True, slots=True)
class EventTape:
    """可作为训练/评估输入的不可变事件带。"""

    initial_seed: int
    event_seed: int
    mode: str
    events: tuple[RandomEvent, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "initial_seed": self.initial_seed,
            "event_seed": self.event_seed,
            "mode": self.mode,
            "events": [event.to_dict() for event in self.events],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """生成字节稳定的 UTF-8 JSON 文本。"""

        separators = (",", ":") if indent is None else None
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=separators,
            indent=indent,
        )

    def to_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventTape":
        return cls(
            initial_seed=int(data["initial_seed"]),
            event_seed=int(data["event_seed"]),
            mode=str(data["mode"]),
            events=tuple(RandomEvent.from_dict(item) for item in data.get("events", ())),
            schema_version=int(data.get("schema_version", 1)),
        )

    @classmethod
    def from_json(cls, content: str | bytes | bytearray) -> "EventTape":
        if isinstance(content, (bytes, bytearray)):
            content = bytes(content).decode("utf-8")
        return cls.from_dict(json.loads(content))


def canonical_event_json(events: Sequence[RandomEvent] | EventTape) -> bytes:
    """便于回放一致性测试的规范 JSON 字节串。"""

    if isinstance(events, EventTape):
        return events.to_bytes()
    return json.dumps(
        [event.to_dict() for event in events],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


# 较短的兼容别名，避免与旧 config.EventType 混淆。
EventKind = RandomEventType
