"""Canonical event data models and byte-stable truth tapes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping


class EventType(str, Enum):
    UAV_DAMAGE = "UAV_DAMAGE"
    TARGET_DISCOVERED = "TARGET_DISCOVERED"
    TARGET_DESTROYED = "TARGET_DESTROYED"
    REGION_VACANCY = "REGION_VACANCY"


class ConfirmationStatus(str, Enum):
    NORMAL = "NORMAL"
    SUSPECTED = "SUSPECTED"
    CONFIRMED = "CONFIRMED"
    RECOVERING = "RECOVERING"
    RESOLVED = "RESOLVED"
    FALSE_ALARM = "FALSE_ALARM"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    data = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    return json.dumps(
        _canonical_value(data),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class TruthEvent:
    event_id: str
    event_type: EventType
    source_event: str
    affected_uavs: tuple[str, ...] = ()
    affected_regions: tuple[str, ...] = ()
    affected_targets: tuple[str, ...] = ()
    severity: float = 1.0
    payload: Mapping[str, Any] = field(default_factory=dict)
    event_seed: int = 0
    state_version: int = 0
    occurred_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id is required")
        if self.occurred_at < 0:
            raise ValueError("occurred_at must be non-negative")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TruthEvent":
        return cls(
            event_id=str(value["event_id"]),
            event_type=EventType(value["event_type"]),
            source_event=str(value["source_event"]),
            affected_uavs=tuple(value.get("affected_uavs", ())),
            affected_regions=tuple(value.get("affected_regions", ())),
            affected_targets=tuple(value.get("affected_targets", ())),
            severity=float(value.get("severity", 1.0)),
            payload=dict(value.get("payload", {})),
            event_seed=int(value.get("event_seed", 0)),
            state_version=int(value.get("state_version", 0)),
            occurred_at=float(value.get("occurred_at", 0.0)),
        )


@dataclass
class ConfirmedEvent:
    event_id: str
    event_type: EventType
    source_event: str
    affected_uavs: tuple[str, ...] = ()
    affected_regions: tuple[str, ...] = ()
    affected_targets: tuple[str, ...] = ()
    severity: float = 1.0
    payload: dict[str, Any] = field(default_factory=dict)
    event_seed: int = 0
    state_version: int = 0
    occurred_at: float = 0.0
    emitted_at: float | None = None
    received_at: float | None = None
    suspected_at: float | None = None
    confirmed_at: float | None = None
    resolved_at: float | None = None
    status: ConfirmationStatus = ConfirmationStatus.CONFIRMED
    evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))

    def transition(self, status: ConfirmationStatus, *, at: float | None = None) -> None:
        allowed = {
            ConfirmationStatus.CONFIRMED: {
                ConfirmationStatus.RECOVERING,
                ConfirmationStatus.REVOKED,
                ConfirmationStatus.EXPIRED,
            },
            ConfirmationStatus.RECOVERING: {
                ConfirmationStatus.RESOLVED,
                ConfirmationStatus.CONFIRMED,
                ConfirmationStatus.REVOKED,
                ConfirmationStatus.EXPIRED,
            },
            ConfirmationStatus.RESOLVED: {ConfirmationStatus.NORMAL},
        }
        if status not in allowed.get(self.status, set()):
            raise ValueError(f"invalid confirmed-event transition {self.status.value}->{status.value}")
        self.status = status
        if status is ConfirmationStatus.RESOLVED:
            self.resolved_at = at


@dataclass(frozen=True)
class TruthEventTape:
    tape_id: str
    initial_seed: int
    event_seed: int
    mode: str
    initial_snapshot_hash: str
    events: tuple[TruthEvent, ...]
    schema_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tape_id": self.tape_id,
            "initial_seed": self.initial_seed,
            "event_seed": self.event_seed,
            "mode": self.mode,
            "initial_snapshot_hash": self.initial_snapshot_hash,
            "events": [event.to_dict() for event in self.events],
            "metadata": _canonical_value(self.metadata),
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TruthEventTape":
        return cls(
            tape_id=str(value["tape_id"]),
            initial_seed=int(value["initial_seed"]),
            event_seed=int(value["event_seed"]),
            mode=str(value["mode"]),
            initial_snapshot_hash=str(value["initial_snapshot_hash"]),
            events=tuple(TruthEvent.from_dict(item) for item in value.get("events", ())),
            schema_version=int(value.get("schema_version", 1)),
            metadata=dict(value.get("metadata", {})),
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> "TruthEventTape":
        return cls.from_dict(json.loads(value.decode("utf-8")))

    @classmethod
    def build(
        cls,
        *,
        tape_id: str,
        initial_seed: int,
        event_seed: int,
        mode: str,
        initial_snapshot_hash: str,
        events: Iterable[TruthEvent],
        metadata: Mapping[str, Any] | None = None,
    ) -> "TruthEventTape":
        return cls(
            tape_id=tape_id,
            initial_seed=initial_seed,
            event_seed=event_seed,
            mode=mode,
            initial_snapshot_hash=initial_snapshot_hash,
            events=tuple(events),
            metadata=metadata or {},
        )
