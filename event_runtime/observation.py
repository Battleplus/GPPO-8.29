"""Observation models and deterministic weak-communication simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping

from .events import EventType, _canonical_value, canonical_json_bytes


@dataclass(frozen=True)
class Observation:
    observation_id: str
    event_id: str
    event_type: EventType
    source_event: str
    source_id: str
    source_type: str
    signal_type: str
    sequence: int
    confidence: float
    positive: bool
    emitted_at: float
    received_at: float
    occurred_at: float | None = None
    affected_uavs: tuple[str, ...] = ()
    affected_regions: tuple[str, ...] = ()
    affected_targets: tuple[str, ...] = ()
    severity: float = 1.0
    event_seed: int = 0
    state_version: int = 0
    is_duplicate: bool = False
    duplicate_of: str | None = None
    is_false_positive: bool = False
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.received_at < self.emitted_at:
            raise ValueError("received_at cannot precede emitted_at")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Observation":
        return cls(
            observation_id=str(value["observation_id"]),
            event_id=str(value["event_id"]),
            event_type=EventType(value["event_type"]),
            source_event=str(value["source_event"]),
            source_id=str(value["source_id"]),
            source_type=str(value["source_type"]),
            signal_type=str(value["signal_type"]),
            sequence=int(value["sequence"]),
            confidence=float(value["confidence"]),
            positive=bool(value["positive"]),
            emitted_at=float(value["emitted_at"]),
            received_at=float(value["received_at"]),
            occurred_at=None if value.get("occurred_at") is None else float(value["occurred_at"]),
            affected_uavs=tuple(value.get("affected_uavs", ())),
            affected_regions=tuple(value.get("affected_regions", ())),
            affected_targets=tuple(value.get("affected_targets", ())),
            severity=float(value.get("severity", 1.0)),
            event_seed=int(value.get("event_seed", 0)),
            state_version=int(value.get("state_version", 0)),
            is_duplicate=bool(value.get("is_duplicate", False)),
            duplicate_of=value.get("duplicate_of"),
            is_false_positive=bool(value.get("is_false_positive", False)),
            payload=dict(value.get("payload", {})),
        )


@dataclass(frozen=True)
class NetworkPartition:
    start_at: float
    end_at: float
    behavior: str = "delay_until_recovery"
    affected_uavs: tuple[str, ...] = ()

    def contains(self, timestamp: float) -> bool:
        return self.start_at <= timestamp < self.end_at


@dataclass(frozen=True)
class WeakCommunicationProfile:
    name: str = "nominal"
    delay_min: float = 0.02
    delay_max: float = 0.20
    packet_loss_rate: float = 0.05
    duplicate_rate: float = 0.05
    false_positive_rate: float = 0.02
    false_negative_rate: float = 0.05
    reorder_jitter: float = 0.30
    heartbeat_interval: float = 1.0
    heartbeat_miss_threshold: int = 3
    target_confirmation_count: int = 3
    destruction_confirmation_count: int = 2
    partitions: tuple[NetworkPartition, ...] = ()

    def __post_init__(self) -> None:
        if self.delay_min < 0 or self.delay_max < self.delay_min:
            raise ValueError("invalid observation delay range")
        for rate in (
            self.packet_loss_rate,
            self.duplicate_rate,
            self.false_positive_rate,
            self.false_negative_rate,
        ):
            if not 0.0 <= rate <= 1.0:
                raise ValueError("communication rates must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))


@dataclass(frozen=True)
class ObservationTape:
    tape_id: str
    truth_tape_sha256: str
    initial_seed: int
    event_seed: int
    profile: Mapping[str, Any]
    observations: tuple[Observation, ...]
    schema_version: int = 1

    def delivery_order(self) -> tuple[Observation, ...]:
        return tuple(sorted(
            self.observations,
            key=lambda item: (item.received_at, item.emitted_at, item.observation_id),
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tape_id": self.tape_id,
            "truth_tape_sha256": self.truth_tape_sha256,
            "initial_seed": self.initial_seed,
            "event_seed": self.event_seed,
            "profile": _canonical_value(self.profile),
            "observations": [item.to_dict() for item in self.observations],
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
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservationTape":
        return cls(
            tape_id=str(value["tape_id"]),
            truth_tape_sha256=str(value["truth_tape_sha256"]),
            initial_seed=int(value["initial_seed"]),
            event_seed=int(value["event_seed"]),
            profile=dict(value.get("profile", {})),
            observations=tuple(Observation.from_dict(item) for item in value.get("observations", ())),
            schema_version=int(value.get("schema_version", 1)),
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> "ObservationTape":
        return cls.from_dict(json.loads(value.decode("utf-8")))

    @classmethod
    def build(
        cls,
        *,
        tape_id: str,
        truth_tape_sha256: str,
        initial_seed: int,
        event_seed: int,
        profile: WeakCommunicationProfile,
        observations: Iterable[Observation],
    ) -> "ObservationTape":
        return cls(
            tape_id=tape_id,
            truth_tape_sha256=truth_tape_sha256,
            initial_seed=initial_seed,
            event_seed=event_seed,
            profile=profile.to_dict(),
            observations=tuple(observations),
        )
