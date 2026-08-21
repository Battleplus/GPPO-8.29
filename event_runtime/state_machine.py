"""Idempotent event confirmation state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .events import ConfirmationStatus, ConfirmedEvent, EventType
from .observation import Observation


@dataclass
class ConfirmationRecord:
    event_id: str
    event_type: EventType
    status: ConfirmationStatus = ConfirmationStatus.NORMAL
    occurred_at: float | None = None
    suspected_at: float | None = None
    confirmed_at: float | None = None
    resolved_at: float | None = None
    observations: list[Observation] = field(default_factory=list)
    positive_evidence_keys: set[str] = field(default_factory=set)
    last_sequence_by_source: dict[str, int] = field(default_factory=dict)
    confirmed_event: ConfirmedEvent | None = None


@dataclass(frozen=True)
class StateMachineResult:
    accepted: bool
    duplicate: bool = False
    late: bool = False
    status_before: ConfirmationStatus = ConfirmationStatus.NORMAL
    status_after: ConfirmationStatus = ConfirmationStatus.NORMAL
    confirmed_event: ConfirmedEvent | None = None


class ConfirmationStateMachine:
    def __init__(
        self,
        *,
        heartbeat_miss_threshold: int = 3,
        target_confirmation_count: int = 3,
        destruction_confirmation_count: int = 2,
        suspicion_timeout: float = 5.0,
    ) -> None:
        self.heartbeat_miss_threshold = int(heartbeat_miss_threshold)
        self.target_confirmation_count = int(target_confirmation_count)
        self.destruction_confirmation_count = int(destruction_confirmation_count)
        self.suspicion_timeout = float(suspicion_timeout)
        self.records: dict[str, ConfirmationRecord] = {}
        self._processed_evidence: set[str] = set()
        self.false_alarm_count = 0
        self.duplicate_observation_count = 0
        self.late_observation_count = 0

    def get(self, event_id: str) -> ConfirmationRecord | None:
        return self.records.get(event_id)

    @staticmethod
    def _evidence_key(observation: Observation) -> str:
        return observation.duplicate_of or observation.observation_id

    def _record_for(self, observation: Observation) -> ConfirmationRecord:
        return self.records.setdefault(
            observation.event_id,
            ConfirmationRecord(
                event_id=observation.event_id,
                event_type=observation.event_type,
                occurred_at=observation.occurred_at,
            ),
        )

    def _clear_suspicions_for_healthy(self, observation: Observation) -> bool:
        cleared = False
        affected = set(observation.affected_uavs)
        for record in self.records.values():
            if record.status is not ConfirmationStatus.SUSPECTED:
                continue
            record_uavs = {
                uav
                for item in record.observations
                for uav in item.affected_uavs
            }
            if record.event_id == observation.event_id or affected.intersection(record_uavs):
                record.status = ConfirmationStatus.FALSE_ALARM
                record.resolved_at = observation.received_at
                record.observations.append(observation)
                self.false_alarm_count += 1
                cleared = True
        return cleared

    def _required_evidence(self, observation: Observation) -> int:
        if observation.signal_type == "HEARTBEAT_MISSED":
            return self.heartbeat_miss_threshold
        if observation.event_type is EventType.TARGET_DISCOVERED:
            return self.target_confirmation_count
        if observation.event_type is EventType.TARGET_DESTROYED:
            return self.destruction_confirmation_count
        return 1

    @staticmethod
    def _direct_confirmation(observation: Observation) -> bool:
        return observation.signal_type in {
            "ACTIVE_FAILURE_REPORT",
            "AUTHORITATIVE_TARGET_DESTROYED",
            "REGION_LEASE_VACANT",
        } and observation.confidence >= 0.95

    @staticmethod
    def _build_confirmed(record: ConfirmationRecord, observation: Observation) -> ConfirmedEvent:
        first = min(record.observations, key=lambda item: (item.emitted_at, item.observation_id))
        latest = max(record.observations, key=lambda item: (item.received_at, item.observation_id))
        return ConfirmedEvent(
            event_id=observation.event_id,
            event_type=observation.event_type,
            source_event=observation.source_event,
            affected_uavs=observation.affected_uavs,
            affected_regions=observation.affected_regions,
            affected_targets=observation.affected_targets,
            severity=observation.severity,
            payload=dict(observation.payload),
            event_seed=observation.event_seed,
            state_version=observation.state_version,
            occurred_at=observation.occurred_at if observation.occurred_at is not None else first.emitted_at,
            emitted_at=first.emitted_at,
            received_at=latest.received_at,
            suspected_at=record.suspected_at,
            confirmed_at=observation.received_at,
            resolved_at=None,
            status=ConfirmationStatus.CONFIRMED,
            evidence_ids=tuple(item.observation_id for item in record.observations),
        )

    def process(self, observation: Observation) -> StateMachineResult:
        evidence_key = self._evidence_key(observation)
        if evidence_key in self._processed_evidence:
            self.duplicate_observation_count += 1
            record = self.records.get(observation.event_id)
            status = record.status if record else ConfirmationStatus.NORMAL
            return StateMachineResult(
                accepted=False,
                duplicate=True,
                status_before=status,
                status_after=status,
            )
        self._processed_evidence.add(evidence_key)

        if observation.signal_type in {"HEALTHY_TELEMETRY", "COMMUNICATION_RECOVERY"}:
            cleared = self._clear_suspicions_for_healthy(observation)
            return StateMachineResult(
                accepted=True,
                status_before=ConfirmationStatus.SUSPECTED if cleared else ConfirmationStatus.NORMAL,
                status_after=ConfirmationStatus.FALSE_ALARM if cleared else ConfirmationStatus.NORMAL,
            )

        record = self._record_for(observation)
        before = record.status
        if before in {
            ConfirmationStatus.CONFIRMED,
            ConfirmationStatus.RECOVERING,
            ConfirmationStatus.RESOLVED,
            ConfirmationStatus.FALSE_ALARM,
            ConfirmationStatus.REVOKED,
            ConfirmationStatus.EXPIRED,
        }:
            self.late_observation_count += 1
            return StateMachineResult(
                accepted=False,
                late=True,
                status_before=before,
                status_after=before,
                confirmed_event=record.confirmed_event,
            )

        record.observations.append(observation)
        record.last_sequence_by_source[observation.source_id] = max(
            observation.sequence,
            record.last_sequence_by_source.get(observation.source_id, -1),
        )
        if not observation.positive:
            if record.status is ConfirmationStatus.SUSPECTED:
                record.status = ConfirmationStatus.FALSE_ALARM
                record.resolved_at = observation.received_at
                self.false_alarm_count += 1
            return StateMachineResult(
                accepted=True,
                status_before=before,
                status_after=record.status,
            )

        record.positive_evidence_keys.add(evidence_key)
        if self._direct_confirmation(observation):
            record.status = ConfirmationStatus.CONFIRMED
            record.confirmed_at = observation.received_at
        else:
            if record.status is ConfirmationStatus.NORMAL:
                record.status = ConfirmationStatus.SUSPECTED
                record.suspected_at = observation.received_at
            if len(record.positive_evidence_keys) >= self._required_evidence(observation):
                record.status = ConfirmationStatus.CONFIRMED
                record.confirmed_at = observation.received_at

        confirmed = None
        if record.status is ConfirmationStatus.CONFIRMED and record.confirmed_event is None:
            confirmed = self._build_confirmed(record, observation)
            record.confirmed_event = confirmed
        return StateMachineResult(
            accepted=True,
            status_before=before,
            status_after=record.status,
            confirmed_event=confirmed,
        )

    def process_many(self, observations: Iterable[Observation]) -> list[ConfirmedEvent]:
        confirmed: list[ConfirmedEvent] = []
        for observation in sorted(
            observations,
            key=lambda item: (item.received_at, item.emitted_at, item.observation_id),
        ):
            result = self.process(observation)
            if result.confirmed_event is not None:
                confirmed.append(result.confirmed_event)
        return confirmed

    def advance(self, now: float) -> tuple[str, ...]:
        expired = []
        for event_id, record in self.records.items():
            if (
                record.status is ConfirmationStatus.SUSPECTED
                and record.suspected_at is not None
                and now - record.suspected_at >= self.suspicion_timeout
            ):
                record.status = ConfirmationStatus.EXPIRED
                record.resolved_at = now
                expired.append(event_id)
        return tuple(expired)

    def mark_recovering(self, event_id: str) -> None:
        record = self.records[event_id]
        if record.status is not ConfirmationStatus.CONFIRMED:
            raise ValueError("only confirmed events can enter recovery")
        record.status = ConfirmationStatus.RECOVERING
        if record.confirmed_event is not None:
            record.confirmed_event.status = ConfirmationStatus.RECOVERING

    def resolve(self, event_id: str, *, resolved_at: float) -> None:
        record = self.records[event_id]
        if record.status not in {ConfirmationStatus.CONFIRMED, ConfirmationStatus.RECOVERING}:
            raise ValueError("only confirmed/recovering events can resolve")
        record.status = ConfirmationStatus.RESOLVED
        record.resolved_at = resolved_at
        if record.confirmed_event is not None:
            record.confirmed_event.status = ConfirmationStatus.RESOLVED
            record.confirmed_event.resolved_at = resolved_at

    def revoke(self, event_id: str, *, at: float) -> None:
        record = self.records[event_id]
        record.status = ConfirmationStatus.REVOKED
        record.resolved_at = at
        if record.confirmed_event is not None:
            record.confirmed_event.status = ConfirmationStatus.REVOKED
            record.confirmed_event.resolved_at = at
