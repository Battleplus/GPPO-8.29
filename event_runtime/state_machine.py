"""Idempotent event confirmation state machine.

Round-2 semantics (per audit):

- Evidence is counted by DISTINCT SOURCE, never by observation count.  Two
  observations from the same source are one piece of evidence; a duplicate
  observation (``duplicate_of``) never adds evidence.
- TARGET_DISCOVERED requires ``target_confirmation_count`` distinct sources
  (3-of-5 semantics: the bridge generates up to 5 opportunities; only >=3
  distinct-source positives confirm).
- TARGET_DESTROYED has two paths:
    * AUTHORITATIVE_TARGET_DESTROYED -> single observation confirms directly.
    * STRONG_TARGET_DESTROYED -> >=2 independent sources confirm.
  Tracking loss / HEALTHY_TELEMETRY can never confirm destruction.
- UAV_DAMAGE heartbeat/probe chain:
    * ACTIVE_FAILURE_REPORT (trusted hard failure) -> direct confirm.
    * 1 heartbeat miss -> SUSPECTED (lease NOT released).
    * ``heartbeat_miss_threshold`` consecutive misses -> PROBE_REQUIRED.
    * probe timeout OR a second independent failure source -> CONFIRMED.
    * healthy telemetry before confirmation -> FALSE_ALARM.
"""

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
    probe_started_at: float | None = None
    confirmed_at: float | None = None
    resolved_at: float | None = None
    observations: list[Observation] = field(default_factory=list)
    positive_evidence_sources: set[str] = field(default_factory=set)
    failure_sources: set[str] = field(default_factory=set)
    heartbeat_miss_count: int = 0
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
        probe_timeout: float = 2.0,
    ) -> None:
        self.heartbeat_miss_threshold = int(heartbeat_miss_threshold)
        self.target_confirmation_count = int(target_confirmation_count)
        self.destruction_confirmation_count = int(destruction_confirmation_count)
        self.suspicion_timeout = float(suspicion_timeout)
        self.probe_timeout = float(probe_timeout)
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
            if record.status not in {
                ConfirmationStatus.SUSPECTED,
                ConfirmationStatus.PROBE_REQUIRED,
            }:
                continue
            record_uavs = {
                uav
                for item in record.observations
                for uav in item.affected_uavs
            }
            if record.event_id == observation.event_id or affected.intersection(record_uavs):
                record.status = ConfirmationStatus.FALSE_ALARM
                record.resolved_at = observation.received_at
                record.heartbeat_miss_count = 0
                record.observations.append(observation)
                self.false_alarm_count += 1
                cleared = True
        return cleared

    def _required_evidence(self, observation: Observation) -> int:
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
    def _is_heartbeat_observation(observation: Observation) -> bool:
        return observation.signal_type in {"HEARTBEAT_MISSED", "HEALTHY_TELEMETRY", "COMMUNICATION_RECOVERY"}

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

    def _confirm(self, record: ConfirmationRecord, observation: Observation) -> ConfirmedEvent:
        record.status = ConfirmationStatus.CONFIRMED
        record.confirmed_at = observation.received_at
        if record.confirmed_event is None:
            confirmed = self._build_confirmed(record, observation)
            record.confirmed_event = confirmed
            return confirmed
        return record.confirmed_event

    def _handle_heartbeat_miss(self, record: ConfirmationRecord, observation: Observation) -> ConfirmedEvent | None:
        """Heartbeat-miss chain: 1 miss -> SUSPECTED, threshold -> PROBE_REQUIRED.

        A second INDEPENDENT failure source confirms immediately.  The count
        alone (even at the threshold) never confirms -- only the probe timeout
        (in ``advance``) or a second independent source does.
        """
        record.observations.append(observation)
        record.last_sequence_by_source[observation.source_id] = max(
            observation.sequence,
            record.last_sequence_by_source.get(observation.source_id, -1),
        )
        record.failure_sources.add(observation.source_id)

        if len(record.failure_sources) >= 2:
            # Second independent failure source -> CONFIRMED.
            return self._confirm(record, observation)

        if record.status is ConfirmationStatus.NORMAL:
            record.status = ConfirmationStatus.SUSPECTED
            record.suspected_at = observation.received_at

        record.heartbeat_miss_count += 1
        if record.heartbeat_miss_count >= self.heartbeat_miss_threshold:
            record.status = ConfirmationStatus.PROBE_REQUIRED
            if record.probe_started_at is None:
                record.probe_started_at = observation.received_at
        return None

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

        if observation.signal_type == "HEARTBEAT_MISSED":
            confirmed = self._handle_heartbeat_miss(record, observation)
            return StateMachineResult(
                accepted=True,
                status_before=before,
                status_after=record.status,
                confirmed_event=confirmed,
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

        # Evidence counted per DISTINCT SOURCE: re-adding the same source does
        # not increase the independent evidence count.
        record.positive_evidence_sources.add(observation.source_id)

        if self._direct_confirmation(observation):
            confirmed = self._confirm(record, observation)
            return StateMachineResult(
                accepted=True,
                status_before=before,
                status_after=record.status,
                confirmed_event=confirmed,
            )

        if record.status is ConfirmationStatus.NORMAL:
            record.status = ConfirmationStatus.SUSPECTED
            record.suspected_at = observation.received_at
        if len(record.positive_evidence_sources) >= self._required_evidence(observation):
            confirmed = self._confirm(record, observation)
            return StateMachineResult(
                accepted=True,
                status_before=before,
                status_after=record.status,
                confirmed_event=confirmed,
            )
        return StateMachineResult(
            accepted=True,
            status_before=before,
            status_after=record.status,
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
        """Advance timers.

        - PROBE_REQUIRED records whose probe window elapsed -> CONFIRMED
          (probe timeout confirms per the round-2 protocol).
        - Plain SUSPECTED records (no probe started) whose suspicion window
          elapsed -> EXPIRED (kept for non-heartbeat suspicions).
        """
        confirmed_ids: list[str] = []
        expired: list[str] = []
        for event_id, record in self.records.items():
            if record.status is ConfirmationStatus.PROBE_REQUIRED:
                if record.probe_started_at is not None and now - record.probe_started_at >= self.probe_timeout:
                    record.status = ConfirmationStatus.CONFIRMED
                    record.confirmed_at = now
                    if record.confirmed_event is None:
                        latest = max(
                            record.observations,
                            key=lambda item: (item.received_at, item.observation_id),
                        )
                        confirmed = self._build_confirmed(record, latest)
                        record.confirmed_event = confirmed
                    confirmed_ids.append(event_id)
            elif (
                record.status is ConfirmationStatus.SUSPECTED
                and record.probe_started_at is None
                and record.suspected_at is not None
                and now - record.suspected_at >= self.suspicion_timeout
            ):
                record.status = ConfirmationStatus.EXPIRED
                record.resolved_at = now
                expired.append(event_id)
        return tuple(confirmed_ids) + tuple(expired)

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
