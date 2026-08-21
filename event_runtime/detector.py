"""Deterministic TruthEvent-to-Observation detector and channel model."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import random

from .events import EventType, TruthEvent, TruthEventTape
from .observation import Observation, ObservationTape, WeakCommunicationProfile


class EventDetector:
    def __init__(self, profile: WeakCommunicationProfile | None = None) -> None:
        self.profile = profile or WeakCommunicationProfile()

    @staticmethod
    def _rng(initial_seed: int, event_seed: int, profile_name: str) -> random.Random:
        digest = hashlib.sha256(
            f"{initial_seed}:{event_seed}:{profile_name}".encode("utf-8")
        ).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _receive_time(self, emitted_at: float, rng: random.Random) -> float | None:
        if rng.random() < self.profile.packet_loss_rate:
            return None
        delay = rng.uniform(self.profile.delay_min, self.profile.delay_max)
        delay += rng.uniform(0.0, self.profile.reorder_jitter)
        received_at = emitted_at + delay
        for partition in self.profile.partitions:
            if partition.contains(emitted_at):
                if partition.behavior == "drop":
                    return None
                received_at = max(received_at, partition.end_at + delay)
        return round(received_at, 6)

    @staticmethod
    def _signal_plan(event: TruthEvent, profile: WeakCommunicationProfile) -> list[tuple[str, str, float, float]]:
        if event.event_type is EventType.UAV_DAMAGE:
            if bool(event.payload.get("active_report", False)):
                return [("ACTIVE_FAILURE_REPORT", "onboard_self_test", 0.05, 0.99)]
            return [
                ("HEARTBEAT_MISSED", "heartbeat_monitor", profile.heartbeat_interval * index, 0.75)
                for index in range(1, profile.heartbeat_miss_threshold + 1)
            ]
        if event.event_type is EventType.TARGET_DISCOVERED:
            return [
                ("TARGET_DETECTION", "eo_ir", 0.10 * index, min(0.99, 0.68 + index * 0.08))
                for index in range(1, profile.target_confirmation_count + 1)
            ]
        if event.event_type is EventType.TARGET_DESTROYED:
            if bool(event.payload.get("authoritative", False)):
                return [("AUTHORITATIVE_TARGET_DESTROYED", "command_system", 0.05, 0.99)]
            return [
                ("TARGET_DESTROYED_EVIDENCE", "eo_sar", 0.15 * index, 0.90)
                for index in range(1, profile.destruction_confirmation_count + 1)
            ]
        return [("REGION_LEASE_VACANT", "lease_monitor", 0.05, 0.99)]

    def generate_observation_tape(self, truth_tape: TruthEventTape) -> ObservationTape:
        rng = self._rng(truth_tape.initial_seed, truth_tape.event_seed, self.profile.name)
        observations: list[Observation] = []
        sequence = 0

        for event in truth_tape.events:
            if rng.random() < self.profile.false_negative_rate:
                continue
            for signal_type, source_type, offset, confidence in self._signal_plan(event, self.profile):
                emitted_at = round(event.occurred_at + offset, 6)
                received_at = self._receive_time(emitted_at, rng)
                if received_at is None:
                    continue
                sequence += 1
                observation_id = f"obs-{truth_tape.tape_id}-{sequence:04d}"
                observation = Observation(
                    observation_id=observation_id,
                    event_id=event.event_id,
                    event_type=event.event_type,
                    source_event=event.source_event,
                    source_id=event.affected_uavs[0] if event.affected_uavs else "CONTROL",
                    source_type=source_type,
                    signal_type=signal_type,
                    sequence=sequence,
                    confidence=confidence,
                    positive=True,
                    emitted_at=emitted_at,
                    received_at=received_at,
                    occurred_at=event.occurred_at,
                    affected_uavs=event.affected_uavs,
                    affected_regions=event.affected_regions,
                    affected_targets=event.affected_targets,
                    severity=event.severity,
                    event_seed=event.event_seed,
                    state_version=event.state_version,
                    payload=dict(event.payload),
                )
                observations.append(observation)
                if rng.random() < self.profile.duplicate_rate:
                    sequence += 1
                    observations.append(replace(
                        observation,
                        observation_id=f"obs-{truth_tape.tape_id}-{sequence:04d}",
                        sequence=sequence,
                        received_at=round(received_at + rng.uniform(0.001, 0.05), 6),
                        is_duplicate=True,
                        duplicate_of=observation_id,
                    ))

        for partition_index, partition in enumerate(self.profile.partitions):
            for uav_index, uav_id in enumerate(partition.affected_uavs):
                sequence += 1
                observations.append(Observation(
                    observation_id=f"obs-{truth_tape.tape_id}-{sequence:04d}",
                    event_id=f"partition-{partition_index}-{uav_id}",
                    event_type=EventType.UAV_DAMAGE,
                    source_event="COMMUNICATION_RECOVERY",
                    source_id=uav_id,
                    source_type="network_monitor",
                    signal_type="HEALTHY_TELEMETRY",
                    sequence=sequence,
                    confidence=0.99,
                    positive=False,
                    emitted_at=partition.end_at,
                    received_at=round(partition.end_at + self.profile.delay_min + 0.001 * uav_index, 6),
                    occurred_at=partition.start_at,
                    affected_uavs=(uav_id,),
                    payload={"partition_start": partition.start_at, "partition_end": partition.end_at},
                ))

        false_positive_trials = max(1, len(truth_tape.events))
        for trial in range(false_positive_trials):
            if rng.random() >= self.profile.false_positive_rate:
                continue
            event_id = f"false-positive-{truth_tape.tape_id}-{trial:03d}"
            uav_id = f"U{rng.randrange(4)}"
            emitted_at = round(5.0 + trial * 0.25, 6)
            received_at = self._receive_time(emitted_at, rng)
            if received_at is None:
                continue
            sequence += 1
            observations.append(Observation(
                observation_id=f"obs-{truth_tape.tape_id}-{sequence:04d}",
                event_id=event_id,
                event_type=EventType.UAV_DAMAGE,
                source_event="HEARTBEAT_ANOMALY",
                source_id=uav_id,
                source_type="heartbeat_monitor",
                signal_type="HEARTBEAT_MISSED",
                sequence=sequence,
                confidence=0.55,
                positive=True,
                emitted_at=emitted_at,
                received_at=received_at,
                occurred_at=None,
                affected_uavs=(uav_id,),
                is_false_positive=True,
                payload={"synthetic_false_positive": True},
            ))
            sequence += 1
            recovery_emitted = round(emitted_at + self.profile.heartbeat_interval, 6)
            recovery_received = max(recovery_emitted, received_at + 0.001)
            observations.append(Observation(
                observation_id=f"obs-{truth_tape.tape_id}-{sequence:04d}",
                event_id=event_id,
                event_type=EventType.UAV_DAMAGE,
                source_event="COMMUNICATION_RECOVERY",
                source_id=uav_id,
                source_type="heartbeat_monitor",
                signal_type="HEALTHY_TELEMETRY",
                sequence=sequence,
                confidence=0.99,
                positive=False,
                emitted_at=recovery_emitted,
                received_at=round(recovery_received, 6),
                occurred_at=None,
                affected_uavs=(uav_id,),
                is_false_positive=True,
                payload={"clears_event_id": event_id},
            ))

        return ObservationTape.build(
            tape_id=f"observations-{truth_tape.tape_id}",
            truth_tape_sha256=truth_tape.sha256(),
            initial_seed=truth_tape.initial_seed,
            event_seed=truth_tape.event_seed,
            profile=self.profile,
            observations=observations,
        )
