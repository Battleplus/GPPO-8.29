"""Deterministic truth-event scheduling for five timing modes."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import random
from typing import Mapping

from .events import EventType, TruthEvent, TruthEventTape, canonical_json_bytes


TRAIN_EVENT_WEIGHTS = {
    EventType.UAV_DAMAGE: 0.30,
    EventType.TARGET_DISCOVERED: 0.30,
    EventType.TARGET_DESTROYED: 0.20,
    EventType.REGION_VACANCY: 0.20,
}

UNSEEN_EVENT_WEIGHTS = {
    EventType.UAV_DAMAGE: 0.10,
    EventType.TARGET_DISCOVERED: 0.10,
    EventType.TARGET_DESTROYED: 0.35,
    EventType.REGION_VACANCY: 0.45,
}


@dataclass
class _ScenarioState:
    alive: dict[str, bool] = field(default_factory=lambda: {f"U{i}": True for i in range(4)})
    modes: dict[str, str] = field(default_factory=lambda: {f"U{i}": "SEARCH" for i in range(4)})
    assignments: dict[str, str | None] = field(
        default_factory=lambda: {f"R{i}": f"U{i}" for i in range(4)}
    )
    target_regions: dict[str, str] = field(
        default_factory=lambda: {"T0": "R0", "T1": "R1", "T2": "R2"}
    )
    target_status: dict[str, str] = field(
        default_factory=lambda: {"T0": "UNDISCOVERED", "T1": "UNDISCOVERED", "T2": "UNDISCOVERED"}
    )
    trackers: dict[str, str | None] = field(
        default_factory=lambda: {"T0": None, "T1": None, "T2": None}
    )

    def snapshot_hash(self) -> str:
        value = {
            "alive": self.alive,
            "modes": self.modes,
            "assignments": self.assignments,
            "target_regions": self.target_regions,
            "target_status": self.target_status,
            "trackers": self.trackers,
        }
        return hashlib.sha256(canonical_json_bytes(value)).hexdigest()

    def valid_types(self) -> tuple[EventType, ...]:
        valid: list[EventType] = []
        assigned_search_uavs = {
            holder
            for holder in self.assignments.values()
            if holder is not None and self.alive[holder] and self.modes[holder] == "SEARCH"
        }
        if assigned_search_uavs:
            valid.append(EventType.UAV_DAMAGE)
        if any(
            status == "UNDISCOVERED"
            and (holder := self.assignments[self.target_regions[target]]) is not None
            and self.alive[holder]
            and self.modes[holder] == "SEARCH"
            for target, status in self.target_status.items()
        ):
            valid.append(EventType.TARGET_DISCOVERED)
        if any(status == "TRACKED" for status in self.target_status.values()):
            valid.append(EventType.TARGET_DESTROYED)
        if any(holder is not None for holder in self.assignments.values()):
            valid.append(EventType.REGION_VACANCY)
        return tuple(valid)


def _weighted_choice(
    rng: random.Random,
    candidates: tuple[EventType, ...],
    weights: Mapping[EventType, float],
) -> EventType:
    total = sum(weights[item] for item in candidates)
    point = rng.random() * total
    cumulative = 0.0
    for candidate in candidates:
        cumulative += weights[candidate]
        if point <= cumulative:
            return candidate
    return candidates[-1]


class TruthEventScheduler:
    """Generate policy-independent truth tapes with conditional event sampling."""

    MODES = ("single", "sequential", "overlap", "burst", "unseen")

    def __init__(
        self,
        *,
        train_weights: Mapping[EventType, float] | None = None,
        unseen_weights: Mapping[EventType, float] | None = None,
    ) -> None:
        self.train_weights = dict(train_weights or TRAIN_EVENT_WEIGHTS)
        self.unseen_weights = dict(unseen_weights or UNSEEN_EVENT_WEIGHTS)

    @staticmethod
    def initial_snapshot_hash() -> str:
        return _ScenarioState().snapshot_hash()

    @staticmethod
    def _event_times(mode: str, count: int, rng: random.Random) -> list[float]:
        if mode == "single":
            return [10.0]
        if mode == "sequential":
            current = 10.0
            result = []
            for _ in range(count):
                result.append(round(current, 6))
                current += rng.uniform(5.0, 7.0)
            return result
        if mode == "overlap":
            current = 10.0
            result = []
            for _ in range(count):
                result.append(round(current, 6))
                current += rng.uniform(0.4, 1.2)
            return result
        if mode == "burst":
            result = []
            current = 10.0
            while len(result) < count:
                batch_size = min(rng.randint(2, 3), count - len(result))
                offsets = (0.0, 0.03, 0.07)
                result.extend(round(current + offsets[index], 6) for index in range(batch_size))
                current += 5.0
            return result
        if mode == "unseen":
            current = 10.0
            result = []
            for index in range(count):
                result.append(round(current, 6))
                current += rng.uniform(0.15, 0.65) if index % 3 else rng.uniform(1.0, 1.8)
            return result
        raise ValueError(f"unsupported event mode: {mode}")

    def generate(
        self,
        *,
        initial_seed: int,
        event_seed: int,
        mode: str,
        event_count: int = 5,
        tape_id: str | None = None,
    ) -> TruthEventTape:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if event_count <= 0:
            raise ValueError("event_count must be positive")
        if mode == "single":
            event_count = 1
        rng = random.Random((int(initial_seed) << 64) ^ int(event_seed))
        state = _ScenarioState()
        snapshot_hash = state.snapshot_hash()
        event_times = self._event_times(mode, event_count, rng)
        weights = self.unseen_weights if mode == "unseen" else self.train_weights
        events: list[TruthEvent] = []

        for index, occurred_at in enumerate(event_times):
            valid = state.valid_types()
            if not valid:
                break
            event_type = _weighted_choice(rng, valid, weights)
            event_id = f"{mode}-{event_seed}-{index:03d}"
            affected_uavs: tuple[str, ...] = ()
            affected_regions: tuple[str, ...] = ()
            affected_targets: tuple[str, ...] = ()
            payload: dict[str, object] = {}

            if event_type is EventType.UAV_DAMAGE:
                candidates = sorted({
                    holder
                    for holder in state.assignments.values()
                    if holder is not None and state.alive[holder] and state.modes[holder] == "SEARCH"
                })
                uav = rng.choice(candidates)
                regions = tuple(sorted(region for region, holder in state.assignments.items() if holder == uav))
                affected_uavs = (uav,)
                affected_regions = regions
                payload = {
                    "active_report": rng.random() < 0.5,
                    "cause": "simulated_hard_failure",
                }
                state.alive[uav] = False
                state.modes[uav] = "IDLE"
                for region in regions:
                    state.assignments[region] = None
            elif event_type is EventType.TARGET_DISCOVERED:
                candidates = [
                    target
                    for target, status in state.target_status.items()
                    if status == "UNDISCOVERED"
                    and state.assignments[state.target_regions[target]] is not None
                    and state.alive[state.assignments[state.target_regions[target]] or ""]
                ]
                target = rng.choice(sorted(candidates))
                region = state.target_regions[target]
                finder = state.assignments[region]
                assert finder is not None
                released = tuple(sorted(item for item, holder in state.assignments.items() if holder == finder))
                affected_uavs = (finder,)
                affected_regions = released
                affected_targets = (target,)
                payload = {"finder_uav": finder, "region_id": region}
                state.target_status[target] = "TRACKED"
                state.trackers[target] = finder
                state.modes[finder] = "TRACK"
                for item in released:
                    state.assignments[item] = None
            elif event_type is EventType.TARGET_DESTROYED:
                target = rng.choice(sorted(
                    item for item, status in state.target_status.items() if status == "TRACKED"
                ))
                tracker = state.trackers[target]
                affected_targets = (target,)
                affected_uavs = () if tracker is None else (tracker,)
                payload = {"tracker_uav": tracker, "authoritative": rng.random() < 0.25}
                state.target_status[target] = "DESTROYED"
                state.trackers[target] = None
                if tracker is not None and state.alive[tracker]:
                    state.modes[tracker] = "SEARCH"
            else:
                region = rng.choice(sorted(
                    item for item, holder in state.assignments.items() if holder is not None
                ))
                holder = state.assignments[region]
                affected_regions = (region,)
                affected_uavs = () if holder is None else (holder,)
                payload = {"previous_holder": holder, "cause": "lease_vacancy"}
                state.assignments[region] = None

            events.append(TruthEvent(
                event_id=event_id,
                event_type=event_type,
                source_event=event_type.value,
                affected_uavs=affected_uavs,
                affected_regions=affected_regions,
                affected_targets=affected_targets,
                severity=round(rng.uniform(0.65, 1.0), 6),
                payload=payload,
                event_seed=event_seed * 1009 + index,
                state_version=index,
                occurred_at=occurred_at,
            ))

        generated_id = tape_id or f"{mode}-{initial_seed}-{event_seed}"
        return TruthEventTape.build(
            tape_id=generated_id,
            initial_seed=initial_seed,
            event_seed=event_seed,
            mode=mode,
            initial_snapshot_hash=snapshot_hash,
            events=events,
            metadata={
                "distribution": "unseen" if mode == "unseen" else "train",
                "event_weights": {key.value: value for key, value in weights.items()},
                "single_is_reset_isolated": mode == "single",
            },
        )
