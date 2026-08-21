"""无拒绝采样、可重放的随机事件调度器。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Any, Callable, Mapping, Sequence

from .events import EventTape, RandomEvent, RandomEventType


EVENT_WEIGHTS: Mapping[RandomEventType, float] = {
    RandomEventType.UAV_DAMAGE: 0.30,
    RandomEventType.TARGET_DISCOVERED: 0.30,
    RandomEventType.TARGET_DESTROYED: 0.20,
    RandomEventType.REGION_VACANCY: 0.20,
}

# Frozen out-of-distribution profile used only by Validation-Unseen and
# Test-Unseen.  It keeps the same four event semantics while deliberately
# shifting both the event mixture and the weak-communication detection delay.
# Training must continue to use ``EVENT_WEIGHTS`` / ``DEFAULT_TIMING``.
UNSEEN_EVENT_WEIGHTS: Mapping[RandomEventType, float] = {
    RandomEventType.UAV_DAMAGE: 0.15,
    RandomEventType.TARGET_DISCOVERED: 0.15,
    RandomEventType.TARGET_DESTROYED: 0.30,
    RandomEventType.REGION_VACANCY: 0.40,
}

SUPPORTED_MODES = ("single", "sequential", "overlap", "burst")


class NoValidEventError(RuntimeError):
    """当场景状态下四类事件均不合法。"""


@dataclass(frozen=True, slots=True)
class SchedulerState:
    """事件调度所需的最小、可复制场景状态。

    所有索引都与旧环境的 ``uid/rid/tid`` 一致。调度器在内部
    对这个快照施加事件，从而计算后续事件的条件可行性，不会
    修改调用方的真实对象。
    """

    uav_alive: tuple[bool, ...]
    uav_tasks: tuple[str, ...]
    uav_regions: tuple[tuple[int, ...], ...]
    region_assignments: tuple[int, ...]
    target_discovered: tuple[bool, ...]
    target_tracked: tuple[bool, ...]
    target_destroyed: tuple[bool, ...]
    target_trackers: tuple[int, ...]
    target_regions: tuple[int, ...]
    state_version: int = 0

    @classmethod
    def from_entities(
        cls,
        uavs: Sequence[Any],
        regions: Sequence[Any],
        targets: Sequence[Any],
        *,
        state_version: int = 0,
    ) -> "SchedulerState":
        """从现有 UAV/Region/Target dataclass 创建快照。"""

        ordered_uavs = sorted(uavs, key=lambda item: int(item.uid))
        ordered_regions = sorted(regions, key=lambda item: int(item.rid))
        ordered_targets = sorted(targets, key=lambda item: int(item.tid))
        return cls(
            uav_alive=tuple(bool(item.alive) for item in ordered_uavs),
            uav_tasks=tuple(_enum_name(item.task) for item in ordered_uavs),
            uav_regions=tuple(
                tuple(sorted(int(region) for region in item.regions))
                for item in ordered_uavs
            ),
            region_assignments=tuple(int(item.assigned_uav) for item in ordered_regions),
            target_discovered=tuple(bool(item.discovered) for item in ordered_targets),
            target_tracked=tuple(bool(item.tracked) for item in ordered_targets),
            target_destroyed=tuple(bool(item.destroyed) for item in ordered_targets),
            target_trackers=tuple(int(item.tracker_id) for item in ordered_targets),
            target_regions=tuple(int(item.region) for item in ordered_targets),
            state_version=int(state_version),
        )

    def __post_init__(self) -> None:
        uav_count = len(self.uav_alive)
        target_count = len(self.target_discovered)
        if len(self.uav_tasks) != uav_count or len(self.uav_regions) != uav_count:
            raise ValueError("UAV state fields must have equal lengths")
        if not (
            len(self.target_tracked)
            == len(self.target_destroyed)
            == len(self.target_trackers)
            == len(self.target_regions)
            == target_count
        ):
            raise ValueError("target state fields must have equal lengths")
        if self.state_version < 0:
            raise ValueError("state_version must be non-negative")

    @property
    def num_uavs(self) -> int:
        return len(self.uav_alive)

    @property
    def num_regions(self) -> int:
        return len(self.region_assignments)

    @property
    def num_targets(self) -> int:
        return len(self.target_discovered)

    def candidates(self, event_type: RandomEventType) -> tuple[int, ...]:
        """返回事件的合法主体，过程中不进行随机重试。"""

        if event_type is RandomEventType.UAV_DAMAGE:
            # 保留至少一架存活 UAV，否则后续不再存在“重分配”
            # 问题。最后一架 UAV 的损毁应由环境终止规则单独测试。
            if sum(self.uav_alive) <= 1:
                return ()
            return tuple(
                uid
                for uid, alive in enumerate(self.uav_alive)
                if alive and self.uav_regions[uid]
            )
        if event_type is RandomEventType.TARGET_DISCOVERED:
            return tuple(
                tid
                for tid in range(self.num_targets)
                if not self.target_discovered[tid]
                and not self.target_destroyed[tid]
                and self.discovery_trackers(tid)
            )
        if event_type is RandomEventType.TARGET_DESTROYED:
            return tuple(
                tid
                for tid in range(self.num_targets)
                if self.target_discovered[tid]
                and self.target_tracked[tid]
                and not self.target_destroyed[tid]
                and 0 <= self.target_trackers[tid] < self.num_uavs
            )
        if event_type is RandomEventType.REGION_VACANCY:
            return tuple(
                rid
                for rid, uid in enumerate(self.region_assignments)
                if 0 <= uid < self.num_uavs and self.uav_alive[uid]
            )
        raise ValueError(f"unsupported event type: {event_type}")

    def available_search_uavs(self) -> tuple[int, ...]:
        return tuple(
            uid
            for uid, alive in enumerate(self.uav_alive)
            if alive and self.uav_tasks[uid] != "TRACK" and self.uav_regions[uid]
        )

    def discovery_trackers(self, tid: int) -> tuple[int, ...]:
        """Search UAVs that can physically discover Target ``tid``.

        The original task background requires the detecting UAV to be
        searching the Region containing the Target; choosing an arbitrary
        search UAV would create an event that cannot arise from the scene.
        """

        target_region = self.target_regions[int(tid)]
        return tuple(
            uid
            for uid in self.available_search_uavs()
            if target_region in self.uav_regions[uid]
        )

    def apply(self, event: RandomEvent) -> "SchedulerState":
        """在快照上施加事件，仅用于后续条件采样。"""

        alive = list(self.uav_alive)
        tasks = list(self.uav_tasks)
        uav_regions = [set(regions) for regions in self.uav_regions]
        assignments = list(self.region_assignments)
        discovered = list(self.target_discovered)
        tracked = list(self.target_tracked)
        destroyed = list(self.target_destroyed)
        trackers = list(self.target_trackers)

        def vacate(rid: int) -> None:
            old_uid = assignments[rid]
            if 0 <= old_uid < len(uav_regions):
                uav_regions[old_uid].discard(rid)
                if not uav_regions[old_uid] and tasks[old_uid] == "SEARCH":
                    tasks[old_uid] = "IDLE"
            assignments[rid] = -1

        if event.event_type is RandomEventType.UAV_DAMAGE:
            for uid in event.affected_uavs:
                alive[uid] = False
                tasks[uid] = "IDLE"
                for rid in tuple(uav_regions[uid]):
                    vacate(rid)
        elif event.event_type is RandomEventType.TARGET_DISCOVERED:
            tid = event.affected_targets[0]
            uid = event.affected_uavs[0]
            discovered[tid] = True
            tracked[tid] = True
            trackers[tid] = uid
            tasks[uid] = "TRACK"
            for rid in tuple(uav_regions[uid]):
                vacate(rid)
        elif event.event_type is RandomEventType.TARGET_DESTROYED:
            tid = event.affected_targets[0]
            uid = trackers[tid]
            destroyed[tid] = True
            tracked[tid] = False
            trackers[tid] = -1
            if 0 <= uid < len(tasks) and alive[uid]:
                tasks[uid] = "IDLE"
        elif event.event_type is RandomEventType.REGION_VACANCY:
            for rid in event.affected_regions:
                vacate(rid)

        return SchedulerState(
            uav_alive=tuple(alive),
            uav_tasks=tuple(tasks),
            uav_regions=tuple(tuple(sorted(regions)) for regions in uav_regions),
            region_assignments=tuple(assignments),
            target_discovered=tuple(discovered),
            target_tracked=tuple(tracked),
            target_destroyed=tuple(destroyed),
            target_trackers=tuple(trackers),
            target_regions=self.target_regions,
            state_version=self.state_version + 1,
        )

    def with_canonical_recovery(self) -> "SchedulerState":
        """用确定性最小负载规则补全空缺区域。

        它只是离线生成公共 event tape 时的“场景驱动器”，不是待测
        算法；因此不会把任一 PPO/基线的输出写入随机事件带。
        真实环境在线调度时可通过 ``state_transition`` 覆盖此规则。
        """

        assignments = list(self.region_assignments)
        uav_regions = [set(regions) for regions in self.uav_regions]
        tasks = list(self.uav_tasks)
        eligible = [
            uid
            for uid, alive in enumerate(self.uav_alive)
            if alive and tasks[uid] != "TRACK"
        ]
        if not eligible:
            return self
        for rid, assigned_uid in enumerate(assignments):
            if assigned_uid >= 0:
                continue
            uid = min(eligible, key=lambda candidate: (len(uav_regions[candidate]), candidate))
            assignments[rid] = uid
            uav_regions[uid].add(rid)
            tasks[uid] = "SEARCH"
        return replace(
            self,
            uav_tasks=tuple(tasks),
            uav_regions=tuple(tuple(sorted(regions)) for regions in uav_regions),
            region_assignments=tuple(assignments),
        )


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value).upper()


@dataclass(frozen=True, slots=True)
class TimingProfile:
    interval_min: float
    interval_max: float
    observation_delay_min: float
    observation_delay_max: float


DEFAULT_TIMING: Mapping[str, TimingProfile] = {
    # single 表示同一时刻只有一个待处理事件，但一个 episode
    # 仍可包含默认 5 个彼此分隔较远的事件。
    "single": TimingProfile(8.0, 12.0, 0.0, 0.5),
    "sequential": TimingProfile(4.0, 8.0, 0.2, 1.0),
    # 下一事件间隔严格小于上一事件的最小观测延迟，
    # 因而 event tape 本身就能证明存在重叠，不依赖训练算法。
    "overlap": TimingProfile(0.25, 0.75, 1.0, 3.0),
    # burst 的 interval 是两个事件簇之间的随机间隔；簇内时间相同。
    "burst": TimingProfile(2.0, 4.0, 0.5, 2.0),
}


# Occurrence-time contracts remain single/sequential/overlap/burst; only the
# observation delays are shifted outside the default ranges.  In particular,
# overlap still guarantees that the next event occurs before the previous one
# is observed.
UNSEEN_TIMING: Mapping[str, TimingProfile] = {
    "single": TimingProfile(8.0, 12.0, 1.5, 3.0),
    "sequential": TimingProfile(4.0, 8.0, 1.5, 3.0),
    "overlap": TimingProfile(0.25, 0.75, 3.0, 6.0),
    "burst": TimingProfile(2.0, 4.0, 2.0, 4.0),
}


class RandomEventScheduler:
    """从当前有效事件类型中一次加权抽样。

    算法先显式列举有效类型，再将 0.30/0.30/0.20/0.20 在该集合
    上重归一化；不使用“抽到非法后重试”的拒绝采样。
    """

    def __init__(
        self,
        *,
        event_count: int = 5,
        weights: Mapping[RandomEventType | str, float] | None = None,
        timing: Mapping[str, TimingProfile] | None = None,
    ) -> None:
        if event_count <= 0:
            raise ValueError("event_count must be positive")
        self.event_count = int(event_count)
        selected_weights = EVENT_WEIGHTS if weights is None else weights
        self.weights = {
            RandomEventType(kind): float(weight)
            for kind, weight in selected_weights.items()
        }
        if set(self.weights) != set(RandomEventType):
            raise ValueError("weights must define all four event types")
        if any(weight < 0 for weight in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("event weights must be non-negative with a positive sum")
        self.timing = dict(DEFAULT_TIMING if timing is None else timing)
        if set(self.timing) != set(SUPPORTED_MODES):
            raise ValueError(f"timing must define modes {SUPPORTED_MODES}")

    def valid_event_types(self, state: SchedulerState) -> tuple[RandomEventType, ...]:
        return tuple(kind for kind in RandomEventType if state.candidates(kind))

    def conditional_probabilities(
        self, state: SchedulerState
    ) -> dict[RandomEventType, float]:
        valid_types = self.valid_event_types(state)
        total = sum(self.weights[kind] for kind in valid_types)
        if not valid_types or total <= 0:
            raise NoValidEventError("no conditionally valid random event type")
        return {kind: self.weights[kind] / total for kind in valid_types}

    def generate_tape(
        self,
        initial_state: SchedulerState,
        *,
        initial_seed: int,
        event_seed: int,
        mode: str = "sequential",
        event_count: int | None = None,
        start_at: float = 0.0,
        state_transition: Callable[[SchedulerState, RandomEvent], SchedulerState] | None = None,
    ) -> EventTape:
        """生成事件带。

        ``state_transition`` 可接入外部场景演化规则；默认使用
        :meth:`SchedulerState.apply` 的最小事件语义。
        """

        mode = str(mode).lower()
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"unsupported mode {mode!r}; expected one of {SUPPORTED_MODES}")
        count = self.event_count if event_count is None else int(event_count)
        if count <= 0:
            raise ValueError("event_count must be positive")
        master_rng = random.Random(int(event_seed))
        profile = self.timing[mode]
        state = initial_state
        events: list[RandomEvent] = []
        occurred_at = float(start_at)
        transition = state_transition or (
            lambda current, event: current.apply(event).with_canonical_recovery()
        )

        burst_cluster_observed_at: float | None = None
        for index in range(count):
            per_event_seed = master_rng.getrandbits(63)
            event_rng = random.Random(per_event_seed)
            if index > 0:
                if mode == "burst" and index % 3 != 0:
                    interval = 0.0
                else:
                    interval = event_rng.uniform(profile.interval_min, profile.interval_max)
                occurred_at += interval
            if mode == "burst" and index % 3 != 0 and burst_cluster_observed_at is not None:
                # Burst cluster members share the cluster leader's observed_at:
                # all events in a 100ms burst window arrive together so they
                # can be merged into one atomic batch (single graph_version
                # increment, single policy call).
                observed_at = burst_cluster_observed_at
            else:
                observed_at = occurred_at + event_rng.uniform(
                    profile.observation_delay_min,
                    profile.observation_delay_max,
                )
                if mode == "burst" and index % 3 == 0:
                    burst_cluster_observed_at = observed_at
            event = self.sample_event(
                state,
                rng=event_rng,
                event_id=f"E{index:04d}",
                occurred_at=occurred_at,
                observed_at=observed_at,
                event_seed=per_event_seed,
            )
            events.append(event)
            state = transition(state, event)

        return EventTape(
            initial_seed=int(initial_seed),
            event_seed=int(event_seed),
            mode=mode,
            events=tuple(events),
        )

    def sample_event(
        self,
        state: SchedulerState,
        *,
        rng: random.Random,
        event_id: str,
        occurred_at: float,
        observed_at: float,
        event_seed: int,
    ) -> RandomEvent:
        probabilities = self.conditional_probabilities(state)
        event_type = _weighted_choice(probabilities, rng.random())
        candidates = state.candidates(event_type)
        subject = candidates[rng.randrange(len(candidates))]
        severity = round(rng.uniform(0.35, 1.0), 6)

        affected_uavs: tuple[int, ...] = ()
        affected_regions: tuple[int, ...] = ()
        affected_targets: tuple[int, ...] = ()
        source_event = "external_random_scheduler"
        payload: dict[str, Any] = {
            "conditional_probabilities": {
                kind.value: probability for kind, probability in probabilities.items()
            }
        }

        if event_type is RandomEventType.UAV_DAMAGE:
            uid = subject
            affected_uavs = (uid,)
            affected_regions = state.uav_regions[uid]
            payload.update({"damaged_uav": uid, "released_regions": list(affected_regions)})
        elif event_type is RandomEventType.TARGET_DISCOVERED:
            tid = subject
            trackers = state.discovery_trackers(tid)
            uid = trackers[rng.randrange(len(trackers))]
            affected_targets = (tid,)
            affected_uavs = (uid,)
            affected_regions = state.uav_regions[uid]
            source_event = "target_sensor_detection"
            payload.update(
                {
                    "discovered_target": tid,
                    "tracker_uav": uid,
                    "target_region": state.target_regions[tid],
                    "generated_region_vacancy": list(affected_regions),
                }
            )
        elif event_type is RandomEventType.TARGET_DESTROYED:
            tid = subject
            uid = state.target_trackers[tid]
            affected_targets = (tid,)
            affected_uavs = (uid,)
            source_event = "target_status_update"
            payload.update({"destroyed_target": tid, "released_uav": uid})
        elif event_type is RandomEventType.REGION_VACANCY:
            rid = subject
            uid = state.region_assignments[rid]
            affected_regions = (rid,)
            affected_uavs = (uid,)
            source_event = "direct_region_vacancy"
            payload.update({"vacant_region": rid, "previous_uav": uid})

        return RandomEvent(
            event_id=event_id,
            event_type=event_type,
            occurred_at=float(occurred_at),
            observed_at=float(observed_at),
            source_event=source_event,
            affected_uavs=affected_uavs,
            affected_regions=affected_regions,
            affected_targets=affected_targets,
            severity=severity,
            payload=payload,
            event_seed=int(event_seed),
            state_version=state.state_version,
        )


def _weighted_choice(
    probabilities: Mapping[RandomEventType, float], draw: float
) -> RandomEventType:
    """对已归一化概率执行一次抽样。"""

    cumulative = 0.0
    last: RandomEventType | None = None
    for event_type, probability in probabilities.items():
        last = event_type
        cumulative += probability
        if draw < cumulative:
            return event_type
    if last is None:
        raise NoValidEventError("cannot sample from an empty probability table")
    return last  # 浮点累加误差保底，不是拒绝采样。


def build_scheduler_state(
    uavs: Sequence[Any],
    regions: Sequence[Any],
    targets: Sequence[Any],
    *,
    state_version: int = 0,
) -> SchedulerState:
    """脚本式旧代码的便捷兼容入口。"""

    return SchedulerState.from_entities(
        uavs, regions, targets, state_version=state_version
    )
