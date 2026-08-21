"""Priority queue for confirmed events and atomic merge-window batches."""

from __future__ import annotations

import heapq
from typing import Iterable

from .events import ConfirmationStatus, ConfirmedEvent, EventType


_PRIORITY = {
    EventType.UAV_DAMAGE: 1,
    EventType.TARGET_DISCOVERED: 2,
    EventType.TARGET_DESTROYED: 2,
    EventType.REGION_VACANCY: 2,
}


class EventQueue:
    def __init__(self, *, merge_window: float = 0.10) -> None:
        self.merge_window = float(merge_window)
        self._heap: list[tuple[int, float, float, str, ConfirmedEvent]] = []
        self._queued_ids: set[str] = set()

    def __len__(self) -> int:
        return len(self._heap)

    def enqueue(self, event: ConfirmedEvent) -> bool:
        if event.status is not ConfirmationStatus.CONFIRMED:
            raise ValueError("only ConfirmedEvent in CONFIRMED state can enter the queue")
        if event.event_id in self._queued_ids:
            return False
        confirmed_at = event.confirmed_at if event.confirmed_at is not None else float("inf")
        received_at = event.received_at if event.received_at is not None else confirmed_at
        heapq.heappush(
            self._heap,
            (_PRIORITY[event.event_type], confirmed_at, received_at, event.event_id, event),
        )
        self._queued_ids.add(event.event_id)
        return True

    def enqueue_many(self, events: Iterable[ConfirmedEvent]) -> int:
        return sum(1 for event in events if self.enqueue(event))

    def peek(self) -> ConfirmedEvent | None:
        return None if not self._heap else self._heap[0][-1]

    def pop_atomic_batch(self, *, force: bool = False, now: float | None = None) -> tuple[ConfirmedEvent, ...]:
        if not self._heap:
            return ()
        first_confirmed = self._heap[0][1]
        if not force and now is not None and now < first_confirmed + self.merge_window:
            return ()
        first_priority = self._heap[0][0]
        batch: list[ConfirmedEvent] = []
        retained: list[tuple[int, float, float, str, ConfirmedEvent]] = []
        while self._heap:
            item = heapq.heappop(self._heap)
            priority, confirmed_at, _, event_id, event = item
            if priority == 0 or (
                priority == first_priority and confirmed_at <= first_confirmed + self.merge_window
            ) or (
                priority > first_priority and confirmed_at <= first_confirmed + self.merge_window
            ):
                self._queued_ids.remove(event_id)
                batch.append(event)
            else:
                retained.append(item)
        for item in retained:
            heapq.heappush(self._heap, item)
        batch.sort(key=lambda event: (
            _PRIORITY[event.event_type],
            event.confirmed_at if event.confirmed_at is not None else float("inf"),
            event.received_at if event.received_at is not None else float("inf"),
            event.event_id,
        ))
        return tuple(batch)

    def drain(self) -> tuple[tuple[ConfirmedEvent, ...], ...]:
        batches = []
        while self._heap:
            batches.append(self.pop_atomic_batch(force=True))
        return tuple(batches)
