"""Integration adapter for event_runtime and ppo_allocation.

This module provides the bridge between the event_runtime layer and the
existing ppo_allocation environment, enabling the event-driven decision
flow while preserving legacy compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .concurrency import ConcurrencyManager, AssignmentCommand, CommandStatus
from .events import ConfirmedEvent, EventType, TruthEvent
from .queue import EventQueue
from .state_machine import ConfirmationStateMachine
from .observation import Observation


@dataclass
class BeliefState:
    """System belief state based on confirmed events."""
    graph_version: int = 0
    pending_regions: set[str] = field(default_factory=set)
    active_commands: dict[str, AssignmentCommand] = field(default_factory=dict)
    confirmed_events: list[ConfirmedEvent] = field(default_factory=list)
    last_decision_version: int = 0


class EventRuntimeAdapter:
    """Adapter between event_runtime and ppo_allocation environment."""

    def __init__(
        self,
        *,
        merge_window: float = 0.10,
        suspicion_timeout: float = 5.0,
        heartbeat_miss_threshold: int = 3,
        target_confirmation_count: int = 3,
        destruction_confirmation_count: int = 2,
    ) -> None:
        self.state_machine = ConfirmationStateMachine(
            heartbeat_miss_threshold=heartbeat_miss_threshold,
            target_confirmation_count=target_confirmation_count,
            destruction_confirmation_count=destruction_confirmation_count,
            suspicion_timeout=suspicion_timeout,
        )
        self.event_queue = EventQueue(merge_window=merge_window)
        self.concurrency = ConcurrencyManager()
        self.belief = BeliefState()
        self._processed_observations: set[str] = set()

    def process_observation(self, observation: Observation) -> ConfirmedEvent | None:
        """Process an observation through the confirmation state machine."""
        result = self.state_machine.process(observation)
        if result.confirmed_event is not None:
            self.event_queue.enqueue(result.confirmed_event)
            self.belief.confirmed_events.append(result.confirmed_event)
            return result.confirmed_event
        return None

    def process_observations(self, observations: list[Observation]) -> list[ConfirmedEvent]:
        """Process multiple observations."""
        confirmed = []
        for observation in observations:
            result = self.state_machine.process(observation)
            if result.confirmed_event is not None:
                self.event_queue.enqueue(result.confirmed_event)
                self.belief.confirmed_events.append(result.confirmed_event)
                confirmed.append(result.confirmed_event)
        return confirmed

    def advance_time(self, now: float) -> list[ConfirmedEvent]:
        """Advance time and process any timeout-triggered events."""
        # Expire suspected events
        expired = self.state_machine.advance(now)
        
        # Process queue
        confirmed = []
        while True:
            batch = self.event_queue.pop_atomic_batch(now=now)
            if not batch:
                break
            confirmed.extend(batch)
            self.belief.confirmed_events.extend(batch)
        
        # Cleanup expired commands and leases
        self.concurrency.cleanup_expired(now)
        
        return confirmed

    def get_pending_batch(self, now: float) -> tuple[ConfirmedEvent, ...]:
        """Get next atomic batch from queue."""
        return self.event_queue.pop_atomic_batch(now=now)

    def update_graph_version(self) -> int:
        """Increment graph version for new confirmed events."""
        self.belief.graph_version += 1
        return self.belief.graph_version

    def create_command(
        self,
        command_id: str,
        uav_id: str,
        region_id: str,
        ttl: float = 0.5,
        now: float = 0.0,
    ) -> AssignmentCommand:
        """Create an assignment command."""
        return self.concurrency.create_command(
            command_id=command_id,
            uav_id=uav_id,
            region_id=region_id,
            graph_version=self.belief.graph_version,
            action_version=self.belief.last_decision_version,
            ttl=ttl,
            now=now,
        )

    def validate_and_commit_command(self, command_id: str) -> bool:
        """Validate and commit a command."""
        if not self.concurrency.validate_command(command_id, self.belief.graph_version):
            return False
        self.concurrency.commit_command(command_id)
        return True

    def reject_stale_actions(self) -> int:
        """Reject all stale actions based on current graph version."""
        rejected = 0
        for command_id, command in list(self.concurrency.commands.items()):
            if command.status in {CommandStatus.PROPOSED, CommandStatus.VALIDATED}:
                if self.concurrency.reject_stale_action(command, self.belief.graph_version):
                    rejected += 1
        return rejected

    def get_valid_assignments(self, region_id: str, at: float) -> list[str]:
        """Get valid UAV assignments for a region."""
        valid = []
        for lease in self.concurrency.leases.values():
            if lease.region_id == region_id and lease.is_valid_at(at):
                valid.append(lease.uav_id)
        return valid

    def has_pending_events(self) -> bool:
        """Check if there are pending events in the queue."""
        return len(self.event_queue) > 0

    def has_pending_regions(self) -> bool:
        """Check if there are pending regions needing reassignment."""
        return len(self.belief.pending_regions) > 0

    def add_pending_region(self, region_id: str) -> None:
        """Add a region to pending set."""
        self.belief.pending_regions.add(region_id)

    def remove_pending_region(self, region_id: str) -> None:
        """Remove a region from pending set."""
        self.belief.pending_regions.discard(region_id)

    def get_snapshot_hash(self) -> str:
        """Get current belief state snapshot hash."""
        import hashlib
        import json
        
        data = {
            "graph_version": self.belief.graph_version,
            "pending_regions": sorted(self.belief.pending_regions),
            "confirmed_events": len(self.belief.confirmed_events),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def reset(self) -> None:
        """Reset adapter state."""
        self.state_machine = ConfirmationStateMachine(
            heartbeat_miss_threshold=self.state_machine.heartbeat_miss_threshold,
            target_confirmation_count=self.state_machine.target_confirmation_count,
            destruction_confirmation_count=self.state_machine.destruction_confirmation_count,
            suspicion_timeout=self.state_machine.suspicion_timeout,
        )
        self.event_queue = EventQueue(merge_window=self.event_queue.merge_window)
        self.concurrency = ConcurrencyManager()
        self.belief = BeliefState()
        self._processed_observations.clear()
