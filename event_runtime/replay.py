"""Replay functionality for truth and observation tapes.

This module provides deterministic replay of truth and observation tapes,
enabling separation of detection randomness from strategy randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .events import TruthEvent, TruthEventTape
from .observation import Observation, ObservationTape
from .state_machine import ConfirmationStateMachine
from .adapter import EventRuntimeAdapter


@dataclass
class ReplayState:
    """State during tape replay."""
    current_time: float = 0.0
    event_index: int = 0
    observation_index: int = 0
    confirmed_events: list = field(default_factory=list)
    pending_observations: list = field(default_factory=list)


class TapeReplayer:
    """Deterministic replayer for truth and observation tapes."""

    def __init__(
        self,
        truth_tape: TruthEventTape,
        observation_tape: ObservationTape | None = None,
        adapter: EventRuntimeAdapter | None = None,
    ) -> None:
        self.truth_tape = truth_tape
        self.observation_tape = observation_tape
        self.adapter = adapter or EventRuntimeAdapter()
        self.state = ReplayState()
        self._event_map: dict[str, TruthEvent] = {
            event.event_id: event for event in truth_tape.events
        }
        if observation_tape is not None:
            self._observation_map: dict[str, list[Observation]] = {}
            for obs in observation_tape.observations:
                self._observation_map.setdefault(obs.event_id, []).append(obs)

    def replay_until(self, target_time: float) -> list[Observation]:
        """Replay observations up to target time."""
        if self.observation_tape is None:
            return []

        observations = []
        while self.state.observation_index < len(self.observation_tape.observations):
            obs = self.observation_tape.observations[self.state.observation_index]
            if obs.received_at > target_time:
                break
            observations.append(obs)
            self.state.observation_index += 1

        # Process observations through state machine
        for obs in observations:
            self.adapter.process_observation(obs)

        return observations

    def get_truth_events_until(self, target_time: float) -> list[TruthEvent]:
        """Get truth events that occurred before target time."""
        events = []
        while self.state.event_index < len(self.truth_tape.events):
            event = self.truth_tape.events[self.state.event_index]
            if event.occurred_at > target_time:
                break
            events.append(event)
            self.state.event_index += 1
        return events

    def advance(self, dt: float) -> dict[str, Any]:
        """Advance replay by dt seconds."""
        target_time = self.state.current_time + dt
        self.state.current_time = target_time

        # Get new truth events
        new_events = self.get_truth_events_until(target_time)

        # Get new observations
        new_observations = self.replay_until(target_time)

        # Process any pending observations from adapter
        confirmed = self.adapter.advance_time(target_time)

        return {
            "time": target_time,
            "new_events": new_events,
            "new_observations": new_observations,
            "confirmed_events": confirmed,
            "pending_regions": list(self.adapter.belief.pending_regions),
        }

    def replay_full(self) -> dict[str, Any]:
        """Replay entire tape."""
        all_events = []
        all_observations = []
        all_confirmed = []

        # Determine replay duration from truth tape
        if self.truth_tape.events:
            max_time = max(event.occurred_at for event in self.truth_tape.events)
            max_time += 10.0  # Add buffer for observation delays
        else:
            max_time = 0.0

        current_time = 0.0
        step_size = 0.1  # 100ms steps

        while current_time <= max_time:
            result = self.advance(step_size)
            all_events.extend(result["new_events"])
            all_observations.extend(result["new_observations"])
            all_confirmed.extend(result["confirmed_events"])
            current_time = result["time"]

        return {
            "truth_events": all_events,
            "observations": all_observations,
            "confirmed_events": all_confirmed,
            "final_time": current_time,
            "belief_state": self.adapter.belief,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get replay statistics."""
        return {
            "truth_events_total": len(self.truth_tape.events),
            "observations_total": (
                len(self.observation_tape.observations)
                if self.observation_tape
                else 0
            ),
            "processed_observations": self.state.observation_index,
            "confirmed_events": len(self.adapter.belief.confirmed_events),
            "pending_regions": len(self.adapter.belief.pending_regions),
            "graph_version": self.adapter.belief.graph_version,
        }


class ReplayManager:
    """Manages multiple tape replays for comparison."""

    def __init__(self) -> None:
        self.replayers: dict[str, TapeReplayer] = {}

    def add_replay(
        self,
        name: str,
        truth_tape: TruthEventTape,
        observation_tape: ObservationTape | None = None,
    ) -> TapeReplayer:
        """Add a new tape replay."""
        replayer = TapeReplayer(truth_tape, observation_tape)
        self.replayers[name] = replayer
        return replayer

    def replay_all(self) -> dict[str, dict[str, Any]]:
        """Replay all tapes and return results."""
        results = {}
        for name, replayer in self.replayers.items():
            results[name] = replayer.replay_full()
        return results

    def compare_results(self) -> dict[str, Any]:
        """Compare results across replays."""
        results = self.replay_all()
        
        comparison = {}
        for name, result in results.items():
            comparison[name] = {
                "truth_events": len(result["truth_events"]),
                "observations": len(result["observations"]),
                "confirmed_events": len(result["confirmed_events"]),
                "final_time": result["final_time"],
            }
        
        return comparison
