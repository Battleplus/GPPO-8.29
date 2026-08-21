"""Metrics tracking for event runtime and experiment evaluation.

This module implements the metrics collection and calculation required for
evaluating event recovery performance, communication efficiency, and
mechanism diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EventMetrics:
    """Metrics for a single event."""
    event_id: str
    event_type: str
    occurred_at: float
    confirmed_at: float | None = None
    handled_at: float | None = None
    resolved_at: float | None = None
    recovery_delay: float | None = None
    recovery_decision_steps: int = 0
    success: bool = False
    infeasible_reason: str | None = None


@dataclass
class EpisodeMetrics:
    """Metrics for a complete episode."""
    episode_id: str
    total_reward: float = 0.0
    episode_return: float = 0.0
    decision_count: int = 0
    event_count: int = 0
    successful_events: int = 0
    failed_events: int = 0
    temporary_infeasible_count: int = 0
    final_infeasible_count: int = 0
    cumulative_vacancy_time: float = 0.0
    distance_cost: float = 0.0
    load_gap: float = 0.0
    switch_count: int = 0
    trigger_count: int = 0
    inference_count: int = 0
    communication_bytes: int = 0
    merged_event_count: int = 0
    stale_action_rejection_count: int = 0
    event_metrics: list[EventMetrics] = field(default_factory=list)


@dataclass
class MechanismMetrics:
    """Mechanism and diagnostic metrics."""
    valid_action_count: int = 0
    action_mask_ratio: float = 0.0
    pre_mask_invalid_probability: float = 0.0
    actor_entropy: float = 0.0
    critic_value_loss: float = 0.0
    explained_variance: float = 0.0
    approximate_kl: float = 0.0
    ppo_clip_fraction: float = 0.0
    gate_mean: float = 0.0
    gate_std: float = 0.0
    gate_p10: float = 0.0
    gate_p50: float = 0.0
    gate_p90: float = 0.0
    gate_gradient_norm: float = 0.0


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across episodes."""
    mean_return: float = 0.0
    std_return: float = 0.0
    mean_success_rate: float = 0.0
    mean_recovery_delay: float = 0.0
    mean_cumulative_vacancy: float = 0.0
    mean_distance_cost: float = 0.0
    mean_load_gap: float = 0.0
    mean_switch_count: float = 0.0
    mean_inference_latency_ms: float = 0.0
    oracle_regret: float = 0.0
    episodes: int = 0


class MetricsTracker:
    """Tracks metrics during experiment execution."""

    def __init__(self) -> None:
        self.episodes: list[EpisodeMetrics] = []
        self.mechanism = MechanismMetrics()
        self._current_episode: EpisodeMetrics | None = None

    def start_episode(self, episode_id: str) -> None:
        """Start tracking a new episode."""
        self._current_episode = EpisodeMetrics(episode_id=episode_id)

    def record_event(
        self,
        event_id: str,
        event_type: str,
        occurred_at: float,
        confirmed_at: float | None = None,
        handled_at: float | None = None,
        resolved_at: float | None = None,
        success: bool = False,
        infeasible_reason: str | None = None,
    ) -> None:
        """Record event metrics."""
        if self._current_episode is None:
            return

        recovery_delay = None
        if resolved_at is not None and occurred_at is not None:
            recovery_delay = resolved_at - occurred_at

        event_metrics = EventMetrics(
            event_id=event_id,
            event_type=event_type,
            occurred_at=occurred_at,
            confirmed_at=confirmed_at,
            handled_at=handled_at,
            resolved_at=resolved_at,
            recovery_delay=recovery_delay,
            success=success,
            infeasible_reason=infeasible_reason,
        )
        self._current_episode.event_metrics.append(event_metrics)
        self._current_episode.event_count += 1

        if success:
            self._current_episode.successful_events += 1
        else:
            self._current_episode.failed_events += 1

    def record_decision(
        self,
        reward: float,
        decision_steps: int = 1,
        communication_bytes: int = 0,
        stale_rejection: bool = False,
    ) -> None:
        """Record decision metrics."""
        if self._current_episode is None:
            return

        self._current_episode.total_reward += reward
        self._current_episode.decision_count += decision_steps
        self._current_episode.communication_bytes += communication_bytes

        if stale_rejection:
            self._current_episode.stale_action_rejection_count += 1

    def record_vacancy(self, duration: float) -> None:
        """Record vacancy duration."""
        if self._current_episode is None:
            return
        self._current_episode.cumulative_vacancy_time += duration

    def record_distance(self, cost: float) -> None:
        """Record distance cost."""
        if self._current_episode is None:
            return
        self._current_episode.distance_cost += cost

    def record_load_gap(self, gap: float) -> None:
        """Record load gap."""
        if self._current_episode is None:
            return
        self._current_episode.load_gap += gap

    def record_switch(self) -> None:
        """Record a task switch."""
        if self._current_episode is None:
            return
        self._current_episode.switch_count += 1

    def record_trigger(self) -> None:
        """Record a communication trigger."""
        if self._current_episode is None:
            return
        self._current_episode.trigger_count += 1

    def record_inference(self) -> None:
        """Record an inference call."""
        if self._current_episode is None:
            return
        self._current_episode.inference_count += 1

    def record_merged_event(self) -> None:
        """Record a merged event."""
        if self._current_episode is None:
            return
        self._current_episode.merged_event_count += 1

    def record_temporary_infeasible(self) -> None:
        """Record temporary infeasibility."""
        if self._current_episode is None:
            return
        self._current_episode.temporary_infeasible_count += 1

    def record_final_infeasible(self) -> None:
        """Record final infeasibility."""
        if self._current_episode is None:
            return
        self._current_episode.final_infeasible_count += 1

    def end_episode(self) -> EpisodeMetrics:
        """End current episode and return metrics."""
        if self._current_episode is None:
            raise ValueError("No episode in progress")

        # Calculate episode return
        self._current_episode.episode_return = self._current_episode.total_reward

        self.episodes.append(self._current_episode)
        episode = self._current_episode
        self._current_episode = None
        return episode

    def aggregate(self) -> AggregatedMetrics:
        """Aggregate metrics across all episodes."""
        if not self.episodes:
            return AggregatedMetrics()

        returns = [ep.episode_return for ep in self.episodes]
        success_rates = [
            ep.successful_events / max(ep.event_count, 1)
            for ep in self.episodes
        ]
        recovery_delays = [
            er.recovery_delay
            for ep in self.episodes
            for er in ep.event_metrics
            if er.recovery_delay is not None
        ]

        import statistics

        return AggregatedMetrics(
            mean_return=statistics.mean(returns) if returns else 0.0,
            std_return=statistics.stdev(returns) if len(returns) > 1 else 0.0,
            mean_success_rate=statistics.mean(success_rates) if success_rates else 0.0,
            mean_recovery_delay=statistics.mean(recovery_delays) if recovery_delays else 0.0,
            mean_cumulative_vacancy=statistics.mean(
                [ep.cumulative_vacancy_time for ep in self.episodes]
            ),
            mean_distance_cost=statistics.mean(
                [ep.distance_cost for ep in self.episodes]
            ),
            mean_load_gap=statistics.mean([ep.load_gap for ep in self.episodes]),
            mean_switch_count=statistics.mean(
                [ep.switch_count for ep in self.episodes]
            ),
            episodes=len(self.episodes),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        aggregated = self.aggregate()
        return {
            "episodes": len(self.episodes),
            "aggregated": {
                "mean_return": aggregated.mean_return,
                "std_return": aggregated.std_return,
                "mean_success_rate": aggregated.mean_success_rate,
                "mean_recovery_delay": aggregated.mean_recovery_delay,
                "mean_cumulative_vacancy": aggregated.mean_cumulative_vacancy,
                "mean_distance_cost": aggregated.mean_distance_cost,
                "mean_load_gap": aggregated.mean_load_gap,
                "mean_switch_count": aggregated.mean_switch_count,
            },
            "mechanism": {
                "valid_action_count": self.mechanism.valid_action_count,
                "action_mask_ratio": self.mechanism.action_mask_ratio,
                "pre_mask_invalid_probability": self.mechanism.pre_mask_invalid_probability,
                "actor_entropy": self.mechanism.actor_entropy,
                "critic_value_loss": self.mechanism.critic_value_loss,
                "explained_variance": self.mechanism.explained_variance,
                "approximate_kl": self.mechanism.approximate_kl,
                "ppo_clip_fraction": self.mechanism.ppo_clip_fraction,
                "gate_mean": self.mechanism.gate_mean,
                "gate_std": self.mechanism.gate_std,
                "gate_p10": self.mechanism.gate_p10,
                "gate_p50": self.mechanism.gate_p50,
                "gate_p90": self.mechanism.gate_p90,
                "gate_gradient_norm": self.mechanism.gate_gradient_norm,
            },
        }
