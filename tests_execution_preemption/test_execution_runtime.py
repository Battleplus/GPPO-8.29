from __future__ import annotations

import unittest

from execution_preemption import (
    CommandStatus,
    CommunicationState,
    DecisionType,
    EventPriority,
    ExecutionRuntime,
    PreemptionController,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeInvariantError,
    ResumePolicy,
    StaleExecutionCommand,
    TaskRuntime,
    TaskState,
    UAVAvailability,
    UAVRuntime,
)


def event(
    event_id: str,
    event_type: RuntimeEventType,
    priority: EventPriority,
    *,
    at: float,
    task_id: str | None = None,
    uav_id: str | None = None,
    task_priority: int = 0,
    deadline: float | None = None,
    payload: dict | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=event_id,
        event_type=event_type,
        priority=priority,
        occurred_at=at,
        received_at=at,
        task_id=task_id,
        uav_id=uav_id,
        task_priority=task_priority,
        deadline=deadline,
        payload=payload or {},
    )


class ExecutionRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = ExecutionRuntime()
        self.controller = PreemptionController()
        self.runtime.add_uav(UAVRuntime("U0", energy_ratio=0.9))
        self.runtime.add_uav(UAVRuntime("U1", energy_ratio=0.8))

    def add_running_task(
        self,
        task_id: str = "T0",
        *,
        uav_id: str = "U0",
        priority: int = 10,
        preemptible: bool = True,
        task_type: str = "SEARCH",
    ) -> TaskRuntime:
        task = TaskRuntime(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            deadline=100.0,
            preemptible=preemptible,
        )
        self.runtime.add_task(task)
        self.runtime.assign_task(task_id, uav_id, at=0.0)
        return self.runtime.tasks[task_id]

    def test_continuous_progress_and_completion_release_owner(self) -> None:
        task = self.add_running_task()
        self.runtime.advance(4.0, now=4.0, work_rate_by_task={"T0": 0.1})
        self.assertAlmostEqual(task.progress, 0.4)
        self.assertAlmostEqual(task.remaining_work, 0.6)
        completed = self.runtime.advance(6.0, now=10.0, work_rate_by_task={"T0": 0.1})
        self.assertEqual(completed, ("T0",))
        self.assertEqual(task.state, TaskState.COMPLETED)
        self.assertIsNone(task.assigned_uav)
        self.assertEqual(self.runtime.uavs["U0"].availability, UAVAvailability.AVAILABLE)
        self.assertGreaterEqual(len(task.progress_history), 4)

    def test_urgent_arrival_preempts_at_40_percent_and_preserves_progress(self) -> None:
        self.add_running_task(priority=10)
        self.runtime.uavs["U1"].availability = UAVAvailability.FAILED
        self.runtime.advance(4.0, now=4.0, work_rate_by_task={"T0": 0.1})
        self.runtime.add_task(TaskRuntime("URGENT", "SEARCH", 90, 8.0))
        result = self.runtime.process_event_batch(
            [event(
                "E-urgent", RuntimeEventType.TASK_ARRIVAL, EventPriority.P1,
                at=4.0, task_id="URGENT", task_priority=90, deadline=8.0,
            )],
            self.controller,
            now=4.0,
        )
        self.assertEqual(result.decisions[0].decision, DecisionType.PREEMPT)
        self.assertEqual(self.runtime.tasks["T0"].state, TaskState.PREEMPTED)
        self.assertAlmostEqual(self.runtime.tasks["T0"].progress, 0.4)
        self.assertEqual(self.runtime.tasks["URGENT"].assigned_uav, "U0")
        self.assertEqual(self.runtime.uavs["U0"].active_task_id, "URGENT")

    def test_low_value_arrival_at_90_percent_is_queued(self) -> None:
        self.add_running_task(priority=50)
        self.runtime.uavs["U1"].availability = UAVAvailability.FAILED
        self.runtime.advance(9.0, now=9.0, work_rate_by_task={"T0": 0.1})
        self.runtime.add_task(TaskRuntime("LOW", "SEARCH", 5, 50.0))
        result = self.runtime.process_event_batch(
            [event(
                "E-low", RuntimeEventType.TASK_ARRIVAL, EventPriority.P3,
                at=9.0, task_id="LOW", task_priority=5,
            )],
            self.controller,
            now=9.0,
        )
        self.assertEqual(result.decisions[0].decision, DecisionType.QUEUE)
        self.assertEqual(self.runtime.tasks["T0"].state, TaskState.RUNNING)
        self.assertAlmostEqual(self.runtime.tasks["T0"].progress, 0.9)
        self.assertEqual(self.runtime.tasks["LOW"].state, TaskState.PENDING)

    def test_non_preemptible_task_is_not_displaced_by_p1(self) -> None:
        self.add_running_task(priority=1, preemptible=False, task_type="STRIKE")
        self.runtime.uavs["U1"].availability = UAVAvailability.FAILED
        self.runtime.add_task(TaskRuntime("NEW", "SEARCH", 100, 10.0))
        result = self.runtime.process_event_batch(
            [event(
                "E-p1", RuntimeEventType.TASK_ARRIVAL, EventPriority.P1,
                at=1.0, task_id="NEW", task_priority=100,
            )],
            self.controller,
            now=1.0,
        )
        self.assertEqual(result.decisions[0].decision, DecisionType.QUEUE)
        self.assertEqual(self.runtime.tasks["T0"].state, TaskState.RUNNING)

    def test_low_energy_forces_rtb_and_releases_task(self) -> None:
        task = self.add_running_task(preemptible=False)
        task.set_progress(0.5, at=2.0, reason="fixture")
        result = self.runtime.process_event_batch(
            [event(
                "E-energy", RuntimeEventType.UAV_LOW_ENERGY, EventPriority.P2,
                at=3.0, uav_id="U0",
            )],
            self.controller,
            now=3.0,
        )
        self.assertEqual(result.decisions[0].decision, DecisionType.RTB)
        task = self.runtime.tasks["T0"]
        self.assertEqual(task.state, TaskState.PREEMPTED)
        self.assertAlmostEqual(task.progress, 0.5)
        self.assertEqual(self.runtime.uavs["U0"].availability, UAVAvailability.RETURNING)

    def test_communication_loss_migrates_with_frozen_retention(self) -> None:
        task = self.add_running_task()
        task.set_progress(0.5, at=2.0, reason="fixture")
        result = self.runtime.process_event_batch(
            [event(
                "E-comm", RuntimeEventType.UAV_COMM_LOST, EventPriority.P2,
                at=3.0, uav_id="U0",
            )],
            self.controller,
            now=3.0,
        )
        self.assertEqual(result.decisions[0].decision, DecisionType.MIGRATE)
        task = self.runtime.tasks["T0"]
        self.assertEqual(task.assigned_uav, "U1")
        self.assertAlmostEqual(task.progress, 0.45)
        self.assertEqual(task.interruption_count, 1)
        self.assertEqual(self.runtime.uavs["U0"].communication_state, CommunicationState.LOST)
        self.assertEqual(self.runtime.uavs["U0"].availability, UAVAvailability.COMMUNICATION_LOST)

    def test_communication_loss_without_replacement_pauses(self) -> None:
        task = self.add_running_task()
        self.runtime.uavs["U1"].availability = UAVAvailability.FAILED
        result = self.runtime.process_event_batch(
            [event(
                "E-comm", RuntimeEventType.UAV_COMM_LOST, EventPriority.P2,
                at=1.0, uav_id="U0",
            )],
            self.controller,
            now=1.0,
        )
        self.assertEqual(result.decisions[0].decision, DecisionType.PAUSE)
        task = self.runtime.tasks["T0"]
        self.assertEqual(task.state, TaskState.PAUSED)
        self.assertEqual(task.assigned_uav, "U0")
        self.assertEqual(self.runtime.uavs["U0"].availability, UAVAvailability.COMMUNICATION_LOST)

    def test_cancellation_releases_resource(self) -> None:
        self.add_running_task()
        result = self.runtime.process_event_batch(
            [event(
                "E-cancel", RuntimeEventType.TASK_CANCELLED, EventPriority.P3,
                at=1.0, task_id="T0",
            )],
            self.controller,
            now=1.0,
        )
        self.assertEqual(result.decisions[0].decision, DecisionType.ABORT)
        self.assertEqual(self.runtime.tasks["T0"].state, TaskState.CANCELLED)
        self.assertIsNone(self.runtime.uavs["U0"].active_task_id)

    def test_two_same_priority_events_have_deterministic_order(self) -> None:
        self.runtime.add_task(TaskRuntime("A", "SEARCH", 20, 20.0))
        self.runtime.add_task(TaskRuntime("B", "SEARCH", 30, 10.0))
        events = [
            event("E-A", RuntimeEventType.TASK_ARRIVAL, EventPriority.P1,
                  at=1.0, task_id="A", task_priority=20, deadline=20.0),
            event("E-B", RuntimeEventType.TASK_ARRIVAL, EventPriority.P1,
                  at=1.0, task_id="B", task_priority=30, deadline=10.0),
        ]
        result = self.runtime.process_event_batch(events, self.controller, now=1.0)
        self.assertEqual([item.event_id for item in result.decisions], ["E-B", "E-A"])
        self.assertEqual(result.graph_version_after, result.graph_version_before + 1)

    def test_priority_change_is_traceable_without_interrupting(self) -> None:
        task = self.add_running_task(priority=10)
        result = self.runtime.process_event_batch(
            [event(
                "E-priority", RuntimeEventType.TASK_PRIORITY_CHANGED, EventPriority.P3,
                at=2.0, task_id="T0", task_priority=80,
                payload={"new_priority": 80},
            )],
            self.controller,
            now=2.0,
        )
        self.assertEqual(result.decisions[0].decision, DecisionType.CONTINUE)
        task = self.runtime.tasks["T0"]
        self.assertEqual(task.priority, 80)
        self.assertEqual(task.state, TaskState.RUNNING)
        self.assertIn("priority changed", task.progress_history[-1].reason)

    def test_new_event_revokes_inflight_command_and_old_ack_cannot_revive_it(self) -> None:
        self.runtime.add_task(TaskRuntime("T0", "SEARCH", 10, 100.0))
        command = self.runtime.issue_assignment_command(
            "C0", "T0", "U0", expected_graph_version=0, at=0.0,
        )
        result = self.runtime.process_event_batch(
            [event(
                "E-cancel", RuntimeEventType.TASK_CANCELLED, EventPriority.P3,
                at=1.0, task_id="T0",
            )],
            self.controller,
            now=1.0,
        )
        self.assertEqual(result.revoked_commands, ("C0",))
        command = self.runtime.commands["C0"]
        self.assertEqual(command.status, CommandStatus.REVOKED)
        with self.assertRaises(StaleExecutionCommand):
            self.runtime.acknowledge_command(
                "C0", graph_version=0, fencing_token=command.fencing_token, at=1.1,
            )
        self.assertEqual(self.runtime.tasks["T0"].state, TaskState.CANCELLED)

    def test_unrelated_graph_change_still_rejects_stale_ack(self) -> None:
        self.runtime.add_task(TaskRuntime("T0", "SEARCH", 10, 100.0))
        command = self.runtime.issue_assignment_command(
            "C0", "T0", "U0", expected_graph_version=0, at=0.0,
        )
        self.runtime.add_task(TaskRuntime("T1", "SEARCH", 1, 100.0))
        self.runtime.process_event_batch(
            [event(
                "E-update", RuntimeEventType.TASK_PRIORITY_CHANGED, EventPriority.P3,
                at=1.0, task_id="T1", task_priority=2,
                payload={"new_priority": 2},
            )],
            self.controller,
            now=1.0,
        )
        with self.assertRaises(StaleExecutionCommand):
            self.runtime.acknowledge_command(
                "C0", graph_version=0, fencing_token=command.fencing_token, at=1.1,
            )
        command = self.runtime.commands["C0"]
        self.assertEqual(command.status, CommandStatus.REJECTED)

    def test_atomic_transaction_rolls_back_on_invalid_event(self) -> None:
        self.add_running_task()
        before_version = self.runtime.graph_version
        before_state = self.runtime.tasks["T0"].state
        with self.assertRaises(ValueError):
            self.runtime.process_event_batch(
                [event(
                    "E-bad", RuntimeEventType.TASK_CANCELLED, EventPriority.P3,
                    at=1.0, task_id="missing",
                )],
                self.controller,
                now=1.0,
            )
        self.assertEqual(self.runtime.graph_version, before_version)
        self.assertEqual(self.runtime.tasks["T0"].state, before_state)

    def test_double_booking_is_rejected(self) -> None:
        self.add_running_task()
        self.runtime.add_task(TaskRuntime("T1", "SEARCH", 10, 100.0))
        with self.assertRaises(RuntimeInvariantError):
            self.runtime.assign_task("T1", "U0", at=1.0)

    def test_energy_cost_is_explicit_and_bounded(self) -> None:
        self.add_running_task()
        self.runtime.advance(
            1.0,
            now=1.0,
            work_rate_by_task={"T0": 0.1},
            energy_cost_by_uav={"U0": 0.15},
        )
        self.assertAlmostEqual(self.runtime.uavs["U0"].energy_ratio, 0.75)

    def test_current_command_ack_assigns_exactly_one_owner(self) -> None:
        self.runtime.add_task(TaskRuntime("T0", "SEARCH", 10, 100.0))
        command = self.runtime.issue_assignment_command(
            "C0", "T0", "U0", expected_graph_version=0, at=0.0,
        )
        acknowledged = self.runtime.acknowledge_command(
            "C0",
            graph_version=0,
            fencing_token=command.fencing_token,
            at=0.1,
        )
        self.assertEqual(acknowledged.status, CommandStatus.ACKED)
        self.assertEqual(self.runtime.tasks["T0"].assigned_uav, "U0")
        self.assertEqual(self.runtime.uavs["U0"].active_task_id, "T0")

    def test_old_fencing_token_cannot_ack_new_command(self) -> None:
        self.runtime.add_task(TaskRuntime("T0", "SEARCH", 10, 100.0))
        old = self.runtime.issue_assignment_command(
            "C0", "T0", "U0", expected_graph_version=0, at=0.0,
        )
        self.runtime.acknowledge_command(
            "C0", graph_version=0, fencing_token=old.fencing_token, at=0.1,
        )
        self.runtime.preempt_task("T0", at=1.0, reason="fixture")
        new = self.runtime.issue_assignment_command(
            "C1", "T0", "U1", expected_graph_version=0, at=1.1,
        )
        self.assertGreater(new.fencing_token, old.fencing_token)
        with self.assertRaises(StaleExecutionCommand):
            self.runtime.acknowledge_command(
                "C1", graph_version=0, fencing_token=old.fencing_token, at=1.2,
            )
        self.assertIsNone(self.runtime.tasks["T0"].assigned_uav)

    def test_same_uav_resume_preserves_all_progress(self) -> None:
        task = self.add_running_task()
        task.set_progress(0.6, at=2.0, reason="fixture")
        self.runtime.pause_task("T0", at=3.0, reason="temporary hold")
        self.runtime.resume_task("T0", "U0", at=4.0)
        task = self.runtime.tasks["T0"]
        self.assertEqual(task.state, TaskState.RUNNING)
        self.assertAlmostEqual(task.progress, 0.6)
        self.assertEqual(task.interruption_count, 1)

    def test_same_uav_resume_policy_rejects_different_uav(self) -> None:
        self.runtime.add_task(TaskRuntime(
            "T0", "SEARCH", 10, 100.0,
            resume_policy=ResumePolicy.SAME_UAV,
        ))
        self.runtime.assign_task("T0", "U0", at=0.0)
        self.runtime.preempt_task("T0", at=1.0, reason="urgent event")
        with self.assertRaises(RuntimeInvariantError):
            self.runtime.resume_task("T0", "U1", at=2.0)

    def test_duplicate_event_replay_is_idempotent(self) -> None:
        self.runtime.add_task(TaskRuntime("T0", "SEARCH", 10, 100.0))
        item = event(
            "E-arrive", RuntimeEventType.TASK_ARRIVAL, EventPriority.P3,
            at=1.0, task_id="T0", task_priority=10,
        )
        first = self.runtime.process_event_batch([item], self.controller, now=1.0)
        second = self.runtime.process_event_batch([item], self.controller, now=1.0)
        self.assertEqual(first.graph_version_after, 1)
        self.assertEqual(second.graph_version_before, 1)
        self.assertEqual(second.graph_version_after, 1)
        self.assertEqual(second.decisions, ())

    def test_reused_event_id_with_different_content_is_rejected(self) -> None:
        self.runtime.add_task(TaskRuntime("T0", "SEARCH", 10, 100.0))
        original = event(
            "E-same", RuntimeEventType.TASK_ARRIVAL, EventPriority.P3,
            at=1.0, task_id="T0", task_priority=10,
        )
        self.runtime.process_event_batch([original], self.controller, now=1.0)
        conflicting = event(
            "E-same", RuntimeEventType.TASK_CANCELLED, EventPriority.P3,
            at=2.0, task_id="T0",
        )
        with self.assertRaises(ValueError):
            self.runtime.process_event_batch([conflicting], self.controller, now=2.0)


if __name__ == "__main__":
    unittest.main()
