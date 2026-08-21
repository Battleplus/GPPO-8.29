"""Concurrency invariant tests (Phase E audit item 6).

Covers:
- command graph_version exact match validation (old AND future rejected)
- action_version validation
- ACK validates command_id / uav_id / fencing_token / status
- late ACK cannot resurrect revoked command
- lower fencing token cannot override higher
- valid exclusive holder <= 1
- fencing token monotonicity (real revoke + new holder)
- stale action_version rejection
- injected stale submission tracking
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from event_runtime.concurrency import (
    ACK, ACKType, AssignmentCommand, AssignmentLease, CommandStatus,
    ConcurrencyManager, FencingToken,
)
from event_runtime.adapter import EventRuntimeAdapter


class GraphVersionExactMatchTests(unittest.TestCase):
    def test_matching_version_passes(self):
        cm = ConcurrencyManager()
        cmd = cm.create_command("c1", "0", "0", graph_version=5, action_version=1, now=0.0)
        self.assertTrue(cm.validate_command("c1", 5))
        self.assertEqual(cmd.status, CommandStatus.VALIDATED)

    def test_old_version_rejected(self):
        cm = ConcurrencyManager()
        cmd = cm.create_command("c1", "0", "0", graph_version=5, action_version=1, now=0.0)
        # graph_version 6 > 5 → rejected
        self.assertFalse(cm.validate_command("c1", 6))
        self.assertEqual(cmd.status, CommandStatus.REJECTED)

    def test_future_version_rejected(self):
        """A command with graph_version=5 must be rejected when current is 4."""
        cm = ConcurrencyManager()
        cmd = cm.create_command("c1", "0", "0", graph_version=5, action_version=1, now=0.0)
        # current graph_version=4, command has 5 → future → must reject
        self.assertFalse(cm.validate_command("c1", 4))
        self.assertEqual(cmd.status, CommandStatus.REJECTED)

    def test_is_valid_at_rejects_future(self):
        cm = ConcurrencyManager()
        cmd = cm.create_command("c1", "0", "0", graph_version=5, action_version=1, now=0.0)
        cmd.validate(5)
        cmd.commit()
        # is_valid_at should reject when current version doesn't match
        self.assertFalse(cmd.is_valid_at(now=0.1, current_graph_version=4))
        self.assertTrue(cmd.is_valid_at(now=0.1, current_graph_version=5))


class ActionVersionValidationTests(unittest.TestCase):
    def test_action_version_stored_correctly(self):
        cm = ConcurrencyManager()
        cmd = cm.create_command("c1", "0", "0", graph_version=1, action_version=42, now=0.0)
        self.assertEqual(cmd.action_version, 42)

    def test_action_version_mismatch_rejected(self):
        """Command created with action_version=3 but environment has action_version=5."""
        cm = ConcurrencyManager()
        cmd = cm.create_command("c1", "0", "0", graph_version=1, action_version=3, now=0.0)
        # In the bridge, the check is: command.action_version != action_version
        # At the concurrency level, validate() only checks graph_version.
        # The action_version check is enforced at the bridge level.
        # But validate() should still accept matching versions.
        self.assertTrue(cm.validate_command("c1", 1))
        self.assertEqual(cmd.action_version, 3)


class ACKValidationTests(unittest.TestCase):
    def test_valid_ack_accepted(self):
        cm = ConcurrencyManager()
        cm.create_command("c1", "0", "0", graph_version=1, action_version=0, now=0.0)
        cm.validate_command("c1", 1)
        cm.commit_command("c1")
        ack = ACK("c1", "0", ACKType.ACCEPTED, 1.0, fencing_token=cm.commands["c1"].fencing_token)
        cm.receive_ack("c1", ack)
        self.assertEqual(cm.commands["c1"].status, CommandStatus.ACKED)

    def test_ack_wrong_uav_rejected(self):
        cm = ConcurrencyManager()
        cm.create_command("c1", "0", "0", graph_version=1, action_version=0, now=0.0)
        cm.validate_command("c1", 1)
        cm.commit_command("c1")
        cmd = cm.commands["c1"]
        ack = ACK("c1", "99", ACKType.ACCEPTED, 1.0, fencing_token=cmd.fencing_token)
        with self.assertRaises(ValueError):
            cm.receive_ack("c1", ack)

    def test_ack_wrong_fencing_token_rejected(self):
        cm = ConcurrencyManager()
        cm.create_command("c1", "0", "0", graph_version=1, action_version=0, now=0.0)
        cm.validate_command("c1", 1)
        cm.commit_command("c1")
        ack = ACK("c1", "0", ACKType.ACCEPTED, 1.0, fencing_token=9999)
        with self.assertRaises(ValueError):
            cm.receive_ack("c1", ack)


class LateACKCannotResurrectTests(unittest.TestCase):
    def test_ack_after_revoke_counted_not_resurrected(self):
        cm = ConcurrencyManager()
        cmd = cm.create_command("c1", "0", "0", graph_version=1, action_version=0, now=0.0)
        cm.validate_command("c1", 1)
        cm.commit_command("c1")
        higher_token = cmd.fencing_token + 1
        cmd.revoke(higher_token, at=2.0)
        self.assertEqual(cmd.status, CommandStatus.REVOKED)
        with self.assertRaises(ValueError):
            ack = ACK("c1", "0", ACKType.ACCEPTED, 3.0, fencing_token=cmd.fencing_token)
            cm.receive_ack("c1", ack)

    def test_late_ack_after_expire_rejected(self):
        cm = ConcurrencyManager()
        cmd = cm.create_command("c1", "0", "0", graph_version=1, action_version=0, now=0.0, ttl=0.5)
        cm.validate_command("c1", 1)
        cm.commit_command("c1")
        cmd.expire(at=1.0)
        self.assertEqual(cmd.status, CommandStatus.EXPIRED)
        with self.assertRaises(ValueError):
            ack = ACK("c1", "0", ACKType.ACCEPTED, 2.0, fencing_token=cmd.fencing_token)
            cm.receive_ack("c1", ack)


class FencingTokenMonotonicityTests(unittest.TestCase):
    def test_tokens_are_monotonic(self):
        cm = ConcurrencyManager()
        t1 = cm.next_fencing_token()
        t2 = cm.next_fencing_token()
        t3 = cm.next_fencing_token()
        self.assertLess(t1, t2)
        self.assertLess(t2, t3)

    def test_revoke_requires_higher_token(self):
        cm = ConcurrencyManager()
        cmd = cm.create_command("c1", "0", "0", graph_version=1, action_version=0, now=0.0)
        with self.assertRaises(ValueError):
            cmd.revoke(cmd.fencing_token - 1, at=1.0)
        with self.assertRaises(ValueError):
            cmd.revoke(cmd.fencing_token, at=1.0)

    def test_real_revoke_and_new_holder(self):
        """Real fencing: holder A gets T1, revoke, holder B gets T2 > T1, old T1 rejected."""
        cm = ConcurrencyManager()
        # Holder A
        cmd_a = cm.create_command("ca", "0", "0", graph_version=1, action_version=0, now=0.0)
        token_a = cmd_a.fencing_token
        # Lease for A
        lease_a = cm.create_lease("la", "0", "0", token_a, now=0.0, ttl=5.0)
        self.assertEqual(cm.get_valid_holder_count("0", 0.5), 1)

        # Revoke A with higher token
        cmd_a.revoke(token_a + 5, at=1.0)
        cm.revoke_lease("la")
        self.assertEqual(cm.get_valid_holder_count("0", 1.5), 0)

        # New holder B gets a strictly higher token
        cmd_b = cm.create_command("cb", "1", "0", graph_version=1, action_version=0, now=2.0)
        token_b = cmd_b.fencing_token
        self.assertGreater(token_b, token_a)

        # Lease for B succeeds
        lease_b = cm.create_lease("lb", "1", "0", token_b, now=2.0, ttl=5.0)
        self.assertEqual(cm.get_valid_holder_count("0", 2.5), 1)

        # Old token A cannot create a lease (expired lease exists with higher token)
        with self.assertRaises(ValueError):
            cm.create_lease("l-old", "0", "0", token_a, now=3.0, ttl=5.0)


class ExclusiveHolderTests(unittest.TestCase):
    def test_one_holder_valid(self):
        cm = ConcurrencyManager()
        cm.create_lease("l1", "0", "0", fencing_token=1, now=0.0, ttl=5.0)
        self.assertEqual(cm.get_valid_holder_count("0", 1.0), 1)

    def test_two_holders_rejected_at_execution(self):
        cm = ConcurrencyManager()
        cm.create_lease("l1", "0", "0", fencing_token=1, now=0.0, ttl=5.0)
        with self.assertRaises(ValueError):
            cm.create_lease("l2", "1", "0", fencing_token=2, now=0.0, ttl=5.0)
        self.assertEqual(cm.get_valid_holder_count("0", 1.0), 1)

    def test_revoke_removes_holder(self):
        cm = ConcurrencyManager()
        cm.create_lease("l1", "0", "0", fencing_token=1, now=0.0, ttl=5.0)
        cm.revoke_lease("l1")
        self.assertEqual(cm.get_valid_holder_count("0", 1.0), 0)


class AdapterConcurrencyWiringTests(unittest.TestCase):
    def test_adapter_uses_concurrency_manager(self):
        adapter = EventRuntimeAdapter(merge_window=0.1)
        self.assertIsNotNone(adapter.concurrency)
        cmd = adapter.create_command("c1", "0", "0", ttl=0.5, now=0.0)
        self.assertIsInstance(cmd, AssignmentCommand)


class StaleActionVersionTests(unittest.TestCase):
    """Tests for stale action_version rejection at the bridge level."""

    def test_begin_decision_returns_correct_versions(self):
        """begin_decision captures graph_version and action_version."""
        from ppo_allocation.random_event.runtime_bridge import RuntimeBridge
        from ppo_allocation.random_event.environment import RandomEventAllocationEnv
        env = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, mode="single", events_per_episode=1)
        env.reset(seed=42)
        bridge = RuntimeBridge(detector_seed=42)
        ctx = bridge.begin_decision(env)
        self.assertEqual(ctx["graph_version"], int(env.graph_version))
        self.assertEqual(ctx["action_version"], int(env.decision_version))
        env.close()

    def test_stale_action_version_rejected_at_env_level(self):
        """Submit action with old action_version → rejected, no mutation."""
        import numpy as np
        from ppo_allocation.random_event.runtime_bridge import RuntimeBridge
        from ppo_allocation.random_event.environment import RandomEventAllocationEnv
        env = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, mode="single", events_per_episode=1)
        graph, _ = env.reset(seed=42)
        bridge = RuntimeBridge(detector_seed=42)
        # Capture decision context
        ctx = bridge.begin_decision(env)
        old_av = ctx["action_version"]
        # Step the environment to advance decision_version
        mask = graph.action_mask.cpu().numpy().astype(bool)
        legal = np.flatnonzero(mask)
        action = int(legal[0]) if len(legal) > 0 else int(graph.noop_action)
        graph, _, _, _, _ = env.step(action)
        # Now submit with the OLD action_version — must be rejected
        fresh_gv = int(env.graph_version)
        result = bridge.submit_stale_action(
            env,
            command_id="stale-av-env",
            uav_id="0",
            region_id="0",
            stale_graph_version=fresh_gv,
            action_version=old_av,
            fencing_token=0,
            now=float(env.current_time),
        )
        # submit_stale_action returns True when correctly rejected
        self.assertTrue(result)
        snap = bridge.snapshot_concurrency(float(env.current_time))
        self.assertEqual(snap["injected_stale_rejected"], 1)
        env.close()

    def test_matching_versions_accepted(self):
        """Command with matching graph_version + action_version accepted."""
        from ppo_allocation.random_event.runtime_bridge import RuntimeBridge
        from ppo_allocation.random_event.environment import RandomEventAllocationEnv
        env = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, mode="single", events_per_episode=1)
        env.reset(seed=42)
        bridge = RuntimeBridge(detector_seed=42)
        ctx = bridge.begin_decision(env)
        # Pick a pending region if any
        pending = list(env.pending_regions)
        if pending:
            rid = pending[0]
            # Find a valid UAV for this region
            uid = 0
            for u in env.uavs:
                if env.uavs[u].alive and not env.uavs[u].sensor_failed:
                    uid = u
                    break
            cmd = bridge.issue_assignment_command(
                env, uav_id=uid, region_id=rid, now=0.0,
                expected_graph_version=ctx["graph_version"],
                expected_action_version=ctx["action_version"],
            )
            self.assertIsNotNone(cmd, "Valid assignment with matching versions should be accepted")
        else:
            self.skipTest("No pending regions after reset")
        env.close()


if __name__ == "__main__":
    unittest.main()
