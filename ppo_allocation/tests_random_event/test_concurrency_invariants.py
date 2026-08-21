"""Concurrency invariant tests (Phase E audit item 6).

Covers:
- command graph_version exact match validation
- action_version validation
- ACK validates command_id / uav_id / fencing_token / status
- late ACK cannot resurrect revoked command
- lower fencing token cannot override higher
- valid exclusive holder <= 1
- fencing token monotonicity
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

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
        self.assertFalse(cm.validate_command("c1", 6))
        self.assertEqual(cmd.status, CommandStatus.REJECTED)


class ActionVersionValidationTests(unittest.TestCase):
    def test_action_version_stored_correctly(self):
        cm = ConcurrencyManager()
        cmd = cm.create_command("c1", "0", "0", graph_version=1, action_version=42, now=0.0)
        self.assertEqual(cmd.action_version, 42)


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
        # Revoke with higher fencing token
        higher_token = cmd.fencing_token + 1
        cmd.revoke(higher_token, at=2.0)
        self.assertEqual(cmd.status, CommandStatus.REVOKED)
        # Late ACK attempt - should fail in our bridge layer, but at
        # concurrency level receive_ack may raise ValueError for wrong status
        with self.assertRaises(ValueError):
            ack = ACK("c1", "0", ACKType.ACCEPTED, 3.0, fencing_token=cmd.fencing_token)
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


if __name__ == "__main__":
    unittest.main()
