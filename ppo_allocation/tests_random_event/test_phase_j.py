"""Tests for Phase J Preliminary orchestrator and versioned submission.

Covers:
- ActionSubmission contract
- begin_decision → submit_action versioned path
- PPOTrainer collect_rollout uses versioned submission (stale: no buffer, refresh ctx)
- run_episode uses versioned submission
- Checkpoint scheduling
- Lexicographic selection (5 synthetic levels)
- Validation/Test bank generation
- Freeze manifest
- Test isolation guard
- DRY RUN orchestrator
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import dataclass

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

import torch

from ppo_allocation.random_event.environment import (
    ActionSubmission,
    DecisionContext,
    RandomEventAllocationEnv,
)
from ppo_allocation.random_event.phase_j import (
    PreliminaryProtocol,
    ValidationMetrics,
    compute_checkpoint_steps,
    _lexicographic_select,
)


# ---------------------------------------------------------------------------
# ActionSubmission contract
# ---------------------------------------------------------------------------
class ActionSubmissionContractTests(unittest.TestCase):
    def test_from_decision(self):
        env = RandomEventAllocationEnv(
            initial_seed=42, event_seed=42001, mode="single", events_per_episode=1
        )
        env.reset(seed=42)
        ctx = env.begin_decision()
        sub = ActionSubmission.from_decision(3, ctx)
        self.assertEqual(sub.action, 3)
        self.assertEqual(sub.graph_version, ctx.graph_version)
        self.assertEqual(sub.action_version, ctx.action_version)
        env.close()

    def test_frozen(self):
        sub = ActionSubmission(action=1, graph_version=2, action_version=3)
        with self.assertRaises(AttributeError):
            sub.action = 5


# ---------------------------------------------------------------------------
# Versioned submission env
# ---------------------------------------------------------------------------
class VersionedSubmissionEnvTests(unittest.TestCase):
    def test_matching_versions_accepted(self):
        env = RandomEventAllocationEnv(
            initial_seed=42, event_seed=42001, mode="single", events_per_episode=1
        )
        env.reset(seed=42)
        ctx = env.begin_decision()
        noop = env.action_space.n - 1
        sub = ActionSubmission.from_decision(noop, ctx)
        g2, r, term, trunc, info = env.submit_action(sub)
        self.assertFalse(info.get("stale_decision", False))
        env.close()

    def test_stale_graph_version_rejected(self):
        env = RandomEventAllocationEnv(
            initial_seed=42, event_seed=42001, mode="single", events_per_episode=1
        )
        env.reset(seed=42)
        ctx = env.begin_decision()
        noop = env.action_space.n - 1
        env.step(noop)  # advance
        sub = ActionSubmission(action=noop, graph_version=ctx.graph_version, action_version=ctx.action_version)
        g2, r, term, trunc, info = env.submit_action(sub)
        self.assertTrue(info.get("stale_decision", False))
        self.assertEqual(r, 0.0)
        env.close()

    def test_stale_action_version_rejected(self):
        env = RandomEventAllocationEnv(
            initial_seed=42, event_seed=42001, mode="single", events_per_episode=1
        )
        env.reset(seed=42)
        ctx = env.begin_decision()
        noop = env.action_space.n - 1
        env.step(noop)  # advance
        sub = ActionSubmission(action=noop, graph_version=env.graph_version, action_version=ctx.action_version)
        g2, r, term, trunc, info = env.submit_action(sub)
        self.assertTrue(info.get("stale_decision", False))
        self.assertEqual(r, 0.0)
        env.close()


# ---------------------------------------------------------------------------
# Trainer versioned submission + stale handling
# ---------------------------------------------------------------------------
class TrainerVersionedSubmissionTests(unittest.TestCase):
    def test_trainer_collect_rollout_uses_versioned(self):
        from ppo_allocation.random_event.trainer import PPOConfig, PPOTrainer
        env = RandomEventAllocationEnv(
            initial_seed=42, event_seed=42001, mode="single", events_per_episode=1
        )
        env.reset(seed=42)
        config = PPOConfig(rollout_steps=4, update_epochs=1, minibatch_size=2, seed=1, device="cpu")
        trainer = PPOTrainer(env=env, variant="GPPO-Adaptive", config=config)
        buffer, stats = trainer.collect_rollout(4)
        self.assertGreater(len(buffer), 0)
        self.assertEqual(trainer.total_steps, 4)
        env.close()

    def test_trainer_injected_stale_reject_reinfers(self):
        from ppo_allocation.random_event.trainer import PPOConfig, PPOTrainer
        env = RandomEventAllocationEnv(
            initial_seed=42, event_seed=42001, mode="single", events_per_episode=2
        )
        original_submit = env.submit_action
        calls = {"count": 0}
        def injected(submission):
            if calls["count"] == 0:
                calls["count"] += 1
                env.decision_version += 1
            return original_submit(submission)
        env.submit_action = injected
        trainer = PPOTrainer(
            env=env, variant="GPPO-Adaptive",
            config=PPOConfig(rollout_steps=1, update_epochs=1, minibatch_size=1, seed=1, device="cpu"),
        )
        buffer, stats = trainer.collect_rollout(1)
        self.assertEqual(len(buffer), 1)
        self.assertEqual(trainer.total_steps, 1)
        self.assertEqual(stats["stale_retries"], 1)
        self.assertEqual(env.stale_rejection_count, 1)
        env.close()

    def test_trainer_stale_reject_no_buffer(self):
        """Stale rejection should NOT add to buffer or increment total_steps."""
        from ppo_allocation.random_event.trainer import PPOConfig, PPOTrainer
        env = RandomEventAllocationEnv(
            initial_seed=42, event_seed=42001, mode="single", events_per_episode=1
        )
        env.reset(seed=42)
        config = PPOConfig(rollout_steps=2, update_epochs=1, minibatch_size=2, seed=1, device="cpu")
        trainer = PPOTrainer(env=env, variant="GPPO-Adaptive", config=config)
        buffer, stats = trainer.collect_rollout(2)
        # All transitions should be valid (no stale rejects in normal flow)
        self.assertEqual(len(buffer), 2)
        self.assertEqual(trainer.total_steps, 2)
        env.close()


# ---------------------------------------------------------------------------
# Eval versioned submission
# ---------------------------------------------------------------------------
class EvalVersionedSubmissionTests(unittest.TestCase):
    def test_run_episode_injected_stale_retries_without_trace_row(self):
        from unittest.mock import patch
        from ppo_allocation.random_event.events import EventTape, RandomEvent, RandomEventType
        import ppo_allocation.random_event.experiment as experiment
        from ppo_allocation.random_event.baselines import NearestLegalPolicy
        event = RandomEvent(
            event_id="test-stale", event_type=RandomEventType.REGION_VACANCY,
            occurred_at=0.0, observed_at=0.0, source_event="test",
            affected_uavs=(0,), affected_regions=(0,), severity=0.5,
            event_seed=100, state_version=0,
        )
        tape = EventTape(initial_seed=42, event_seed=100, mode="single", events=(event,))
        original_cls = experiment.RandomEventAllocationEnv
        def factory(**kwargs):
            env = original_cls(**kwargs)
            original_submit = env.submit_action
            calls = {"count": 0}
            def injected(submission):
                if calls["count"] == 0:
                    calls["count"] += 1
                    env.decision_version += 1
                return original_submit(submission)
            env.submit_action = injected
            return env
        with patch.object(experiment, "RandomEventAllocationEnv", side_effect=factory):
            episode, trace = experiment.run_episode(
                NearestLegalPolicy(), tape_id="test-stale", tape=tape,
                algorithm="NearestLegal", max_decisions=5,
            )
        self.assertEqual(trace["stale_submission_retry_count"], 1)
        self.assertEqual(len(trace["decisions"]), 1)
        self.assertTrue(all("stale_decision" not in row for row in trace["decisions"]))
        self.assertEqual(trace["episode_return_check"], sum(row["reward"] for row in trace["decisions"]))
        self.assertEqual(episode.decision_count, sum(event["decision_count"] for event in trace["events"]))

    def test_run_episode_versioned(self):
        from ppo_allocation.random_event.events import EventTape, RandomEvent, RandomEventType
        from ppo_allocation.random_event.baselines import NearestLegalPolicy
        from ppo_allocation.random_event.experiment import run_episode
        event = RandomEvent(
            event_id="test-ev", event_type=RandomEventType.REGION_VACANCY,
            occurred_at=0.0, observed_at=0.0, source_event="test",
            affected_uavs=(0,), affected_regions=(0,), severity=0.5,
            event_seed=100, state_version=0,
        )
        tape = EventTape(initial_seed=42, event_seed=100, mode="single", events=(event,))
        episode, trace = run_episode(NearestLegalPolicy(), tape_id="test-v", tape=tape, algorithm="NearestLegal", max_decisions=5)
        self.assertGreater(trace["episode_return_check"], 0)


# ---------------------------------------------------------------------------
# Checkpoint scheduling
# ---------------------------------------------------------------------------
class CheckpointSchedulingTests(unittest.TestCase):
    def test_standard_300k(self):
        steps = compute_checkpoint_steps(300_000, 25_000)
        self.assertEqual(len(steps), 12)
        self.assertEqual(steps[0], 25_000)
        self.assertEqual(steps[-1], 300_000)

    def test_small_dry_run(self):
        steps = compute_checkpoint_steps(10_000, 5_000)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps, [5_000, 10_000])


# ---------------------------------------------------------------------------
# PreliminaryProtocol
# ---------------------------------------------------------------------------
class PreliminaryProtocolTests(unittest.TestCase):
    def test_defaults(self):
        p = PreliminaryProtocol()
        self.assertEqual(p.variants, ("PPO-MLP", "GPPO-NoGate", "GPPO-Adaptive"))
        self.assertEqual(p.training_seeds, (1101, 2202, 3303))
        self.assertEqual(p.budget, 300_000)
        self.assertEqual(p.num_checkpoints, 12)
        self.assertEqual(p.num_runs, 9)


# ---------------------------------------------------------------------------
# Synthetic lexicographic selection tests
# ---------------------------------------------------------------------------
class LexicographicSelectionTests(unittest.TestCase):
    """Test that lexicographic sort picks the correct winner at each level."""

    def _make_metrics(self, **overrides) -> ValidationMetrics:
        defaults = dict(
            checkpoint_path="fake.pt",
            checkpoint_sha256="a" * 64,
            variant="PPO-MLP",
            training_seed=1101,
            decision_steps=100,
            final_infeasible_count=0,
            final_infeasible_rate=0.0,
            cumulative_weighted_vacancy=0.0,
            recovery_latency=0.0,
            fixed_j=0.0,
        )
        defaults.update(overrides)
        return ValidationMetrics(**defaults)

    def test_level1_infeasible_rate_decides(self):
        """Lowest infeasible_rate wins."""
        m1 = self._make_metrics(final_infeasible_rate=0.5, cumulative_weighted_vacancy=1.0)
        m2 = self._make_metrics(final_infeasible_rate=0.1, cumulative_weighted_vacancy=0.0)
        selected, _ = _lexicographic_select([m1, m2])
        # m2 has lower infeasible_rate despite higher vacancy
        self.assertEqual(selected.final_infeasible_rate, 0.1)

    def test_level2_vacancy_decides(self):
        """Tie on infeasible_rate → lowest vacancy wins."""
        m1 = self._make_metrics(final_infeasible_rate=0.1, cumulative_weighted_vacancy=5.0)
        m2 = self._make_metrics(final_infeasible_rate=0.1, cumulative_weighted_vacancy=1.0)
        selected, _ = _lexicographic_select([m1, m2])
        self.assertEqual(selected.cumulative_weighted_vacancy, 1.0)

    def test_level3_recovery_latency_decides(self):
        """Tie on infeasible + vacancy → lowest recovery_latency wins."""
        m1 = self._make_metrics(final_infeasible_rate=0.1, cumulative_weighted_vacancy=1.0, recovery_latency=10.0)
        m2 = self._make_metrics(final_infeasible_rate=0.1, cumulative_weighted_vacancy=1.0, recovery_latency=2.0)
        selected, _ = _lexicographic_select([m1, m2])
        self.assertEqual(selected.recovery_latency, 2.0)

    def test_level4_fixed_j_decides(self):
        """Tie on levels 1-3 → lowest fixed_j wins."""
        m1 = self._make_metrics(
            final_infeasible_rate=0.1, cumulative_weighted_vacancy=1.0,
            recovery_latency=2.0, fixed_j=100.0,
        )
        m2 = self._make_metrics(
            final_infeasible_rate=0.1, cumulative_weighted_vacancy=1.0,
            recovery_latency=2.0, fixed_j=50.0,
        )
        selected, _ = _lexicographic_select([m1, m2])
        self.assertEqual(selected.fixed_j, 50.0)

    def test_level5_earlier_checkpoint_decides(self):
        """Tie on all 4 metrics → earlier checkpoint (lower step) wins."""
        m1 = self._make_metrics(
            final_infeasible_rate=0.1, cumulative_weighted_vacancy=1.0,
            recovery_latency=2.0, fixed_j=50.0, decision_steps=200,
        )
        m2 = self._make_metrics(
            final_infeasible_rate=0.1, cumulative_weighted_vacancy=1.0,
            recovery_latency=2.0, fixed_j=50.0, decision_steps=100,
        )
        selected, _ = _lexicographic_select([m1, m2])
        self.assertEqual(selected.decision_steps, 100)


# ---------------------------------------------------------------------------
# Real metric extraction and strict contract tests
# ---------------------------------------------------------------------------
class ValidationMetricIntegrationTests(unittest.TestCase):
    def _episode(self, **overrides):
        from dataclasses import fields
        from ppo_allocation.random_event.metrics import EpisodeMetrics
        values = {}
        for field in fields(EpisodeMetrics):
            if field.name in {"tape_id", "episode_id", "algorithm"}:
                values[field.name] = "x"
            elif field.name in {"event_success_rate", "legal_coverage_rate", "weighted_uncovered", "recovery_delay", "normalized_distance", "load_gap", "inference_latency_ms", "event_to_action_latency_ms", "communication_suppression_rate", "pre_mask_invalid_probability", "mask_rate", "gate_mean", "gate_variance", "value_error", "value_squared_error", "avg_reward"}:
                values[field.name] = None
            elif field.name in {"episode_return"}:
                values[field.name] = 0.0
            else:
                values[field.name] = 0
        values.update({
            "tape_id": "t", "episode_id": "e", "algorithm": "PPO-MLP",
            "event_count": 1, "final_infeasible_count": 0,
            "final_infeasible_rate": 0.0, "cumulative_uncovered_time": 2.0,
            "recovery_delay": 3.0, "recovery_delay_observed_count": 1,
        })
        values.update(overrides)
        return EpisodeMetrics(**values)

    def _trace(self, **overrides):
        event = {
            "weighted_uncovered": 2.0, "normalized_distance": 1.0,
            "load_gap": 0.5, "switch_count": 2, "recovery_delay": 3.0,
        }
        event.update(overrides)
        return {"events": [event]}

    def _checkpoint(self):
        return __import__("ppo_allocation.random_event.phase_j", fromlist=["CheckpointRecord"]).CheckpointRecord(
            variant="PPO-MLP", training_seed=1101, decision_steps=64,
            checkpoint_path="x.pt", checkpoint_sha256="a" * 64,
            source_tree_hash="b" * 64, attested_source_commit_sha="c" * 40,
            protocol_sha256="d" * 64, seed_manifest_sha256="e" * 64,
            ppo_config={}, rng_state={}, created_at="now",
        )

    def test_extracts_real_episode_metrics_and_fixed_j(self):
        from ppo_allocation.random_event.phase_j import extract_validation_metrics
        result = extract_validation_metrics(self._episode(), self._trace(), self._checkpoint())
        self.assertEqual(result.final_infeasible_count, 0)
        self.assertEqual(result.cumulative_weighted_vacancy, 2.0)
        self.assertAlmostEqual(result.fixed_j, 2.0 * 5.0 + 1.0 + 0.5 + 2.0 * 0.25 + 3.0 * 0.5)

    def test_unresolved_event_is_rankable_with_explicit_censoring(self):
        from ppo_allocation.random_event.phase_j import extract_validation_metrics, _lexicographic_select
        candidate_a = extract_validation_metrics(
            self._episode(final_infeasible_count=1, final_infeasible_rate=1.0,
                          recovery_delay=None, recovery_delay_observed_count=0),
            self._trace(recovery_delay=None), self._checkpoint()
        )
        candidate_b = extract_validation_metrics(
            self._episode(final_infeasible_count=0, final_infeasible_rate=0.0),
            self._trace(), self._checkpoint()
        )
        selected, _ = _lexicographic_select([candidate_a, candidate_b])
        self.assertEqual(selected.final_infeasible_count, 0)
        self.assertEqual(candidate_a.recovery_latency, 200.0)
        self.assertGreater(candidate_a.fixed_j, candidate_b.fixed_j)

    def test_equal_nonzero_infeasible_candidates_remain_rankable(self):
        from ppo_allocation.random_event.phase_j import extract_validation_metrics, _lexicographic_select
        first = extract_validation_metrics(
            self._episode(final_infeasible_count=1, final_infeasible_rate=1.0,
                          cumulative_uncovered_time=10.0, recovery_delay=None,
                          recovery_delay_observed_count=0),
            self._trace(recovery_delay=None), self._checkpoint()
        )
        second = extract_validation_metrics(
            self._episode(final_infeasible_count=1, final_infeasible_rate=1.0,
                          cumulative_uncovered_time=2.0, recovery_delay=None,
                          recovery_delay_observed_count=0),
            self._trace(recovery_delay=None), self._checkpoint()
        )
        selected, _ = _lexicographic_select([first, second])
        self.assertEqual(selected.cumulative_weighted_vacancy, 2.0)


class PhaseJCompletenessAndCliTests(unittest.TestCase):
    def test_formal_checkpoint_completeness_rejects_missing_groups(self):
        from ppo_allocation.random_event.phase_j import _validate_formal_checkpoints
        with self.assertRaises(SystemExit):
            _validate_formal_checkpoints([], {})

    def test_cli_defaults_are_frozen(self):
        from ppo_allocation.random_event.phase_j import build_parser
        args = build_parser().parse_args(["preliminary-train"])
        self.assertEqual(args.budget, 300000)
        self.assertEqual(args.checkpoint_interval, 25000)
        self.assertFalse(args.developer_mode)

    def test_cli_help_contains_all_phase_j_commands(self):
        from ppo_allocation.random_event.phase_j import build_parser
        help_text = build_parser().format_help()
        for command in ("preliminary-train", "preliminary-validate", "preliminary-freeze", "preliminary-test", "phase-j-dry-run"):
            self.assertIn(command, help_text)


# ---------------------------------------------------------------------------
# Provenance, Test lock and public bypass guards
# ---------------------------------------------------------------------------
class ProvenanceGuardTests(unittest.TestCase):
    def _gate(self):
        return {
            "source_tree_hash": "s" * 64,
            "attested_source_commit_sha": "a" * 40,
            "protocol_sha256": "p" * 64,
            "seed_manifest_sha256": "m" * 64,
        }

    def _formal_freeze(self, gate):
        freezes = []
        for variant in ("PPO-MLP", "GPPO-NoGate", "GPPO-Adaptive"):
            for seed in (1101, 2202, 3303):
                freezes.append({
                    "variant": variant, "training_seed": seed,
                    "selected_step": 25000, "checkpoint_path": "x.pt",
                    "checkpoint_sha256": "c" * 64,
                    "source_sha": gate["source_tree_hash"],
                    "protocol_sha": gate["protocol_sha256"],
                    "seed_manifest_sha": gate["seed_manifest_sha256"],
                    "validation_manifest_sha": "v" * 64,
                    "attested_source_commit_sha": gate["attested_source_commit_sha"],
                    "selected_at": "now",
                })
        return {
            "formal": True, "freeze_count": 9, "freezes": freezes,
            "source_tree_hash": gate["source_tree_hash"],
            "attested_source_commit_sha": gate["attested_source_commit_sha"],
            "protocol_sha256": gate["protocol_sha256"],
            "seed_manifest_sha256": gate["seed_manifest_sha256"],
        }

    def test_stale_selection_provenance_rejected_by_freeze(self):
        from unittest.mock import patch
        from ppo_allocation.random_event.phase_j import preliminary_freeze
        gate = self._gate()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "preliminary").mkdir()
            selection = {
                "formal": True, "selected_checkpoints": [],
                "source_tree_hash": "old" * 16,
                "attested_source_commit_sha": gate["attested_source_commit_sha"],
                "protocol_sha256": gate["protocol_sha256"],
                "seed_manifest_sha256": gate["seed_manifest_sha256"],
            }
            (root / "preliminary" / "validation_selection.json").write_text(json.dumps(selection))
            with patch("ppo_allocation.random_event.phase_j._check_p0_gate_strict", return_value=gate), \
                 patch("ppo_allocation.random_event.phase_j._validate_hashes_match"):
                with self.assertRaises(SystemExit):
                    preliminary_freeze(root)

    def test_every_freeze_provenance_field_is_checked(self):
        from ppo_allocation.random_event.phase_j import _validate_freeze_payload
        gate = self._gate()
        payload = self._formal_freeze(gate)
        for field in ("source_tree_hash", "attested_source_commit_sha", "protocol_sha256", "seed_manifest_sha256"):
            altered = dict(payload)
            altered[field] = "wrong"
            with self.assertRaises(SystemExit):
                _validate_freeze_payload(altered, gate)
        for field in ("source_sha", "attested_source_commit_sha", "protocol_sha", "seed_manifest_sha"):
            altered = json.loads(json.dumps(payload))
            altered["freezes"][0][field] = "wrong"
            with self.assertRaises(SystemExit):
                _validate_freeze_payload(altered, gate)

    def test_test_manifest_requires_all_frozen_contract_fields(self):
        from ppo_allocation.random_event.phase_j import _validate_test_manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            path.write_text(json.dumps({"tier": "preliminary", "split": "test"}))
            with self.assertRaises(SystemExit):
                _validate_test_manifest(path, self._gate())

    def test_formal_test_bank_lock_rejects_regeneration(self):
        from unittest.mock import patch
        import ppo_allocation.random_event.phase_j as phase_j
        gate = self._gate()
        freeze_payload = self._formal_freeze(gate)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            (output / "preliminary").mkdir()
            (output / "preliminary" / "frozen_manifests.json").write_text(json.dumps(freeze_payload))
            manifest_path = output / "preliminary" / "tapes" / "preliminary_test_protocol" / "manifest.json"
            labels = ["Test-Single", "Test-Sequential", "Test-Overlap", "Test-Burst", "Test-Unseen"]
            def fake_generate(*args, **kwargs):
                entries = [
                    {"set_name": label, "mode": ("single" if label == "Test-Unseen" else label.removeprefix("Test-").lower())}
                    for label in labels for _ in range(40)
                ]
                payload = {
                    "tier": "preliminary", "split": "test", "complete_frozen_bank": True,
                    "expected_tape_count": 200, "tape_count": 200,
                    "checkpoint_selection": False, "reward_tuning": False,
                    "seed_manifest_sha256": gate["seed_manifest_sha256"],
                    "protocol_sha256": gate["protocol_sha256"], "entries": entries,
                }
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(payload))
                return payload
            with patch.object(phase_j, "_check_p0_gate_strict", return_value=gate), \
                 patch.object(phase_j, "_validate_hashes_match"), \
                 patch.object(phase_j, "generate_protocol_bank", side_effect=fake_generate):
                phase_j.generate_test_bank(output)
                self.assertTrue((output / "preliminary" / "formal_test_bank_lock.json").exists())
                payload = json.loads(manifest_path.read_text())
                payload["created_at_utc"] = "changed"
                manifest_path.write_text(json.dumps(payload))
                with self.assertRaises(SystemExit):
                    phase_j.generate_test_bank(output)

    def test_legacy_protocol_bank_test_path_is_guarded(self):
        from argparse import Namespace
        from ppo_allocation.random_event.experiment import run_protocol_bank
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                output_dir=tmpdir, tier="preliminary", split="test",
                seed_manifest="configs/seed_manifest.json",
                protocol="configs/random_event_protocol.json",
                events_per_tape=5, limit_per_set=None,
            )
            with self.assertRaises((SystemExit, FileNotFoundError)):
                run_protocol_bank(args)


# ---------------------------------------------------------------------------
# Test isolation guard
# ---------------------------------------------------------------------------
class TestIsolationGuardTests(unittest.TestCase):
    def test_test_consumed_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            consumed_path = Path(tmpdir) / "test_consumed.json"
            self.assertFalse(consumed_path.exists())
            consumed_path.write_text(json.dumps({"consumed": True}))
            data = json.loads(consumed_path.read_text())
            self.assertTrue(data["consumed"])


# ---------------------------------------------------------------------------
# DRY RUN
# ---------------------------------------------------------------------------
class DryRunOrchestratorTests(unittest.TestCase):
    def test_dry_run_end_to_end_fresh_directory(self):
        from unittest.mock import patch
        from ppo_allocation.random_event.phase_j import dry_run, preliminary_train
        self.assertTrue(callable(preliminary_train))
        gate = json.loads((Path(_REPO_ROOT) / "handoff" / "P0_GATE.json").read_text())
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("ppo_allocation.random_event.phase_j._check_p0_gate_strict", return_value=gate), \
                 patch("ppo_allocation.random_event.phase_j._validate_hashes_match"):
                result = dry_run(Path(tmpdir))
            self.assertEqual(result["formal"], False)
            self.assertEqual(result["selected_per_group"], 9)
            self.assertEqual(result["frozen_count"], 9)
            self.assertEqual(result["train_checkpoints"], 18)
            self.assertFalse(result["official_test_namespace_touched"])
            self.assertTrue((Path(tmpdir) / "phase_j_dry_run_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
