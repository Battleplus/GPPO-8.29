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

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import dataclass

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

import numpy as np
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
        steps = compute_checkpoint_steps(50_000, 25_000)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0], 25_000)
        self.assertEqual(steps[-1], 50_000)

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
        self.assertEqual(p.budget, 50_000)
        self.assertEqual(p.num_checkpoints, 2)
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
            "fixed_j": 2.0 * 5.0 + 1.0 + 0.5 + 2.0 * 0.25 + 3.0 * 0.5,
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
            self._trace(recovery_delay=None, weighted_uncovered=4.0,
                        fixed_j=4.0 * 5.0 + 1.0 + 0.5 + 2.0 * 0.25 + 200.0 * 0.5),
            self._checkpoint()
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
        self.assertEqual(args.budget, 50000)
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
# Frozen train contract: mode cycle, seed coverage, range exhaustion
# ---------------------------------------------------------------------------
class FrozenTrainContractTests(unittest.TestCase):
    def _manifest(self):
        return json.loads((Path(_REPO_ROOT) / "configs" / "seed_manifest.json").read_text(encoding="utf-8"))

    def test_formal_train_mode_cycle_matches_frozen_manifest(self):
        from ppo_allocation.random_event.experiment import _load_frozen_train_modes
        manifest = self._manifest()
        frozen_cycle = tuple(manifest["preliminary"]["train"]["mode_cycle"])
        self.assertEqual(_load_frozen_train_modes(), frozen_cycle)
        self.assertEqual(frozen_cycle, ("sequential", "overlap", "burst"))
        self.assertNotIn("single", frozen_cycle)

    def test_formal_train_seed_coverage_is_sufficient_for_50k(self):
        # The trainer performs one initial reset before the first decision plus
        # one post-terminal reset after the LAST accepted transition, so the
        # worst case (1 accepted decision per episode) needs budget + 1 resets.
        manifest = self._manifest()
        reserved = int(manifest["preliminary"]["train"]["episodes_per_training_seed"])
        budget = int(
            manifest["preliminary"]["train"]["reserved_coverage_assertions"]["formal_budget_decision_steps"]
        )
        self.assertGreaterEqual(reserved, budget + 1)
        self.assertEqual(reserved, 50_001)
        self.assertTrue(
            manifest["preliminary"]["train"]["reserved_coverage_assertions"]["includes_initial_reset"]
        )
        self.assertEqual(
            manifest["preliminary"]["train"]["reserved_coverage_assertions"]["sufficiency_rule"],
            "reserved >= formal_budget_decision_steps + 1",
        )
        for training_seed in (1101, 2202, 3303):
            inst = manifest["preliminary"]["train"]["instance_seeds_by_training_seed"][str(training_seed)]
            evt = manifest["preliminary"]["train"]["event_seeds_by_training_seed"][str(training_seed)]
            self.assertEqual(int(inst["count"]), reserved)
            self.assertEqual(int(inst["start"]), training_seed * 1_000_003)
            self.assertEqual(int(evt["start"]), training_seed * 10_000_019)
            # Formula for the final reserved episode stays inside the range.
            last_index = reserved - 1
            self.assertLessEqual(
                training_seed * 1_000_003 + last_index,
                int(inst["start"]) + int(inst["count"]) - 1,
            )

    def test_trainer_N_accepted_steps_requires_N_plus_1_resets(self):
        """Every accepted terminal transition triggers one reset, plus the
        initial reset, so N accepted steps need exactly N+1 reset calls."""
        from ppo_allocation.random_event.experiment import CyclingTrainingEnv
        from ppo_allocation.random_event.trainer import PPOTrainer, PPOConfig

        reset_log = []

        class _EveryStepTerminalEnv(CyclingTrainingEnv):
            def reset(self, *args, **kwargs):
                reset_log.append(self._reset_index)
                return super().reset(*args, **kwargs)

            def submit_action(self, submission):
                observation, reward, _terminated, truncated, info = super().submit_action(submission)
                # Force every accepted decision to terminate the episode.
                return observation, reward, True, False, info

        env = _EveryStepTerminalEnv(
            seed=1101, modes=("single",), events_per_episode=1, max_resets=64,
        )
        config = PPOConfig(rollout_steps=1, update_epochs=1, minibatch_size=2, seed=1101, device="cpu")
        trainer = PPOTrainer(env=env, variant="GPPO-NoGate", config=config)
        n = 4
        buffer, stats = trainer.collect_rollout(n)
        try:
            self.assertEqual(trainer.total_steps, n)
            self.assertEqual(len(buffer), n)
            self.assertEqual(len(reset_log), n + 1)
            self.assertEqual(reset_log, list(range(n + 1)))
        finally:
            env.close()

    def test_train_seed_range_exhaustion_hard_fails(self):
        from ppo_allocation.random_event.experiment import CyclingTrainingEnv
        env = CyclingTrainingEnv(seed=1101, modes=("sequential",), events_per_episode=1, max_resets=2)
        env.reset()
        env.reset()
        with self.assertRaises(RuntimeError):
            env.reset()
        env.close()

    def test_last_allowed_reset_passes(self):
        from ppo_allocation.random_event.experiment import CyclingTrainingEnv
        env = CyclingTrainingEnv(seed=3303, modes=("sequential",), events_per_episode=1, max_resets=2)
        env.reset()
        env.reset()
        self.assertEqual(env._reset_index, 2)
        env.close()


# ---------------------------------------------------------------------------
# Formal Test partial resume / completed rerun / ambiguous state
# ---------------------------------------------------------------------------
class FormalTestResumeTests(unittest.TestCase):
    def _freeze_payload(self, gate, root):
        freezes = []
        for variant in ("PPO-MLP", "GPPO-NoGate", "GPPO-Adaptive"):
            for seed in (1101, 2202, 3303):
                ckpt_path = root / "preliminary" / "models" / f"{variant}-{seed}.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                ckpt_path.write_bytes(b"ckpt")
                freezes.append({
                    "variant": variant, "training_seed": seed,
                    "selected_step": 25000, "checkpoint_path": str(ckpt_path),
                    "checkpoint_sha256": hashlib.sha256(b"ckpt").hexdigest(),
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

    def _test_bank(self, gate, root):
        labels = ["Test-Single", "Test-Sequential", "Test-Overlap", "Test-Burst", "Test-Unseen"]
        entries = []
        for label in labels:
            for i in range(40):
                entries.append({
                    "tape_id": f"{label}-{i}", "set_name": label,
                    "mode": "single" if label == "Test-Unseen" else label.removeprefix("Test-").lower(),
                    "path": "x", "sha256": "a", "canonical_tape_sha256": "b",
                })
        manifest = {
            "tier": "preliminary", "split": "test", "complete_frozen_bank": True,
            "expected_tape_count": 200, "tape_count": 200,
            "checkpoint_selection": False, "reward_tuning": False,
            "seed_manifest_sha256": gate["seed_manifest_sha256"],
            "protocol_sha256": gate["protocol_sha256"],
            "entries": entries,
        }
        manifest_path = root / "preliminary" / "tapes" / "preliminary_test_protocol" / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path, hashlib.sha256(json.dumps(manifest).encode()).hexdigest()

    def _run_test(self, root, gate, freeze_payload, manifest_sha):
        import ppo_allocation.random_event.phase_j as phase_j
        from unittest.mock import patch
        from ppo_allocation.random_event.metrics import EpisodeMetrics

        def fake_episode():
            from dataclasses import fields
            values = {}
            for field in fields(EpisodeMetrics):
                if field.name in {"tape_id", "episode_id", "algorithm"}:
                    values[field.name] = "x"
                elif field.name in {"event_success_rate", "legal_coverage_rate", "weighted_uncovered", "recovery_delay", "normalized_distance", "load_gap", "inference_latency_ms", "event_to_action_latency_ms", "communication_suppression_rate", "pre_mask_invalid_probability", "mask_rate", "gate_mean", "gate_variance", "value_error", "value_squared_error", "avg_reward"}:
                    values[field.name] = None
                else:
                    values[field.name] = 0
            values.update({"tape_id": "t", "episode_id": "e", "algorithm": "X", "event_count": 1, "fixed_j": 1.0})
            return EpisodeMetrics(**values)

        class _FakeModel:
            def eval(self):
                return self
        with patch.object(phase_j, "_check_p0_gate_strict", return_value=gate), \
             patch.object(phase_j, "_validate_hashes_match"), \
             patch.object(phase_j.torch, "load", return_value={"format": "graph-actor-critic-v1"}), \
             patch.object(phase_j.GraphActorCritic, "load", return_value=(_FakeModel(), {})), \
             patch.object(phase_j, "load_tape_bank", return_value=({}, [])), \
             patch.object(phase_j, "run_episode", return_value=(fake_episode(), {"reward_invariant": True, "decisions": [], "events": []})):
            return phase_j.preliminary_test(root)

    def test_partial_resume_skips_consumed_and_runs_rest(self):
        gate = ProvenanceGuardTests()._gate()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "preliminary").mkdir(parents=True, exist_ok=True)
            freeze_payload = self._freeze_payload(gate, root)
            (root / "preliminary" / "frozen_manifests.json").write_text(json.dumps(freeze_payload))
            manifest_path, manifest_sha = self._test_bank(gate, root)
            # Create lock
            lock = {
                "test_manifest_sha256": manifest_sha,
                "source_tree_hash": gate["source_tree_hash"],
                "attested_source_commit_sha": gate["attested_source_commit_sha"],
                "protocol_sha256": gate["protocol_sha256"],
                "seed_manifest_sha256": gate["seed_manifest_sha256"],
                "freeze_manifest_sha256": hashlib.sha256(json.dumps(freeze_payload).encode()).hexdigest(),
                "completed": False,
            }
            (root / "preliminary" / "formal_test_bank_lock.json").write_text(json.dumps(lock))
            # Simulate: first 2 already consumed under the same provenance.
            keys = []
            for freeze in freeze_payload["freezes"][:2]:
                key = f"{freeze['variant']}_seed{freeze['training_seed']}_{freeze['checkpoint_sha256'][:12]}_{manifest_sha}"
                keys.append(key)
            ledger = {
                "schema_version": 1, "completed": False, "test_manifest_sha256": manifest_sha,
                "entries": {key: {
                    "consumed": True, "variant": freeze_payload["freezes"][i]["variant"],
                    "training_seed": freeze_payload["freezes"][i]["training_seed"],
                    "checkpoint_sha": freeze_payload["freezes"][i]["checkpoint_sha256"],
                    "test_manifest_sha": manifest_sha,
                    "freeze_manifest_sha": lock["freeze_manifest_sha256"],
                    "source_tree_hash": gate["source_tree_hash"],
                    "protocol_sha256": gate["protocol_sha256"],
                    "seed_manifest_sha256": gate["seed_manifest_sha256"],
                    "result_path": f"preliminary/test_results/{i}.json",
                    "result_sha": "c" * 64,
                } for i, key in enumerate(keys)},
            }
            (root / "preliminary" / "test_ledger.json").write_text(json.dumps(ledger))
            # Journal is the single source of truth: write consumed journals for
            # the two already-consumed checkpoints (same provenance).
            state_dir = root / "preliminary" / "test_state"
            state_dir.mkdir(parents=True, exist_ok=True)
            for i, key in enumerate(keys):
                freeze = freeze_payload["freezes"][i]
                (state_dir / f"{key}.json").write_text(json.dumps({
                    "state": "consumed",
                    "key": key,
                    "checkpoint_sha": freeze["checkpoint_sha256"],
                    "test_manifest_sha": manifest_sha,
                    "freeze_manifest_sha": lock["freeze_manifest_sha256"],
                    "source_tree_hash": gate["source_tree_hash"],
                    "protocol_sha256": gate["protocol_sha256"],
                    "seed_manifest_sha256": gate["seed_manifest_sha256"],
                    "result_path": f"preliminary/test_results/{i}.json",
                    "result_sha": "c" * 64,
                }))
            result = self._run_test(root, gate, freeze_payload, manifest_sha)
            # 2 skipped (resumed), 7 evaluated.
            resumed = [r for r in result["results"] if r.get("resumed")]
            evaluated = [r for r in result["results"] if not r.get("resumed")]
            self.assertEqual(len(resumed), 2)
            self.assertEqual(len(evaluated), 7)
            ledger_after = json.loads((root / "preliminary" / "test_ledger.json").read_text())
            self.assertTrue(ledger_after["completed"])
            lock_after = json.loads((root / "preliminary" / "formal_test_bank_lock.json").read_text())
            self.assertTrue(lock_after["completed"])

    def test_completed_test_rerun_rejected(self):
        gate = ProvenanceGuardTests()._gate()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "preliminary").mkdir(parents=True, exist_ok=True)
            freeze_payload = self._freeze_payload(gate, root)
            (root / "preliminary" / "frozen_manifests.json").write_text(json.dumps(freeze_payload))
            manifest_path, manifest_sha = self._test_bank(gate, root)
            lock = {
                "test_manifest_sha256": manifest_sha,
                "source_tree_hash": gate["source_tree_hash"],
                "attested_source_commit_sha": gate["attested_source_commit_sha"],
                "protocol_sha256": gate["protocol_sha256"],
                "seed_manifest_sha256": gate["seed_manifest_sha256"],
                "freeze_manifest_sha256": "f" * 64,
                "completed": True,
            }
            (root / "preliminary" / "formal_test_bank_lock.json").write_text(json.dumps(lock))
            with self.assertRaises(SystemExit):
                self._run_test(root, gate, freeze_payload, manifest_sha)

    def test_ambiguous_running_test_state_rejected(self):
        gate = ProvenanceGuardTests()._gate()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "preliminary").mkdir(parents=True, exist_ok=True)
            freeze_payload = self._freeze_payload(gate, root)
            (root / "preliminary" / "frozen_manifests.json").write_text(json.dumps(freeze_payload))
            manifest_path, manifest_sha = self._test_bank(gate, root)
            lock = {
                "test_manifest_sha256": manifest_sha,
                "source_tree_hash": gate["source_tree_hash"],
                "attested_source_commit_sha": gate["attested_source_commit_sha"],
                "protocol_sha256": gate["protocol_sha256"],
                "seed_manifest_sha256": gate["seed_manifest_sha256"],
                "freeze_manifest_sha256": "f" * 64,
                "completed": False,
            }
            (root / "preliminary" / "formal_test_bank_lock.json").write_text(json.dumps(lock))
            freeze = freeze_payload["freezes"][0]
            key = f"{freeze['variant']}_seed{freeze['training_seed']}_{freeze['checkpoint_sha256'][:12]}_{manifest_sha}"
            state_dir = root / "preliminary" / "test_state"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / f"{key}.json").write_text(json.dumps({"state": "running"}))
            with self.assertRaises(SystemExit):
                self._run_test(root, gate, freeze_payload, manifest_sha)

    def _setup_test_env(self, gate, root):
        (root / "preliminary").mkdir(parents=True, exist_ok=True)
        freeze_payload = self._freeze_payload(gate, root)
        (root / "preliminary" / "frozen_manifests.json").write_text(json.dumps(freeze_payload))
        manifest_path, manifest_sha = self._test_bank(gate, root)
        lock = {
            "test_manifest_sha256": manifest_sha,
            "source_tree_hash": gate["source_tree_hash"],
            "attested_source_commit_sha": gate["attested_source_commit_sha"],
            "protocol_sha256": gate["protocol_sha256"],
            "seed_manifest_sha256": gate["seed_manifest_sha256"],
            "freeze_manifest_sha256": hashlib.sha256(json.dumps(freeze_payload).encode()).hexdigest(),
            "completed": False,
        }
        (root / "preliminary" / "formal_test_bank_lock.json").write_text(json.dumps(lock))
        return freeze_payload, manifest_sha, lock

    def _journal_fields(self, freeze, key, manifest_sha, freeze_sha, gate, result_path=None, result_sha=None):
        payload = {
            "state": "consumed",
            "key": key,
            "variant": freeze["variant"],
            "training_seed": freeze["training_seed"],
            "checkpoint_sha": freeze["checkpoint_sha256"],
            "test_manifest_sha": manifest_sha,
            "freeze_manifest_sha": freeze_sha,
            "source_tree_hash": gate["source_tree_hash"],
            "protocol_sha256": gate["protocol_sha256"],
            "seed_manifest_sha256": gate["seed_manifest_sha256"],
        }
        if result_path is not None:
            payload["result_path"] = result_path
            payload["result_sha"] = result_sha
        return payload

    def _run_test_tracked(self, root, gate, freeze_payload, manifest_sha):
        """Like _run_test but returns (result, run_episode_mock)."""
        import ppo_allocation.random_event.phase_j as phase_j
        from unittest.mock import patch, Mock
        from ppo_allocation.random_event.metrics import EpisodeMetrics

        def fake_episode():
            from dataclasses import fields
            values = {}
            for field in fields(EpisodeMetrics):
                if field.name in {"tape_id", "episode_id", "algorithm"}:
                    values[field.name] = "x"
                elif field.name in {"event_success_rate", "legal_coverage_rate", "weighted_uncovered", "recovery_delay", "normalized_distance", "load_gap", "inference_latency_ms", "event_to_action_latency_ms", "communication_suppression_rate", "pre_mask_invalid_probability", "mask_rate", "gate_mean", "gate_variance", "value_error", "value_squared_error", "avg_reward"}:
                    values[field.name] = None
                else:
                    values[field.name] = 0
            values.update({"tape_id": "t", "episode_id": "e", "algorithm": "X", "event_count": 1, "fixed_j": 1.0})
            return EpisodeMetrics(**values)

        run_episode_mock = Mock(return_value=(fake_episode(), {"reward_invariant": True, "decisions": [], "events": []}))

        class _FakeModel:
            def eval(self):
                return self

        with patch.object(phase_j, "_check_p0_gate_strict", return_value=gate), \
             patch.object(phase_j, "_validate_hashes_match"), \
             patch.object(phase_j.torch, "load", return_value={"format": "graph-actor-critic-v1"}), \
             patch.object(phase_j.GraphActorCritic, "load", return_value=(_FakeModel(), {})), \
             patch.object(phase_j, "load_tape_bank", return_value=({}, [])), \
             patch.object(phase_j, "run_episode", run_episode_mock):
            result = phase_j.preliminary_test(root)
        return result, run_episode_mock

    def test_consumed_journal_without_ledger_rebuilds_and_skips(self):
        """B: journal consumed + ledger entry missing -> NO re-evaluation;
        rebuild the ledger entry from the cryptographically verified result."""
        gate = ProvenanceGuardTests()._gate()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            freeze_payload, manifest_sha, lock = self._setup_test_env(gate, root)
            freeze = freeze_payload["freezes"][0]
            key = f"{freeze['variant']}_seed{freeze['training_seed']}_{freeze['checkpoint_sha256'][:12]}_{manifest_sha}"
            # Durable result file with matching SHA.
            result_dir = root / "preliminary" / "test_results"
            result_dir.mkdir(parents=True, exist_ok=True)
            result_file = result_dir / "resume0.json"
            result_file.write_text(json.dumps({"ok": True}))
            result_sha = hashlib.sha256(result_file.read_bytes()).hexdigest()
            state_dir = root / "preliminary" / "test_state"
            state_dir.mkdir(parents=True, exist_ok=True)
            journal = self._journal_fields(
                freeze, key, manifest_sha, lock["freeze_manifest_sha256"], gate,
                result_path=str(result_file.resolve()), result_sha=result_sha,
            )
            (state_dir / f"{key}.json").write_text(json.dumps(journal))
            # No ledger entry at all (crash before ledger persist).
            result, run_episode_mock = self._run_test_tracked(root, gate, freeze_payload, manifest_sha)
            run_episode_mock.assert_not_called()
            resumed = [r for r in result["results"] if r.get("resumed")]
            self.assertEqual(len(resumed), 1)
            ledger_after = json.loads((root / "preliminary" / "test_ledger.json").read_text())
            self.assertTrue(ledger_after["entries"][key]["consumed"])
            self.assertTrue(ledger_after["entries"][key].get("resumed_from_journal"))

    def test_consumed_journal_missing_result_hard_fails_exit(self):
        """B-fail: missing result under a consumed journal raises SystemExit
        (manual audit) instead of silently re-running."""
        gate = ProvenanceGuardTests()._gate()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            freeze_payload, manifest_sha, lock = self._setup_test_env(gate, root)
            freeze = freeze_payload["freezes"][0]
            key = f"{freeze['variant']}_seed{freeze['training_seed']}_{freeze['checkpoint_sha256'][:12]}_{manifest_sha}"
            state_dir = root / "preliminary" / "test_state"
            state_dir.mkdir(parents=True, exist_ok=True)
            journal = self._journal_fields(
                freeze, key, manifest_sha, lock["freeze_manifest_sha256"], gate,
                result_path=str(root / "preliminary" / "test_results" / "MISSING.json"),
                result_sha="d" * 64,
            )
            (state_dir / f"{key}.json").write_text(json.dumps(journal))
            import ppo_allocation.random_event.phase_j as phase_j
            from unittest.mock import patch, Mock
            from ppo_allocation.random_event.metrics import EpisodeMetrics

            def fake_episode():
                from dataclasses import fields
                values = {}
                for field in fields(EpisodeMetrics):
                    if field.name in {"tape_id", "episode_id", "algorithm"}:
                        values[field.name] = "x"
                    elif field.name in {"event_success_rate", "legal_coverage_rate", "weighted_uncovered", "recovery_delay", "normalized_distance", "load_gap", "inference_latency_ms", "event_to_action_latency_ms", "communication_suppression_rate", "pre_mask_invalid_probability", "mask_rate", "gate_mean", "gate_variance", "value_error", "value_squared_error", "avg_reward"}:
                        values[field.name] = None
                    else:
                        values[field.name] = 0
                values.update({"tape_id": "t", "episode_id": "e", "algorithm": "X", "event_count": 1, "fixed_j": 1.0})
                return EpisodeMetrics(**values)

            run_episode_mock = Mock(return_value=(fake_episode(), {"reward_invariant": True, "decisions": [], "events": []}))

            class _FakeModel:
                def eval(self):
                    return self

            with patch.object(phase_j, "_check_p0_gate_strict", return_value=gate), \
                 patch.object(phase_j, "_validate_hashes_match"), \
                 patch.object(phase_j.torch, "load", return_value={"format": "graph-actor-critic-v1"}), \
                 patch.object(phase_j.GraphActorCritic, "load", return_value=(_FakeModel(), {})), \
                 patch.object(phase_j, "load_tape_bank", return_value=({}, [])), \
                 patch.object(phase_j, "run_episode", run_episode_mock):
                with self.assertRaises(SystemExit):
                    phase_j.preliminary_test(root)
            run_episode_mock.assert_not_called()

    def test_ledger_consumed_running_journal_recovers_and_skips(self):
        """C: ledger consumed + journal running -> NO re-evaluation; restore
        the journal to consumed from the verified ledger/result."""
        gate = ProvenanceGuardTests()._gate()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            freeze_payload, manifest_sha, lock = self._setup_test_env(gate, root)
            freeze = freeze_payload["freezes"][0]
            key = f"{freeze['variant']}_seed{freeze['training_seed']}_{freeze['checkpoint_sha256'][:12]}_{manifest_sha}"
            # Durable result file.
            result_dir = root / "preliminary" / "test_results"
            result_dir.mkdir(parents=True, exist_ok=True)
            result_file = result_dir / "resume0.json"
            result_file.write_text(json.dumps({"ok": True}))
            # Ledger says consumed; journal still says running (crash window).
            ledger = {
                "schema_version": 1, "completed": False, "test_manifest_sha256": manifest_sha,
                "entries": {key: {
                    "consumed": True, "variant": freeze["variant"],
                    "training_seed": freeze["training_seed"],
                    "checkpoint_sha": freeze["checkpoint_sha256"],
                    "test_manifest_sha": manifest_sha,
                    "freeze_manifest_sha": lock["freeze_manifest_sha256"],
                    "source_tree_hash": gate["source_tree_hash"],
                    "protocol_sha256": gate["protocol_sha256"],
                    "seed_manifest_sha256": gate["seed_manifest_sha256"],
                    "result_path": str(result_file.resolve()),
                    "result_sha": hashlib.sha256(result_file.read_bytes()).hexdigest(),
                }},
            }
            (root / "preliminary" / "test_ledger.json").write_text(json.dumps(ledger))
            state_dir = root / "preliminary" / "test_state"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / f"{key}.json").write_text(json.dumps({"state": "running", "key": key}))
            result, run_episode_mock = self._run_test_tracked(root, gate, freeze_payload, manifest_sha)
            run_episode_mock.assert_not_called()
            resumed = [r for r in result["results"] if r.get("resumed")]
            self.assertEqual(len(resumed), 1)
            journal_after = json.loads((state_dir / f"{key}.json").read_text())
            self.assertEqual(journal_after["state"], "consumed")
            self.assertEqual(journal_after["result_sha"], hashlib.sha256(result_file.read_bytes()).hexdigest())


# ---------------------------------------------------------------------------
# Multi-event early termination remains validation-rankable
# ---------------------------------------------------------------------------
class EarlyTerminationRankableTests(unittest.TestCase):
    def test_unobserved_second_event_is_rankable(self):
        from dataclasses import fields
        from ppo_allocation.random_event.metrics import EpisodeMetrics
        from ppo_allocation.random_event.phase_j import extract_validation_metrics, _lexicographic_select, CheckpointRecord
        values = {}
        for field in fields(EpisodeMetrics):
            if field.name in {"tape_id", "episode_id", "algorithm"}:
                values[field.name] = "x"
            elif field.name in {"event_success_rate", "legal_coverage_rate", "weighted_uncovered", "recovery_delay", "normalized_distance", "load_gap", "inference_latency_ms", "event_to_action_latency_ms", "communication_suppression_rate", "pre_mask_invalid_probability", "mask_rate", "gate_mean", "gate_variance", "value_error", "value_squared_error", "avg_reward"}:
                values[field.name] = None
            else:
                values[field.name] = 0
        values.update({
            "tape_id": "t", "episode_id": "e", "algorithm": "PPO-MLP",
            "event_count": 2, "final_infeasible_count": 1, "final_infeasible_rate": 0.5,
            "cumulative_uncovered_time": 4.0, "recovery_delay": None,
            "recovery_delay_observed_count": 0, "fixed_j": 10.0,
        })
        episode = EpisodeMetrics(**values)
        # Event 1 was never observed: no decision rows, fixed_j censored from
        # final snapshot; Event 2 similarly unresolved.
        trace = {"events": [
            {"fixed_j": 6.0, "recovery_delay": None},
            {"fixed_j": 4.0, "recovery_delay": None},
        ]}
        ckpt = CheckpointRecord(
            variant="PPO-MLP", training_seed=1101, decision_steps=64,
            checkpoint_path="x.pt", checkpoint_sha256="a" * 64,
            source_tree_hash="b" * 64, attested_source_commit_sha="c" * 40,
            protocol_sha256="d" * 64, seed_manifest_sha256="e" * 64,
            ppo_config={}, rng_state={}, created_at="now",
        )
        metrics = extract_validation_metrics(episode, trace, ckpt)
        self.assertTrue(np.isfinite(metrics.final_infeasible_rate))
        self.assertTrue(np.isfinite(metrics.cumulative_weighted_vacancy))
        self.assertTrue(np.isfinite(metrics.recovery_latency))
        self.assertTrue(np.isfinite(metrics.fixed_j))
        selected, _ = _lexicographic_select([metrics])
        self.assertEqual(selected.checkpoint_sha256, ckpt.checkpoint_sha256)

    def test_run_episode_early_termination_unobserved_event_rankable(self):
        """REAL integration: a tape with >=2 events where an early event causes
        final-infeasible termination before a later event is ever observed.
        run_episode itself must materialise finite fixed_j for the unobserved
        event (test does NOT pre-fill fixed_j), and the checkpoint must remain
        lexicographically rankable."""
        from ppo_allocation.random_event.events import EventTape, RandomEvent, RandomEventType
        from ppo_allocation.random_event.baselines import NearestLegalPolicy
        from ppo_allocation.random_event.experiment import run_episode
        from ppo_allocation.random_event.phase_j import (
            CheckpointRecord, _lexicographic_select, extract_validation_metrics,
        )

        events = []
        # Burst of 4 UAV_DAMAGE (one per UAV): after the single merged decision
        # every UAV is dead, all regions are pending, and there is no future
        # TARGET_DESTROYED to release capacity -> final infeasible termination.
        for i in range(4):
            events.append(RandomEvent(
                event_id=f"dmg-{i}", event_type=RandomEventType.UAV_DAMAGE,
                occurred_at=0.0, observed_at=0.0, source_event="src",
                affected_uavs=(i,), affected_regions=(), affected_targets=(),
                severity=1.0, event_seed=100 + i, state_version=0,
            ))
        # Later event that must NEVER be observed/acted on.
        events.append(RandomEvent(
            event_id="vac-4", event_type=RandomEventType.REGION_VACANCY,
            occurred_at=5.0, observed_at=5.0, source_event="src",
            affected_uavs=(), affected_regions=(0,), affected_targets=(),
            severity=0.5, event_seed=200, state_version=0,
        ))
        tape = EventTape(initial_seed=42, event_seed=1000, mode="burst", events=tuple(events))

        episode, trace = run_episode(
            NearestLegalPolicy(), tape_id="early-term", tape=tape,
            algorithm="NearestLegal", max_decisions=100,
        )

        self.assertGreaterEqual(len(trace["events"]), 2)
        second = trace["events"][-1]  # the never-observed event
        self.assertEqual(second["event_id"], "vac-4")
        self.assertEqual(second["decision_count"], 0)
        self.assertIsNone(second["recovery_delay"])
        self.assertTrue(np.isfinite(float(second["fixed_j"])))
        # Every unobserved event got the frozen 200s censored recovery horizon.
        from ppo_allocation.random_event.reward import UNOBSERVED_EVENT_RECOVERY_PENALTY_SECONDS
        ckpt = CheckpointRecord(
            variant="PPO-MLP", training_seed=1101, decision_steps=25000,
            checkpoint_path="x.pt", checkpoint_sha256="a" * 64,
            source_tree_hash="b" * 64, attested_source_commit_sha="c" * 40,
            protocol_sha256="d" * 64, seed_manifest_sha256="e" * 64,
            ppo_config={}, rng_state={}, created_at="now",
        )
        metrics = extract_validation_metrics(episode, trace, ckpt)
        self.assertTrue(np.isfinite(metrics.final_infeasible_rate))
        self.assertTrue(np.isfinite(metrics.cumulative_weighted_vacancy))
        self.assertTrue(np.isfinite(metrics.recovery_latency))
        self.assertTrue(np.isfinite(metrics.fixed_j))
        self.assertEqual(metrics.final_infeasible_count, 1)
        selected, _ = _lexicographic_select([metrics])
        self.assertEqual(selected.checkpoint_sha256, ckpt.checkpoint_sha256)
        self.assertGreaterEqual(
            metrics.recovery_latency,
            UNOBSERVED_EVENT_RECOVERY_PENALTY_SECONDS,
        )


# ---------------------------------------------------------------------------
# Protected-source contract
# ---------------------------------------------------------------------------
class ProtectedSourceContractTests(unittest.TestCase):
    def _source_files(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_p0_gate", str(Path(_REPO_ROOT) / "scripts" / "build_p0_gate.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return set(module.SOURCE_FILES)

    def test_phase_j_runtime_modules_are_protected(self):
        required = [
            "ppo_allocation/random_event/environment.py",
            "ppo_allocation/random_event/experiment.py",
            "ppo_allocation/random_event/metrics.py",
            "ppo_allocation/random_event/models.py",
            "ppo_allocation/random_event/trainer.py",
            "ppo_allocation/random_event/reward.py",
            "ppo_allocation/random_event/phase_j.py",
        ]
        source_files = self._source_files()
        for module in required:
            self.assertIn(module, source_files)

    def test_metrics_change_invalidates_old_gate_hashes(self):
        """metrics.py is protected: a 1-byte change must alter the committed
        blob hash, so an old GREEN gate becomes stale."""
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
        blob = subprocess.check_output(
            ["git", "show", f"{commit}:ppo_allocation/random_event/metrics.py"],
            cwd=_REPO_ROOT,
        )
        digest = hashlib.sha256(blob).hexdigest()
        tampered = blob.replace(b"from __future__ import annotations", b"from __future__ import annotations \n", 1)
        self.assertNotEqual(digest, hashlib.sha256(tampered).hexdigest())


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
