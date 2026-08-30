from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from execution_preemption.contract import (
    CHECKPOINT_STEPS,
    EXPECTED_COMPARISON_METHODS,
    FIXED_EVALUATION_CHECKPOINT,
    LEARNED_METHODS,
    TRAINING_SCALES,
    TRAINING_SEEDS,
    TRAINING_STEPS_PER_RUN,
    TrainingContractError,
    load_training_contract,
    validate_training_contract,
)
from execution_preemption.metrics import ExecutionMetricAccumulator, evaluate_acceptance
from execution_preemption.reward import (
    HardSafetyViolation,
    REWARD_CONTRACT_ID,
    REWARD_WEIGHTS,
    TransitionSignals,
    compute_transition_reward,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "execution_training_contract_v1.json"


class RewardContractTests(unittest.TestCase):
    def test_exact_frozen_weighted_sum(self) -> None:
        signals = TransitionSignals(
            weighted_progress_gain=0.50,
            urgent_deadline_miss_rate=0.10,
            weighted_vacancy_time=0.20,
            progress_loss=0.10,
            starvation_exposure=0.10,
            switch_time=0.20,
            energy_consumed=0.10,
            normalized_distance=0.40,
            load_gap=0.20,
        )
        result = compute_transition_reward(signals)
        expected = sum(getattr(signals, name) * weight for name, weight in REWARD_WEIGHTS.items())
        self.assertAlmostEqual(result.reward, expected)
        self.assertEqual(result.contract_id, REWARD_CONTRACT_ID)
        self.assertFalse(result.hard_constraints_in_reward)
        self.assertTrue(result.eligible_for_learning)

    def test_safety_violation_is_hard_failure_not_reward_penalty(self) -> None:
        signals = TransitionSignals(resource_conflicts=1)
        with self.assertRaisesRegex(HardSafetyViolation, "resource_conflicts=1"):
            compute_transition_reward(signals)

    def test_all_soft_signals_require_unit_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "progress_loss"):
            TransitionSignals(progress_loss=1.0001)
        with self.assertRaisesRegex(ValueError, "normalized_distance"):
            TransitionSignals(normalized_distance=float("nan"))

    def test_hard_counts_require_non_negative_integers(self) -> None:
        with self.assertRaisesRegex(ValueError, "energy_safety_violations"):
            TransitionSignals(energy_safety_violations=-1)
        with self.assertRaisesRegex(ValueError, "resource_conflicts"):
            TransitionSignals(resource_conflicts=True)


class MetricsContractTests(unittest.TestCase):
    def test_metrics_keep_censoring_counts_and_tail_latency(self) -> None:
        accumulator = ExecutionMetricAccumulator("candidate", "tape-1", 4)
        accumulator.record_transition(
            TransitionSignals(
                weighted_vacancy_time=0.2,
                progress_loss=0.1,
                starvation_exposure=0.3,
                switch_time=0.2,
                energy_consumed=0.1,
                normalized_distance=0.4,
                load_gap=0.2,
            ),
            inference_latency_ms=1.0,
            preemption_response_latency=2.0,
        )
        accumulator.record_transition(
            TransitionSignals(weighted_vacancy_time=0.3, load_gap=0.4),
            inference_latency_ms=9.0,
            preemption_response_latency=4.0,
        )
        accumulator.record_event(urgent=True, deadline_missed=True, p0=True, p0_handled=True)
        accumulator.record_event(urgent=True, deadline_missed=False, p0=True, p0_handled=True)
        accumulator.record_displacement(resumed=True, recovery_latency=3.0)
        accumulator.record_displacement(resumed=False)
        accumulator.record_task_outcome(starved=False)
        accumulator.record_task_outcome(starved=True)

        metrics = accumulator.finalize()
        self.assertEqual(metrics.accepted_decision_count, 2)
        self.assertEqual(metrics.urgent_deadline_miss_rate, 0.5)
        self.assertEqual(metrics.p0_handling_rate, 1.0)
        self.assertEqual(metrics.normal_task_recovery_rate, 0.5)
        self.assertEqual(metrics.right_censored_recovery_count, 1)
        self.assertEqual(metrics.task_starvation_rate, 0.5)
        self.assertAlmostEqual(metrics.cumulative_weighted_vacancy, 0.5)
        self.assertAlmostEqual(metrics.mean_recovery_latency or -1.0, 3.0)
        self.assertAlmostEqual(metrics.mean_preemption_response_latency or -1.0, 3.0)
        self.assertAlmostEqual(metrics.inference_latency_mean_ms or -1.0, 5.0)
        self.assertAlmostEqual(metrics.inference_latency_p95_ms or -1.0, 8.6)
        self.assertAlmostEqual(metrics.inference_latency_p99_ms or -1.0, 8.92)

    def test_missing_denominators_remain_none(self) -> None:
        metrics = ExecutionMetricAccumulator("method", "empty", 8).finalize()
        self.assertIsNone(metrics.urgent_deadline_miss_rate)
        self.assertIsNone(metrics.p0_handling_rate)
        self.assertIsNone(metrics.normal_task_recovery_rate)
        self.assertIsNone(metrics.task_starvation_rate)

    def test_invalid_censoring_or_event_flags_are_rejected(self) -> None:
        accumulator = ExecutionMetricAccumulator("method", "bad", 4)
        with self.assertRaisesRegex(ValueError, "urgent"):
            accumulator.record_event(deadline_missed=True)
        with self.assertRaisesRegex(ValueError, "recovery_latency"):
            accumulator.record_displacement(resumed=True)
        with self.assertRaisesRegex(ValueError, "cannot have"):
            accumulator.record_displacement(resumed=False, recovery_latency=1.0)

    def test_acceptance_passes_only_with_all_hard_and_effect_gates(self) -> None:
        template = ExecutionMetricAccumulator("candidate", "paired", 16)
        template.record_event(urgent=True, p0=True, p0_handled=True)
        template.record_displacement(resumed=True, recovery_latency=1.0)
        candidate = replace(
            template.finalize(),
            urgent_deadline_miss_rate=0.17,
            cumulative_weighted_vacancy=80.0,
            normal_task_recovery_rate=0.95,
        )
        baseline = replace(
            candidate,
            algorithm_id="senior_legacy_method_v1",
            urgent_deadline_miss_rate=0.20,
            cumulative_weighted_vacancy=100.0,
        )
        result = evaluate_acceptance(candidate, baseline)
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.violations, ())
        self.assertAlmostEqual(result.urgent_deadline_miss_relative_improvement or 0.0, 0.15)
        self.assertAlmostEqual(result.cumulative_vacancy_relative_improvement or 0.0, 0.20)

    def test_acceptance_fails_closed_on_missing_or_zero_baseline(self) -> None:
        empty = ExecutionMetricAccumulator("candidate", "paired", 4).finalize()
        result = evaluate_acceptance(empty, replace(empty, algorithm_id="baseline"))
        self.assertEqual(result.status, "FAIL")
        self.assertIn("p0_handling_rate_must_equal_1", result.violations)
        self.assertIn("urgent_deadline_miss_rate_missing", result.violations)
        self.assertIn("cumulative_vacancy_baseline_zero", result.violations)


class TrainingContractTests(unittest.TestCase):
    def test_machine_readable_contract_matches_code(self) -> None:
        validated = load_training_contract(CONTRACT_PATH)
        self.assertFalse(validated.training_allowed)
        self.assertEqual(validated.learned_run_count, 36)
        self.assertEqual(validated.checkpoint_count, 72)
        self.assertEqual(len(validated.canonical_sha256), 64)

    def test_budget_seed_checkpoint_and_scale_are_frozen(self) -> None:
        self.assertEqual(TRAINING_SEEDS, (1101, 2202, 3303))
        self.assertEqual(TRAINING_STEPS_PER_RUN, 50_000)
        self.assertEqual(CHECKPOINT_STEPS, (25_000, 50_000))
        self.assertEqual(FIXED_EVALUATION_CHECKPOINT, 50_000)
        self.assertEqual(TRAINING_SCALES, (4, 8, 16))

    def test_all_seven_comparison_methods_are_explicit(self) -> None:
        self.assertEqual(len(EXPECTED_COMPARISON_METHODS), 7)
        self.assertEqual(len(LEARNED_METHODS), 4)
        self.assertIn("beam_mpc_v1", EXPECTED_COMPARISON_METHODS)
        self.assertIn("ppo_mlp_rule_arbiter_v1", LEARNED_METHODS)
        self.assertIn("gppo_adaptive_rule_arbiter_v1", LEARNED_METHODS)

    def test_training_contract_binds_policy_adapter_layout(self) -> None:
        value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            value["adapter"]["layout_sha256"],
            "f903860f4ede2ffd8a0ac79fdaf90486c06232169211d9d81e993b41ef9ec544",
        )
        self.assertEqual(value["adapter"]["flat_observation_dimension"], 37976)
        self.assertEqual(value["adapter"]["action_capacity"], 3073)
        self.assertTrue(value["adapter"]["flat_and_hetero_share_action_space"])
        self.assertTrue(value["adapter"]["torch_tensor_conversion_complete"])
        self.assertTrue(value["adapter"]["gym_rollout_smoke_complete"])
        self.assertFalse(value["adapter"]["native_pyg_required"])

    def test_contract_rejects_training_or_checkpoint_drift(self) -> None:
        value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        value["training"]["accepted_decision_steps_per_run"] = 50_001
        with self.assertRaisesRegex(TrainingContractError, "budget drift"):
            validate_training_contract(value)

        value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        value["training"]["checkpoint_selection"] = True
        with self.assertRaisesRegex(TrainingContractError, "selection"):
            validate_training_contract(value)

    def test_contract_rejects_legacy_reuse_or_launch_enablement(self) -> None:
        value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        value["compatibility"]["legacy_checkpoint_compatible"] = True
        with self.assertRaisesRegex(TrainingContractError, "incompatible"):
            validate_training_contract(value)

        value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        value["launch_gate"]["training_allowed"] = True
        with self.assertRaisesRegex(TrainingContractError, "disabled"):
            validate_training_contract(value)

    def test_launch_gate_contract_is_source_bound_and_exact(self) -> None:
        value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        launch_gate = value["launch_gate"]
        self.assertEqual(launch_gate["requires_legacy_required_tests_min"], 130)
        self.assertEqual(launch_gate["requires_execution_preemption_tests_min"], 111)
        self.assertTrue(launch_gate["requires_framework_rollout_smoke"])
        self.assertTrue(launch_gate["requires_training_runner_smoke"])
        self.assertTrue(launch_gate["evidence_whitelist_exact"])
        self.assertEqual(
            launch_gate["gate_evidence_path"],
            "experiments/dynamic_preemption/evidence_v1/EXECUTION_PREEMPTION_V1_GATE.json",
        )

    def test_training_hyperparameters_and_tape_namespace_are_frozen(self) -> None:
        value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        training = value["training"]
        self.assertEqual(training["training_tape_namespace"], "execution_preemption_v1/train")
        self.assertFalse(training["development_tape_reuse"])
        self.assertFalse(training["hidden_tape_reuse"])
        self.assertEqual(training["optimizer"], "Adam")
        self.assertEqual(training["learning_rate"], 0.0003)
        self.assertEqual(training["rollout_accepted_decision_steps"], 64)
        self.assertEqual(training["update_epochs"], 4)
        self.assertTrue(training["fresh_output_required"])
        self.assertFalse(training["resume_supported"])
        self.assertTrue(training["checkpoint_includes_optimizer_rng_provenance"])


if __name__ == "__main__":
    unittest.main()
