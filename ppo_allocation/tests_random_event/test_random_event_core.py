"""Contract tests for the replayable random-event GPPO experiment.

Run from ``ppo_allocation`` with::

    python -m unittest discover -s tests_random_event -v

The tests intentionally use the public random-event APIs.  Hand-authored
event tapes are used for environment invariants so a scheduler regression
cannot make an environment test pass accidentally.
"""

from __future__ import annotations

import copy
import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from config import NO_TARGET, NO_UAV, TaskType
from random_event.baselines import (
    CurrentPendingExactPlannerPolicy,
    GreedyCostPolicy,
    MaskedRandomPolicy,
    MinLoadPolicy,
    NearestLegalPolicy,
)
from random_event.environment import RandomEventAllocationEnv, StaleDecisionError
from random_event.events import EventTape, RandomEvent, RandomEventType
from random_event.experiment import generate_protocol_bank
from random_event.graph import build_graph_state, decode_edge_action
from random_event.legacy_adapter import LegacyMLPPPOPolicy
from random_event.models import GraphActorCritic, make_adaptive_model, make_no_gate_model
from random_event.reward import assignment_map, compute_cost, cost_difference_reward
from random_event.scheduler import (
    RandomEventScheduler,
    SchedulerState,
    UNSEEN_EVENT_WEIGHTS,
    UNSEEN_TIMING,
)


def event(
    index: int,
    kind: RandomEventType,
    *,
    occurred: float = 0.0,
    observed: float | None = None,
    uavs: tuple[int, ...] = (),
    regions: tuple[int, ...] = (),
    targets: tuple[int, ...] = (),
) -> RandomEvent:
    """Create a concise, protocol-complete event for deterministic tests."""

    return RandomEvent(
        event_id=f"E{index:04d}",
        event_type=kind,
        occurred_at=occurred,
        observed_at=occurred if observed is None else observed,
        source_event="unit_test",
        affected_uavs=uavs,
        affected_regions=regions,
        affected_targets=targets,
        severity=0.75,
        payload={"case": index},
        event_seed=10_000 + index,
        state_version=index,
    )


def tape(*events: RandomEvent, seed: int = 11, mode: str = "sequential") -> EventTape:
    return EventTape(initial_seed=seed, event_seed=991, mode=mode, events=tuple(events))


def make_env(*events: RandomEvent, seed: int = 11, mode: str = "sequential", **kwargs):
    return RandomEventAllocationEnv(
        initial_seed=seed,
        event_seed=991,
        mode=mode,
        events_per_episode=max(1, len(events)),
        event_tape=tape(*events, seed=seed, mode=mode),
        max_decisions=kwargs.pop("max_decisions", 30),
        max_time=kwargs.pop("max_time", 100.0),
        **kwargs,
    )


def all_event_types_valid_state() -> SchedulerState:
    """A fixed state in which all four conditional event sets are non-empty."""

    return SchedulerState(
        uav_alive=(True, True, True, True),
        uav_tasks=("SEARCH", "SEARCH", "SEARCH", "TRACK"),
        uav_regions=((0,), (1,), (2,), (3,)),
        region_assignments=(0, 1, 2, 3),
        target_discovered=(False, True, False),
        target_tracked=(False, True, False),
        target_destroyed=(False, False, False),
        target_trackers=(-1, 3, -1),
        target_regions=(0, 1, 2),
        state_version=7,
    )


def first_legal_edge(graph) -> int:
    legal = torch.nonzero(graph.action_mask[:-1], as_tuple=False).flatten()
    if not len(legal):
        raise AssertionError("test setup expected at least one legal edge")
    return int(legal[0])


class SchedulerContractTests(unittest.TestCase):
    def test_unseen_profile_shifts_mixture_and_detection_delay_only(self):
        state = all_event_types_valid_state()
        scheduler = RandomEventScheduler(
            event_count=8,
            weights=UNSEEN_EVENT_WEIGHTS,
            timing=UNSEEN_TIMING,
        )
        probabilities = scheduler.conditional_probabilities(state)
        self.assertEqual(probabilities, dict(UNSEEN_EVENT_WEIGHTS))
        generated = scheduler.generate_tape(
            state,
            initial_seed=71,
            event_seed=72,
            mode="sequential",
            event_count=8,
            state_transition=lambda current, _event: current,
        )
        self.assertEqual({item.event_type for item in generated.events} <= set(RandomEventType), True)
        delays = [item.observed_at - item.occurred_at for item in generated.events]
        self.assertTrue(all(1.5 <= delay <= 3.0 for delay in delays))

    def test_protocol_banks_use_frozen_disjoint_validation_and_test_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            validation = generate_protocol_bank(
                output,
                tier="preliminary",
                split="validation",
                limit_per_set=4,
            )
            test = generate_protocol_bank(
                output,
                tier="preliminary",
                split="test",
                limit_per_set=4,
            )
        self.assertFalse(validation["complete_frozen_bank"])
        self.assertFalse(test["complete_frozen_bank"])
        self.assertTrue(validation["checkpoint_selection"])
        self.assertFalse(test["checkpoint_selection"])
        validation_seeds = {
            value
            for item in validation["entries"]
            for value in (item["initial_seed"], item["event_seed"])
        }
        test_seeds = {
            value
            for item in test["entries"]
            for value in (item["initial_seed"], item["event_seed"])
        }
        self.assertFalse(validation_seeds & test_seeds)
        unseen = [item for item in test["entries"] if item["set_name"] == "Test-Unseen"]
        self.assertEqual([item["initial_seed"] for item in unseen], [350001, 350002, 350003, 350004])
        self.assertEqual([item["mode"] for item in unseen], ["single", "sequential", "overlap", "burst"])
        self.assertTrue(all(item["distribution_profile"] == "unseen_shift_v1" for item in unseen))

    def test_all_four_event_types_are_conditionally_valid(self):
        scheduler = RandomEventScheduler()
        state = all_event_types_valid_state()
        self.assertEqual(set(scheduler.valid_event_types(state)), set(RandomEventType))
        probabilities = scheduler.conditional_probabilities(state)
        expected = {
            RandomEventType.UAV_DAMAGE: 0.30,
            RandomEventType.TARGET_DISCOVERED: 0.30,
            RandomEventType.TARGET_DESTROYED: 0.20,
            RandomEventType.REGION_VACANCY: 0.20,
        }
        self.assertEqual(probabilities, expected)

        # Every sampled subject must come from the explicit candidate set.
        rng = random.Random(123)
        for index in range(500):
            sampled = scheduler.sample_event(
                state,
                rng=rng,
                event_id=f"S{index}",
                occurred_at=float(index),
                observed_at=float(index),
                event_seed=index,
            )
            if sampled.event_type is RandomEventType.UAV_DAMAGE:
                self.assertIn(sampled.affected_uavs[0], state.candidates(sampled.event_type))
            elif sampled.event_type is RandomEventType.REGION_VACANCY:
                self.assertIn(sampled.affected_regions[0], state.candidates(sampled.event_type))
            else:
                self.assertIn(sampled.affected_targets[0], state.candidates(sampled.event_type))

    def test_empirical_distribution_matches_configured_weights(self):
        scheduler = RandomEventScheduler()
        state = all_event_types_valid_state()
        rng = random.Random(20260820)
        counts = {kind: 0 for kind in RandomEventType}
        samples = 20_000
        for index in range(samples):
            sampled = scheduler.sample_event(
                state,
                rng=rng,
                event_id=f"D{index}",
                occurred_at=0.0,
                observed_at=0.0,
                event_seed=index,
            )
            counts[sampled.event_type] += 1
        expected = {
            RandomEventType.UAV_DAMAGE: 0.30,
            RandomEventType.TARGET_DISCOVERED: 0.30,
            RandomEventType.TARGET_DESTROYED: 0.20,
            RandomEventType.REGION_VACANCY: 0.20,
        }
        for kind, probability in expected.items():
            self.assertAlmostEqual(counts[kind] / samples, probability, delta=0.015)

    def test_same_seed_produces_byte_identical_replay(self):
        scheduler = RandomEventScheduler(event_count=8)
        state = all_event_types_valid_state()
        # Freeze the scenario state to isolate the seed/replay contract.
        transition = lambda current, _event: current
        a = scheduler.generate_tape(
            state,
            initial_seed=17,
            event_seed=99,
            mode="sequential",
            event_count=8,
            state_transition=transition,
        )
        b = scheduler.generate_tape(
            state,
            initial_seed=17,
            event_seed=99,
            mode="sequential",
            event_count=8,
            state_transition=transition,
        )
        self.assertEqual(a.to_bytes(), b.to_bytes())
        self.assertEqual(EventTape.from_json(a.to_bytes()).to_bytes(), a.to_bytes())

    def test_four_timing_modes_have_distinct_contracts(self):
        scheduler = RandomEventScheduler(event_count=7)
        state = all_event_types_valid_state()
        transition = lambda current, _event: current
        tapes = {
            mode: scheduler.generate_tape(
                state,
                initial_seed=3,
                event_seed=404,
                mode=mode,
                event_count=7,
                state_transition=transition,
            )
            for mode in ("single", "sequential", "overlap", "burst")
        }
        single_gaps = np.diff([e.occurred_at for e in tapes["single"].events])
        sequential_gaps = np.diff([e.occurred_at for e in tapes["sequential"].events])
        overlap_gaps = np.diff([e.occurred_at for e in tapes["overlap"].events])
        burst_times = [e.occurred_at for e in tapes["burst"].events]
        self.assertTrue(np.all((8.0 <= single_gaps) & (single_gaps <= 12.0)))
        self.assertTrue(np.all((4.0 <= sequential_gaps) & (sequential_gaps <= 8.0)))
        self.assertTrue(np.all((0.25 <= overlap_gaps) & (overlap_gaps <= 0.75)))
        self.assertTrue(all(
            nxt.occurred_at < prev.observed_at
            for prev, nxt in zip(tapes["overlap"].events, tapes["overlap"].events[1:])
        ))
        self.assertEqual(burst_times[0], burst_times[1])
        self.assertEqual(burst_times[1], burst_times[2])
        self.assertGreater(burst_times[3], burst_times[2])
        self.assertEqual(burst_times[3], burst_times[4])
        self.assertEqual(burst_times[4], burst_times[5])


class EnvironmentContractTests(unittest.TestCase):
    def test_one_replay_applies_all_four_event_kinds(self):
        env = make_env(
            event(
                0,
                RandomEventType.TARGET_DISCOVERED,
                targets=(0,),
                uavs=(0,),
                regions=(0,),
            ),
            event(
                1,
                RandomEventType.TARGET_DESTROYED,
                occurred=2.0,
                targets=(0,),
                uavs=(0,),
            ),
            event(
                2,
                RandomEventType.REGION_VACANCY,
                occurred=4.0,
                regions=(1,),
                uavs=(1,),
            ),
            event(
                3,
                RandomEventType.UAV_DAMAGE,
                occurred=6.0,
                uavs=(2,),
                regions=(2,),
            ),
        )
        graph, _ = env.reset()
        terminated = truncated = False
        for _ in range(15):
            action = first_legal_edge(graph) if bool(graph.action_mask[:-1].any()) else graph.noop_action
            graph, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(len(env.event_records), 4)
        self.assertEqual(
            {runtime.event.event_type for runtime in env.event_records.values()},
            set(RandomEventType),
        )
        self.assertFalse(env.uavs[2].alive)
        self.assertTrue(env.targets[0].destroyed)

    def test_episode_does_not_terminate_after_first_recovery(self):
        env = make_env(
            event(0, RandomEventType.REGION_VACANCY, regions=(0,), uavs=(0,)),
            event(1, RandomEventType.REGION_VACANCY, occurred=5.0, regions=(1,), uavs=(1,)),
        )
        graph, info = env.reset()
        self.assertEqual(info["event_index"], 1)
        graph, _, terminated, truncated, info = env.step(first_legal_edge(graph))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["event_index"], 2)
        self.assertIn(1, info["pending_regions"])

    def test_unaffected_assignments_are_preserved(self):
        env = make_env(event(0, RandomEventType.REGION_VACANCY, regions=(0,), uavs=(0,)))
        graph, _ = env.reset()
        unaffected_before = {rid: env.regions[rid].assigned_uav for rid in (1, 2, 3)}
        env.step(first_legal_edge(graph))
        unaffected_after = {rid: env.regions[rid].assigned_uav for rid in (1, 2, 3)}
        self.assertEqual(unaffected_after, unaffected_before)

    def test_overlap_ingests_multiple_events_into_pending_queue(self):
        env = make_env(
            event(0, RandomEventType.REGION_VACANCY, regions=(0,), uavs=(0,)),
            event(1, RandomEventType.REGION_VACANCY, regions=(1,), uavs=(1,)),
            mode="overlap",
        )
        _, info = env.reset()
        self.assertEqual(set(info["pending_regions"]), {0, 1})
        self.assertEqual(info["event_queue"], ["E0000", "E0001"])
        self.assertEqual(info["communication_trigger_count"], 2)

    def test_observation_order_does_not_block_later_generated_event(self):
        # E0 occurred first but is reported late; E1 must be ingested at t=1
        # without waiting for E0's report at t=3.
        env = make_env(
            event(
                0,
                RandomEventType.REGION_VACANCY,
                occurred=0.0,
                observed=3.0,
                regions=(0,),
                uavs=(0,),
            ),
            event(
                1,
                RandomEventType.REGION_VACANCY,
                occurred=0.5,
                observed=1.0,
                regions=(1,),
                uavs=(1,),
            ),
            mode="overlap",
        )
        _, info = env.reset()
        self.assertEqual(info["event_queue"], ["E0001"])
        self.assertEqual(info["pending_regions"], [1])
        self.assertNotIn("E0000", info["event_records"])
        self.assertEqual(env.advance_time(2.0), ["E0000"])

    def test_stale_graph_version_is_rejected(self):
        env = make_env(
            event(0, RandomEventType.REGION_VACANCY, regions=(0,), uavs=(0,)),
            event(
                1,
                RandomEventType.REGION_VACANCY,
                occurred=0.25,
                observed=0.5,
                regions=(1,),
                uavs=(1,),
            ),
            mode="overlap",
        )
        env.reset()
        graph, version = env.begin_decision()
        action = first_legal_edge(graph)
        self.assertEqual(env.advance_time(0.5), ["E0001"])
        with self.assertRaises(StaleDecisionError):
            env.submit_action(action, version, strict=True)
        self.assertEqual(env.stale_rejection_count, 1)

    def test_edge_mask_and_noop_contract(self):
        env = make_env(event(0, RandomEventType.REGION_VACANCY, regions=(2,), uavs=(2,)))
        graph, _ = env.reset()
        self.assertEqual(graph.noop_action, 16)
        self.assertEqual(graph.num_actions, 17)
        self.assertFalse(bool(graph.action_mask[graph.noop_action]))
        legal_edges = torch.nonzero(graph.action_mask[:-1], as_tuple=False).flatten().tolist()
        self.assertTrue(legal_edges)
        self.assertTrue(all(decode_edge_action(graph, action)[1] == 2 for action in legal_edges))

        empty_env = make_env()
        empty_graph, _ = empty_env.reset()
        self.assertFalse(bool(empty_graph.action_mask[:-1].any()))
        self.assertTrue(bool(empty_graph.action_mask[empty_graph.noop_action]))

    def test_graph_node_edge_and_action_dimensions(self):
        env = make_env(event(0, RandomEventType.REGION_VACANCY, regions=(0,), uavs=(0,)))
        graph, _ = env.reset()
        self.assertEqual(tuple(graph.nodes["uav"].shape), (4, 12))
        self.assertEqual(tuple(graph.nodes["region"].shape), (4, 12))
        self.assertEqual(tuple(graph.nodes["target"].shape), (3, 16))
        self.assertEqual(tuple(graph.candidate_edges.shape), (16, 2))
        self.assertEqual(tuple(graph.action_mask.shape), (17,))
        relation = ("uav", "can_serve", "region")
        self.assertEqual(tuple(graph.edge_index[relation].shape), (2, 16))
        self.assertEqual(tuple(graph.edge_attr[relation].shape), (16, 5))

    def test_temporary_and_final_infeasibility_are_distinct(self):
        future_release = event(
            1,
            RandomEventType.TARGET_DESTROYED,
            occurred=10.0,
            targets=(0,),
            uavs=(0,),
        )
        temporary = make_env(
            event(0, RandomEventType.REGION_VACANCY, regions=(0,), uavs=(0,)),
            future_release,
        )
        temporary.reset()
        for uav in temporary.uavs.values():
            uav.task = TaskType.TRACK
        temporary.targets[0].discovered = True
        temporary.targets[0].tracked = True
        temporary.targets[0].tracker_id = 0
        temporary.uavs[0].target_id = 0
        self.assertTrue(temporary._is_temporarily_infeasible())
        self.assertFalse(temporary._is_final_infeasible())
        self.assertTrue(bool(build_graph_state(temporary).action_mask[-1]))

        final = make_env(event(0, RandomEventType.REGION_VACANCY, regions=(0,), uavs=(0,)))
        final.reset()
        for uav in final.uavs.values():
            uav.task = TaskType.TRACK
        self.assertFalse(final._is_temporarily_infeasible())
        self.assertTrue(final._is_final_infeasible())


class ModelRewardAndBaselineTests(unittest.TestCase):
    def setUp(self):
        self.env = make_env(event(0, RandomEventType.REGION_VACANCY, regions=(0,), uavs=(0,)))
        self.graph, _ = self.env.reset()

    def test_no_gate_and_adaptive_forward_save_load(self):
        for factory, adaptive in ((make_no_gate_model, False), (make_adaptive_model, True)):
            with self.subTest(adaptive=adaptive):
                torch.manual_seed(7)
                model = factory(self.graph, hidden_dim=16, layers=2)
                logits, value, diagnostics = model(self.graph)
                self.assertEqual(tuple(logits.shape), (17,))
                self.assertEqual(tuple(value.shape), ())
                self.assertTrue(torch.isfinite(logits[self.graph.action_mask]).all())
                self.assertTrue(math.isfinite(float(diagnostics["pre_mask_invalid_probability"])))
                self.assertEqual(model.config.adaptive_gate, adaptive)
                if adaptive:
                    self.assertTrue(all(
                        bool(((gate >= 0) & (gate <= 1)).all())
                        for gate in diagnostics["gates"].values()
                    ))
                else:
                    self.assertTrue(all(
                        torch.equal(gate, torch.ones_like(gate))
                        for gate in diagnostics["gates"].values()
                    ))

                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "model.pt"
                    model.save(checkpoint, extra={"sentinel": 42})
                    restored, metadata = GraphActorCritic.load(checkpoint)
                    restored_logits, restored_value, _ = restored(self.graph)
                    self.assertEqual(metadata["sentinel"], 42)
                    self.assertEqual(restored.config, model.config)
                    self.assertTrue(torch.equal(restored_logits, logits))
                    self.assertTrue(torch.equal(restored_value, value))

    def test_cost_difference_is_exact_componentwise_j_difference(self):
        reference = assignment_map(self.env)
        before = compute_cost(self.env, reference_assignments=reference)
        clone = copy.deepcopy(self.env)
        uid, rid = decode_edge_action(self.graph, first_legal_edge(self.graph))
        clone._assign_region_to_uav(rid, uid)
        clone.pending_regions.discard(rid)
        after = compute_cost(clone, reference_assignments=reference)
        reward, trace = cost_difference_reward(before, after)
        self.assertAlmostEqual(reward, before.total - after.total)
        self.assertAlmostEqual(reward, sum(trace["reward_components"].values()))
        self.assertEqual(trace["definition"], "J(before)-J(after)")
        self.assertGreater(reward, 0.0)

    def test_five_baseline_interfaces_plus_current_pending_exact_planner_are_legal(self):
        class DummyLegacyModel:
            def predict(self, observation, action_masks, deterministic):
                self.observation_shape = np.asarray(observation).shape
                self.mask_shape = np.asarray(action_masks).shape
                return np.zeros(4, dtype=np.int64), None

        dummy = DummyLegacyModel()
        policies = [
            LegacyMLPPPOPolicy(dummy),
            MaskedRandomPolicy(seed=5),
            NearestLegalPolicy(),
            MinLoadPolicy(),
            GreedyCostPolicy(),
        ]
        for policy in policies:
            with self.subTest(policy=policy.name):
                action = policy.select_action(self.env, self.graph)
                self.assertTrue(bool(self.graph.action_mask[action]))
        self.assertEqual(dummy.observation_shape, (165,))
        self.assertEqual(dummy.mask_shape, (24,))

        planner = CurrentPendingExactPlannerPolicy()
        planner_action = planner.select_action(self.env, self.graph)
        self.assertTrue(bool(self.graph.action_mask[planner_action]))


if __name__ == "__main__":
    unittest.main()
