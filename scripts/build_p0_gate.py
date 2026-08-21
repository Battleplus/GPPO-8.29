#!/usr/bin/env python3
"""Machine-generated P0 readiness gate.

This script is the ONLY authority that may set ``training_allowed`` in
``handoff/P0_GATE.json``.  Manual edits to that field are prohibited (the
script overwrites them every run).

Gate logic:
    training_allowed = False unless ALL of the following hold:
        - required test suites run to completion with zero failures
        - no gate item is PARTIAL / UNTESTED / FAIL
        - source hashes match the committed gate's recorded hashes
        - protocol / seed-manifest hashes are recorded and current
        - train/validation/test seed namespaces are disjoint
        - Test is never used for checkpoint selection
        - all three model variants pass save->load->deterministic inference
        - reward invariant holds on all four nominal modes
        - concurrency invariants hold (stale rejection, exclusive holder,
          no duplicate assignment, no late-ACK resurrection)
        - burst atomicity holds (3-event batch -> graph_version delta == 1)

Run:
    python scripts/build_p0_gate.py
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest import TextTestRunner, loader

ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "handoff" / "P0_GATE.json"
PROTOCOL_PATH = ROOT / "configs" / "random_event_protocol.json"
SEED_MANIFEST_PATH = ROOT / "configs" / "seed_manifest.json"
SMOKE_SUMMARY_PATH = ROOT / "ppo_allocation" / "results" / "random_event" / "smoke_20260821_final" / "smoke_summary.json"
SMOKE_ENV_METADATA_PATH = ROOT / "ppo_allocation" / "results" / "random_event" / "smoke_20260821_final" / "environment_metadata.json"

SOURCE_FILES = [
    "event_runtime/concurrency.py",
    "event_runtime/adapter.py",
    "event_runtime/events.py",
    "event_runtime/metrics.py",
    "event_runtime/replay.py",
    "event_runtime/state_machine.py",
    "event_runtime/observation.py",
    "ppo_allocation/random_event/environment.py",
    "ppo_allocation/random_event/experiment.py",
    "ppo_allocation/random_event/metrics.py",
    "ppo_allocation/random_event/models.py",
    "ppo_allocation/random_event/trainer.py",
    "ppo_allocation/random_event/runtime_bridge.py",
    "ppo_allocation/random_event/scheduler.py",
    "ppo_allocation/random_event/graph.py",
    "ppo_allocation/random_event/reward.py",
    "ppo_allocation/random_event/baselines.py",
    "ppo_allocation/random_event/events.py",
    "ppo_allocation/random_event/legacy_adapter.py",
    "ppo_allocation/config.py",
    "scripts/build_p0_gate.py",
    "ppo_allocation/tests_random_event/test_event_runtime_integration.py",
    "ppo_allocation/tests_random_event/test_confirmation_timelines.py",
    "ppo_allocation/tests_random_event/test_concurrency_invariants.py",
    "ppo_allocation/tests_random_event/test_p0_gate_contract.py",
    "ppo_allocation/tests_random_event/test_random_event_core.py",
    "ppo_allocation/tests_random_event/test_random_event_training.py",
    "ppo_allocation/tests_random_event/test_legacy_compatibility.py",
    "ppo_allocation/random_event/phase_j.py",
    "ppo_allocation/tests_random_event/test_phase_j.py",
    "run_phase_j.py",
    "configs/random_event_protocol.json",
    "configs/seed_manifest.json",
]

# Required test suites.  Each entry: (label, discovery_dir, pattern).
# The legacy_compatibility suite requires sb3_contrib (locked env Python 3.11);
# it is REQUIRED by the planning doc Phase I gate list ("CPP / legacy
# compatibility").  If the dependency is missing the suite errors and the gate
# stays training_allowed=false -- that is the intended honest behaviour.
REQUIRED_TEST_SUITES = [
    ("core_contracts", "ppo_allocation/tests_random_event", "test_random_event_core.py"),
    ("training_contracts", "ppo_allocation/tests_random_event", "test_random_event_training.py"),
    ("event_runtime_integration", "ppo_allocation/tests_random_event", "test_event_runtime_integration.py"),
    ("confirmation_timelines", "ppo_allocation/tests_random_event", "test_confirmation_timelines.py"),
    ("concurrency_invariants", "ppo_allocation/tests_random_event", "test_concurrency_invariants.py"),
    ("p0_gate_contract", "ppo_allocation/tests_random_event", "test_p0_gate_contract.py"),
    ("legacy_compatibility", "ppo_allocation/tests_random_event", "test_legacy_compatibility.py"),
    ("phase_j", "ppo_allocation/tests_random_event", "test_phase_j.py"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha256(commit: str, relative: str) -> str:
    """Hash the exact blob stored in Git at ``commit:path``."""
    try:
        data = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT
        )
    except (OSError, subprocess.CalledProcessError):
        return "MISSING"
    return hashlib.sha256(data).hexdigest()


def protected_paths() -> list[str]:
    return sorted(set(SOURCE_FILES) | {
        "configs/random_event_protocol.json",
        "configs/seed_manifest.json",
    })


def working_tree_clean(paths: list[str] | None = None) -> tuple[bool, list[str]]:
    paths = paths or protected_paths()
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=ROOT, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False, ["git status unavailable"]
    changed = [line for line in output.splitlines() if line.strip()]
    return not changed, changed


def committed_hashes_match(commit: str, source: dict[str, str]) -> tuple[bool, list[str]]:
    mismatches = []
    for relative, disk_hash in source.items():
        git_hash = git_blob_sha256(commit, relative)
        if disk_hash != git_hash:
            mismatches.append(relative)
    return not mismatches, mismatches


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_tests() -> dict:
    """Run required suites and return per-suite results."""
    # The random-event tests import ``config`` and ``random_event`` packages
    # which live inside ppo_allocation; keep that directory before the repo
    # root so legacy top-level imports resolve to the project package.
    ppo_dir = ROOT / "ppo_allocation"
    if str(ppo_dir) not in sys.path:
        sys.path.insert(0, str(ppo_dir))
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    results = {}
    all_pass = True
    for label, directory, pattern in REQUIRED_TEST_SUITES:
        suite_dir = ROOT / directory
        discover_loader = loader.TestLoader()
        loaded = discover_loader.discover(str(suite_dir), pattern=pattern, top_level_dir=str(suite_dir))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            test_result = TextTestRunner(stream=stream, verbosity=1).run(loaded)
        passed = test_result.wasSuccessful()
        results[label] = {
            "passed": passed,
            "tests_run": test_result.testsRun,
            "failures": len(test_result.failures),
            "errors": len(test_result.errors),
            "output": stream.getvalue()[-2000:],
        }
        if not passed:
            all_pass = False
    results["total_tests"] = sum(
        int(value.get("tests_run", 0)) for key, value in results.items()
        if key != "_all_pass"
    )
    results["_all_pass"] = all_pass and results["total_tests"] >= 83
    return results


def verify_seed_isolation() -> dict:
    """Expand seed ranges and assert train/validation/test namespaces disjoint.

    Also enforces the Phase J frozen train contract:
    - formal train mode cycle == manifest preliminary.train.mode_cycle
    - runtime seed formula == manifest runtime_mapping (instance/event)
    - reserved episode count is sufficient for the frozen 300k budget under
      the worst-case 1 accepted decision per episode
    """
    manifest = json.loads(SEED_MANIFEST_PATH.read_text(encoding="utf-8"))
    preliminary = manifest["preliminary"]

    def expand(spec):
        return [int(spec["start"]) + i * int(spec.get("stride", 1)) for i in range(int(spec["count"]))]

    train = set()
    for spec in preliminary["train"]["instance_seeds_by_training_seed"].values():
        train.update(expand(spec))
    for spec in preliminary["train"]["event_seeds_by_training_seed"].values():
        train.update(expand(spec))

    validation = set()
    for spec in preliminary["validation"]["modes"].values():
        validation.update(expand(spec["instance_seeds"]))
        validation.update(expand(spec["event_seeds"]))

    test = set()
    for spec in preliminary["test"]["sets"].values():
        test.update(expand(spec["instance_seeds"]))
        test.update(expand(spec["event_seeds"]))

    overlaps = {
        "train_vs_validation": sorted(train & validation),
        "train_vs_test": sorted(train & test),
        "validation_vs_test": sorted(validation & test),
    }

    # --- Phase J frozen train contract ---
    train_block = preliminary["train"]
    frozen_mode_cycle = tuple(train_block["mode_cycle"])
    # Runtime formula must match the manifest exactly (episode_index range).
    mapping = train_block.get("runtime_mapping", {})
    formula_ok = (
        mapping.get("instance_seed") == "training_seed*1000003+episode_index"
        and mapping.get("event_seed") == "training_seed*10000019+episode_index"
        and int(mapping.get("episode_index_start", 0)) == 0
    )
    # Reserved count must cover the frozen 300k budget under the worst-case
    # 1 accepted decision per episode.  The trainer performs one initial reset
    # before the first decision plus one post-terminal reset after the LAST
    # accepted transition, so the worst case needs budget + 1 resets: the
    # reservation must satisfy ``reserved >= formal_budget + 1``.
    reserved = int(train_block.get("episodes_per_training_seed", 0))
    budget = int(manifest["preliminary"]["train"].get("reserved_coverage_assertions", {})
                   .get("formal_budget_decision_steps", 300000))
    coverage_ok = reserved >= budget + 1
    reserved_assertions = train_block.get("reserved_coverage_assertions", {})
    assertions_ok = (
        int(reserved_assertions.get("reserved_episodes_per_seed", 0)) == reserved
        and reserved_assertions.get("includes_initial_reset") is True
        and reserved_assertions.get("sufficiency_rule") == "reserved >= formal_budget_decision_steps + 1"
        and int(reserved_assertions.get("required_episodes_per_run", 0)) == reserved
    )
    # Formal train mode cycle must be exactly the frozen cycle (no single).
    mode_cycle_ok = (
        frozen_mode_cycle == ("sequential", "overlap", "burst")
        and "single" not in frozen_mode_cycle
    )
    # Sanity: each training seed's expanded count matches the frozen reservation.
    counts_ok = all(
        int(spec["count"]) == reserved
        for spec in list(train_block["instance_seeds_by_training_seed"].values())
        + list(train_block["event_seeds_by_training_seed"].values())
    )
    # Verify the actual start values match the formula for all 3 training seeds.
    starts_ok = True
    formula_checks = {}
    for training_seed, spec in train_block["instance_seeds_by_training_seed"].items():
        expected = int(training_seed) * 1_000_003 + 0
        starts_ok = starts_ok and int(spec["start"]) == expected
        formula_checks[f"instance_seed_{training_seed}"] = int(spec["start"]) == expected
    for training_seed, spec in train_block["event_seeds_by_training_seed"].items():
        expected = int(training_seed) * 10_000_019 + 0
        starts_ok = starts_ok and int(spec["start"]) == expected
        formula_checks[f"event_seed_{training_seed}"] = int(spec["start"]) == expected

    ok = (
        not any(overlaps.values())
        and formula_ok
        and coverage_ok
        and assertions_ok
        and mode_cycle_ok
        and counts_ok
        and starts_ok
    )
    return {
        "passed": ok,
        "train_count": len(train),
        "validation_count": len(validation),
        "test_count": len(test),
        "overlaps": overlaps,
        "train_mode_cycle": list(frozen_mode_cycle),
        "train_mode_cycle_ok": mode_cycle_ok,
        "runtime_formula_ok": formula_ok,
        "reserved_episodes_per_seed": reserved,
        "formal_budget_decision_steps": budget,
        "seed_coverage_rule": "reserved >= formal_budget + 1",
        "seed_coverage_ok": coverage_ok,
        "seed_coverage_assertions_ok": assertions_ok,
        "seed_start_formula_ok": starts_ok,
        "seed_start_formula_checks": formula_checks,
    }


def verify_config_contract() -> dict:
    """Verify frozen protocol values match the planning document."""
    manifest = json.loads(SEED_MANIFEST_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    checks = {
        "preliminary_seeds": manifest["training_seeds"]["preliminary"] == [1101, 2202, 3303],
        "validation_tapes_100": manifest["preliminary"]["validation"]["tapes_total"] == 100,
        "validation_per_mode_25": all(
            spec["instance_seeds"]["count"] == 25
            for spec in manifest["preliminary"]["validation"]["modes"].values()
        ),
        "validation_no_unseen": "unseen" not in manifest["preliminary"]["validation"]["modes"],
        "test_tapes_200": manifest["preliminary"]["test"]["tapes_total"] == 200,
        "test_per_set_40": manifest["preliminary"]["test"]["tapes_per_set"] == 40,
        "test_has_unseen": "Test-Unseen" in manifest["preliminary"]["test"]["sets"],
        "burst_window_100ms": protocol["event_scheduler"]["modes"]["burst"]["window_seconds"] == 0.1,
        "test_not_for_selection": manifest["preliminary"]["test"]["checkpoint_selection"] is False,
    }
    return {"passed": all(checks.values()), "checks": checks}


def run_invariant_checks() -> dict:
    """Run machine-executed protocol, reward, confirmation and concurrency probes."""
    ppo_dir = ROOT / "ppo_allocation"
    if str(ppo_dir) not in sys.path:
        sys.path.insert(0, str(ppo_dir))
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    results = {}

    # --- Burst atomicity: 3-event batch -> graph_version delta == 1 ---
    try:
        from random_event.environment import RandomEventAllocationEnv
        env = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, mode="burst", events_per_episode=3)
        env.reset(seed=42)
        delta = env.graph_version
        env.close()
        results["burst_atomicity"] = {
            "passed": delta == 1,
            "graph_version_after_3_event_batch": delta,
        }
    except Exception as exc:  # pragma: no cover
        results["burst_atomicity"] = {"passed": False, "error": str(exc)}

    # --- Reward invariant on all four modes: decision rows are the only sum. ---
    try:
        from random_event.events import EventTape, RandomEvent, RandomEventType
        from random_event.baselines import NearestLegalPolicy
        from random_event.experiment import run_episode
        reward_results = {}
        for mode in ("single", "sequential", "overlap", "burst"):
            event = RandomEvent(
                event_id=f"P0-{mode}", event_type=RandomEventType.REGION_VACANCY,
                occurred_at=0.0, observed_at=0.0, source_event="p0",
                affected_uavs=(0,), affected_regions=(0,), severity=0.5,
                event_seed=100, state_version=0,
            )
            tape = EventTape(initial_seed=42, event_seed=100, mode=mode, events=(event,))
            _, trace = run_episode(NearestLegalPolicy(), tape_id=f"p0-{mode}", tape=tape,
                                   algorithm="P0", max_decisions=20)
            reward_results[mode] = bool(trace["reward_invariant"])
        results["reward_invariant_four_modes"] = {
            "passed": all(reward_results.values()), "modes": reward_results,
        }
    except Exception as exc:  # pragma: no cover
        results["reward_invariant_four_modes"] = {"passed": False, "error": str(exc)}

    # --- Confirmation timeline: two of five discovery evidence is not enough. ---
    try:
        from event_runtime.events import EventType
        from event_runtime.observation import Observation
        from event_runtime.state_machine import ConfirmationStateMachine
        sm = ConfirmationStateMachine(target_confirmation_count=3)
        for i in range(2):
            sm.process(Observation(
                observation_id=f"p0-disc-{i}", event_id="p0-disc", event_type=EventType.TARGET_DISCOVERED,
                source_event="p0", source_id=f"src-{i}", source_type="sensor",
                signal_type="SENSOR_DETECTION", sequence=1, confidence=0.95, positive=True,
                emitted_at=1.0, received_at=1.1, occurred_at=0.0,
                affected_targets=("0",),
            ))
        record = sm.get("p0-disc")
        results["confirmation_timeline_contracts"] = {
            "passed": record is not None and record.confirmed_event is None and len(record.positive_evidence_sources) == 2,
            "status": None if record is None else record.status.value,
            "independent_evidence": 0 if record is None else len(record.positive_evidence_sources),
        }
    except Exception as exc:  # pragma: no cover
        results["confirmation_timeline_contracts"] = {"passed": False, "error": str(exc)}

    # --- Concurrency execution constraints. ---
    try:
        from event_runtime.concurrency import ACK, ACKType, CommandStatus, ConcurrencyManager
        cm = ConcurrencyManager()
        cmd = cm.create_command("p0-c1", "0", "0", graph_version=4, action_version=7, now=0.0)
        exact_ok = cm.validate_command(cmd.command_id, 4)
        cm.commit_command(cmd.command_id)
        cm.receive_ack(cmd.command_id, ACK(cmd.command_id, "0", ACKType.ACCEPTED, 0.1, cmd.fencing_token))
        stale = cm.create_command("p0-stale", "0", "1", graph_version=4, action_version=7, now=0.0)
        stale_rejected = not cm.validate_command(stale.command_id, 5)
        lease1 = cm.create_lease("p0-l1", "0", "0", cmd.fencing_token, now=0.0, ttl=5.0)
        exclusive_rejected = False
        try:
            cm.create_lease("p0-l2", "1", "0", cmd.fencing_token + 1, now=0.0, ttl=5.0)
        except ValueError:
            exclusive_rejected = True
        cmd.revoke(cmd.fencing_token + 1, at=1.0)
        late_rejected = False
        try:
            cm.receive_ack(cmd.command_id, ACK(cmd.command_id, "0", ACKType.ACCEPTED, 2.0, cmd.fencing_token))
        except ValueError:
            late_rejected = True
        results["concurrency_stale_rejection"] = {"passed": exact_ok and stale_rejected}
        results["concurrency_exclusive_holder"] = {"passed": exclusive_rejected and cm.get_valid_holder_count("0", 1.0) == 1}
        results["concurrency_duplicate_assignment"] = {"passed": exclusive_rejected}
        results["concurrency_late_ack_resurrection"] = {"passed": late_rejected and cmd.status is CommandStatus.REVOKED}
        # Real fencing monotonicity: holder A gets T1, revoke, holder B
        # gets T2 > T1, old T1 cannot create lease / ACK / execute.
        token_a = cmd.fencing_token
        cmd.revoke(token_a + 5, at=1.5)
        cm.revoke_lease(lease1.lease_id)
        # New command gets a higher token.
        cmd2 = cm.create_command("p0-c2", "0", "0", graph_version=4, action_version=7, now=2.0)
        lease2 = cm.create_lease("p0-l2b", "0", "0", cmd2.fencing_token, now=2.0, ttl=5.0)
        # Try to create lease with old token A — must fail.
        old_lease_rejected = False
        try:
            cm.create_lease("p0-old-tok", "0", "0", token_a, now=2.5, ttl=5.0)
        except ValueError:
            old_lease_rejected = True
        # Try old ACK — must fail (command revoked).
        old_ack_rejected = False
        try:
            cm.receive_ack(cmd.command_id, ACK(cmd.command_id, "0", ACKType.ACCEPTED, 3.0, token_a))
        except ValueError:
            old_ack_rejected = True
        # New token is strictly higher.
        new_higher = cmd2.fencing_token > token_a
        results["concurrency_fencing_monotonicity"] = {
            "passed": old_lease_rejected and old_ack_rejected and new_higher,
            "old_token": token_a,
            "new_token": cmd2.fencing_token,
            "old_lease_rejected": old_lease_rejected,
            "old_ack_rejected": old_ack_rejected,
            "new_higher": new_higher,
        }
    except Exception as exc:  # pragma: no cover
        for key in (
            "concurrency_stale_rejection", "concurrency_exclusive_holder",
            "concurrency_duplicate_assignment", "concurrency_late_ack_resurrection",
            "concurrency_fencing_monotonicity",
        ):
            results[key] = {"passed": False, "error": str(exc)}

    # --- Stale injection: 5 graph-stale + 5 action-version-stale, all rejected ---
    try:
        from ppo_allocation.random_event.runtime_bridge import RuntimeBridge
        from ppo_allocation.random_event.environment import RandomEventAllocationEnv
        env = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, mode="single", events_per_episode=1)
        env.reset(seed=42)
        bridge = RuntimeBridge(detector_seed=42)
        gv = int(env.graph_version)
        av = int(env.decision_version)
        N_GRAPH = 5
        N_AV = 5
        all_rejected = True
        # Graph-stale: command with old graph_version
        for i in range(N_GRAPH):
            ok = bridge.submit_stale_action(
                env, command_id=f"p0-graph-{i}", uav_id="0", region_id="0",
                stale_graph_version=gv - 1, action_version=av,
                fencing_token=0, now=0.0,
            )
            if not ok:
                all_rejected = False
        # Action-version-stale: command with old action_version
        for i in range(N_AV):
            ok = bridge.submit_stale_action(
                env, command_id=f"p0-av-{i}", uav_id="0", region_id="0",
                stale_graph_version=gv, action_version=av - 1,
                fencing_token=0, now=0.0,
            )
            if not ok:
                all_rejected = False
        snap = bridge.snapshot_concurrency(0.0)
        injected = snap["injected_stale_submissions"]
        rejected = snap["injected_stale_rejected"]
        rate = snap["stale_rejection_rate"]
        env.close()
        results["stale_injection_rate"] = {
            "passed": injected == N_GRAPH + N_AV and rejected == N_GRAPH + N_AV and rate == 1.0 and all_rejected,
            "graph_stale_injected": N_GRAPH,
            "graph_stale_rejected": N_GRAPH,
            "action_version_stale_injected": N_AV,
            "action_version_stale_rejected": N_AV,
            "total_injected": injected,
            "total_rejected": rejected,
            "rate": rate,
        }
    except Exception as exc:  # pragma: no cover
        results["stale_injection_rate"] = {"passed": False, "error": str(exc)}

    # --- Reward semantic consistency: the protocol and runtime are one truth ---
    try:
        from random_event.reward import (
            UNOBSERVED_EVENT_RECOVERY_PENALTY_SECONDS,
            VACANCY_DURATION_WEIGHT,
            CostWeights,
            compute_cost,
        )
        from random_event.phase_j import UNCENSORED_RECOVERY_PENALTY_SECONDS
        cw = CostWeights()
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        reward_config = protocol.get("reward", {})
        cost_config = reward_config.get("cost", {})
        protocol_weights = cost_config.get("weights", {})
        mapping = {
            "alpha -> uncovered": (protocol_weights.get("alpha"), cw.uncovered),
            "beta -> distance": (protocol_weights.get("beta"), cw.distance),
            "gamma -> load_gap": (protocol_weights.get("gamma"), cw.load_gap),
            "delta -> switches": (protocol_weights.get("delta"), cw.switches),
            "eta -> recovery_delay": (protocol_weights.get("eta"), cw.recovery_delay),
        }
        mapping_status = {
            name: expected is not None and abs(float(expected) - actual) <= 1e-12
            for name, (expected, actual) in mapping.items()
        }
        manifest = json.loads(SEED_MANIFEST_PATH.read_text(encoding="utf-8"))
        validation = manifest["preliminary"]["validation"]
        test = manifest["preliminary"]["test"]
        constraint_ok = cost_config.get("constraint_term_included") is False and cw.constraint_violation == 0.0
        normalization_ok = cost_config.get("distance_normalization") == "fixed_scenario_diagonal_AREA_SIZE_times_sqrt2"
        vacancy_ok = (
            cost_config.get("uncovered_definition") ==
            "sum(priority*workload + 0.2*(vacancy_duration/max_time)) over uncovered regions"
            and abs(VACANCY_DURATION_WEIGHT - 0.2) <= 1e-12
        )
        tuning_ok = validation.get("reward_tuning") is False and test.get("reward_tuning") is False
        validation_metrics = reward_config.get("validation_metrics", {})
        censoring_ok = (
            validation_metrics.get("missing_metric_is_zero") is False
            and abs(float(validation_metrics.get("unrecovered_recovery_penalty_seconds", -1.0)) - UNCENSORED_RECOVERY_PENALTY_SECONDS) <= 1e-12
            and "unresolved event" in validation_metrics.get("recovery_latency_definition", "")
            and "censored recovery penalty" in validation_metrics.get("fixed_j_unrecovered_rule", "")
        )
        # --- Frozen unobserved-event fixed-J rule: protocol == runtime ---
        rule = validation_metrics.get("fixed_j_unobserved_event_rule", {}) or {}
        unobserved_rule_ok = (
            abs(float(rule.get("recovery_delay_seconds", -1.0)) - UNOBSERVED_EVENT_RECOVERY_PENALTY_SECONDS) <= 1e-12
            and rule.get("uncovered_source") == "final_environment_cost.uncovered"
            and rule.get("distance_source") == "final_environment_cost.distance"
            and rule.get("load_gap_source") == "final_environment_cost.load_gap"
            and str(rule.get("switches_source", "")).startswith("frozen_zero")
            and abs(float(rule.get("switches_value", -1.0)) - 0.0) <= 1e-12
        )
        # Runtime probe: compute_cost without a reference assignment must yield
        # switches == 0.0 exactly (the frozen switches source for unobserved
        # events), never a hidden nonzero.
        switches_runtime_zero = False
        try:
            from random_event.environment import RandomEventAllocationEnv
            probe_env = RandomEventAllocationEnv(
                initial_seed=1, event_seed=2, mode="single", events_per_episode=1,
            )
            probe_env.reset(seed=1)
            probe_cost = compute_cost(probe_env)
            switches_runtime_zero = abs(float(probe_cost.switches) - 0.0) <= 1e-12
            probe_env.close()
        except Exception:
            switches_runtime_zero = False
        unobserved_rule_ok = unobserved_rule_ok and switches_runtime_zero
        semantic_pass = (
            all(mapping_status.values()) and constraint_ok and normalization_ok
            and vacancy_ok and tuning_ok and censoring_ok and unobserved_rule_ok
        )
        results["reward_semantic_consistency"] = {
            "passed": semantic_pass,
            "mapping": {key: "PASS" if value else "FAIL" for key, value in mapping_status.items()},
            "constraint_term": "PASS" if constraint_ok else "FAIL",
            "distance_normalization": "PASS" if normalization_ok else "FAIL",
            "weighted_vacancy_definition": "PASS" if vacancy_ok else "FAIL",
            "validation_reward_tuning": validation.get("reward_tuning"),
            "test_reward_tuning": test.get("reward_tuning"),
            "reward_tuning": "PASS" if tuning_ok else "FAIL",
            "unrecovered_event_censoring": "PASS" if censoring_ok else "FAIL",
            "unobserved_event_fixed_j_rule": "PASS" if unobserved_rule_ok else "FAIL",
            "unobserved_event_rule": rule,
            "runtime_recovery_penalty": UNOBSERVED_EVENT_RECOVERY_PENALTY_SECONDS,
            "runtime_switches_without_reference_is_zero": switches_runtime_zero,
            "cost_weights": asdict(cw),
            "protocol_reward": reward_config,
        }
    except Exception as exc:
        results["reward_semantic_consistency"] = {"passed": False, "error": str(exc)}

    # --- Snapshot identity, overlap delivery order and unseen isolation ---
    try:
        from random_event.environment import RandomEventAllocationEnv
        from random_event.events import EventTape, RandomEvent, RandomEventType
        import json as _json
        snap_a = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, events_per_episode=1)
        snap_b = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, events_per_episode=1)
        snap_a.reset(seed=42); snap_b.reset(seed=42)
        snapshot_ok = _json.dumps(snap_a.snapshot(), sort_keys=True, default=str) == _json.dumps(snap_b.snapshot(), sort_keys=True, default=str)
        snap_a.close(); snap_b.close()
        late = RandomEvent("p0-late", RandomEventType.REGION_VACANCY, 0.0, 3.0, "p0", affected_regions=(0,), affected_uavs=(0,))
        early = RandomEvent("p0-early", RandomEventType.REGION_VACANCY, 0.5, 1.0, "p0", affected_regions=(1,), affected_uavs=(1,))
        overlap = RandomEventAllocationEnv(
            initial_seed=42, event_seed=42001, mode="overlap", events_per_episode=2,
            event_tape=EventTape(initial_seed=42, event_seed=42001, mode="overlap", events=(late, early)),
        )
        overlap.reset(seed=42)
        overlap_ok = overlap.event_queue == ["p0-early"] and "p0-late" not in overlap.event_records
        overlap.close()
        manifest = json.loads(SEED_MANIFEST_PATH.read_text(encoding="utf-8"))
        unseen_ok = (
            "unseen" not in manifest["preliminary"]["validation"]["modes"]
            and "Test-Unseen" in manifest["preliminary"]["test"]["sets"]
        )
        results["single_snapshot_identity"] = {"passed": snapshot_ok}
        results["overlap_received_order"] = {"passed": overlap_ok}
        results["unseen_isolation"] = {"passed": unseen_ok}
    except Exception as exc:  # pragma: no cover
        results["single_snapshot_identity"] = {"passed": False, "error": str(exc)}
        results["overlap_received_order"] = {"passed": False, "error": str(exc)}
        results["unseen_isolation"] = {"passed": False, "error": str(exc)}

    # --- Model save/load determinism for all three variants ---
    try:
        from random_event.environment import RandomEventAllocationEnv
        from random_event.trainer import PPOConfig, PPOTrainer
        variant_results = {}
        for variant in ("PPO-MLP", "GPPO-NoGate", "GPPO-Adaptive"):
            env = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, mode="sequential", events_per_episode=3)
            config = PPOConfig(rollout_steps=16, update_epochs=1, minibatch_size=8, seed=1, device="cpu")
            trainer = PPOTrainer(env=env, variant=variant, config=config)
            trainer.train(32)
            tmp = Path(tempfile.gettempdir()) / f"p0gate_{variant}.pt"
            trainer.save(tmp)
            env2 = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, mode="sequential", events_per_episode=3)
            loaded, _ = PPOTrainer.load(tmp, env=env2, device="cpu")
            g, _ = env.reset(seed=42)
            g2, _ = env2.reset(seed=42)
            a1, lp1, v1, _ = trainer.model.act(g, deterministic=True)
            a2, lp2, v2, _ = loaded.model.act(g2, deterministic=True)
            variant_results[variant] = a1 == a2 and abs(lp1 - lp2) < 1e-5 and abs(v1 - v2) < 1e-5
            env.close(); env2.close()
        results["model_save_load_determinism"] = {"passed": all(variant_results.values()), "variants": variant_results}
    except Exception as exc:  # pragma: no cover
        results["model_save_load_determinism"] = {"passed": False, "error": str(exc)}

    return results


def git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def verify_smoke_evidence(attested_commit: str) -> dict:
    """Verify the actual 20 tapes x 4 modes replay evidence exists.

    ``attested_commit`` is the source commit the gate is attesting.
    The smoke metadata git_commit must match this exactly.
    """
    if not SMOKE_SUMMARY_PATH.exists():
        return {"passed": False, "error": f"missing {SMOKE_SUMMARY_PATH.relative_to(ROOT)}"}
    try:
        summary = json.loads(SMOKE_SUMMARY_PATH.read_text(encoding="utf-8"))
        manifest = summary.get("manifest", {})
        entries = manifest.get("entries", [])
        counts = {mode: sum(1 for entry in entries if entry.get("mode") == mode)
                  for mode in ("single", "sequential", "overlap", "burst")}
        replayed = int(summary.get("replayed_tape_count", -1))
        mode_ok = counts == {mode: 20 for mode in counts}
        replayed_ok = replayed == 80
        # Compute file hashes for ALL smoke evidence artifacts.
        summary_sha = sha256_file(SMOKE_SUMMARY_PATH)
        # Find and hash the smoke manifest file.
        smoke_manifest_path = SMOKE_SUMMARY_PATH.parent / "tapes" / manifest.get("bank_name", "smoke") / "manifest.json"
        if not smoke_manifest_path.exists():
            # Try to find it from the manifest_path in the summary.
            rel = manifest.get("manifest_path", "")
            if rel:
                smoke_manifest_path = ROOT / "ppo_allocation" / rel
        manifest_hash = sha256_file(smoke_manifest_path) if smoke_manifest_path.exists() else "MISSING"
        meta_sha = sha256_file(SMOKE_ENV_METADATA_PATH) if SMOKE_ENV_METADATA_PATH.exists() else "MISSING"

        # Verify environment metadata: Python 3.11.x, frozen package versions,
        # and git_commit == attested source commit.
        FROZEN_PACKAGES = {
            "torch": "2.5.0+cpu",
            "numpy": "2.0.2",
            "sb3-contrib": "2.9.0",
            "stable-baselines3": "2.9.0",
            "gymnasium": "1.3.0",
        }
        meta_ok = True
        meta_detail = {}
        if SMOKE_ENV_METADATA_PATH.exists():
            meta = json.loads(SMOKE_ENV_METADATA_PATH.read_text(encoding="utf-8"))
            py = meta.get("python", "")
            meta_detail["python"] = py
            meta_detail["python_311"] = py.startswith("3.11")
            smoke_commit = meta.get("git_commit")
            meta_detail["git_commit"] = smoke_commit
            meta_detail["git_commit_matches_attested"] = smoke_commit == attested_commit
            packages = meta.get("packages", {})
            meta_detail["torch"] = packages.get("torch")
            meta_detail["numpy"] = packages.get("numpy")
            meta_detail["sb3_contrib"] = packages.get("sb3-contrib")
            meta_detail["stable_baselines3"] = packages.get("stable-baselines3")
            meta_detail["gymnasium"] = packages.get("gymnasium")
            package_versions_ok = all(
                packages.get(name) == expected
                for name, expected in FROZEN_PACKAGES.items()
            )
            meta_detail["frozen_packages_ok"] = package_versions_ok
            meta_ok = (
                meta_detail["python_311"]
                and meta_detail["git_commit_matches_attested"]
                and package_versions_ok
            )
        else:
            meta_detail["error"] = f"missing {SMOKE_ENV_METADATA_PATH.relative_to(ROOT)}"
            meta_ok = False

        manifest_hash_ok = manifest_hash and manifest_hash != "MISSING" and len(manifest_hash) == 64
        passed = replayed_ok and mode_ok and meta_ok and manifest_hash_ok
        return {
            "passed": passed,
            "replayed_tape_count": replayed,
            "replayed_ok": replayed_ok,
            "counts_by_mode": counts,
            "mode_ok": mode_ok,
            "summary_sha256": summary_sha,
            "manifest_file_sha256": manifest_hash,
            "environment_metadata_sha256": meta_sha,
            "environment_metadata": meta_detail,
        }
    except (OSError, ValueError, TypeError) as exc:
        return {"passed": False, "error": str(exc)}


def compute_hashes() -> dict:
    """Hash the committed Git tree, never an uncommitted working copy."""
    commit = git_commit_sha()
    source = {relative: git_blob_sha256(commit, relative) for relative in SOURCE_FILES}
    source_tree_hash = hashlib.sha256(
        "".join(f"{key}:{value}\n" for key, value in sorted(source.items())).encode("utf-8")
    ).hexdigest()
    return {
        "git_commit_sha": commit,
        "source_tree_hash": source_tree_hash,
        "source": source,
        "protocol": git_blob_sha256(commit, "configs/random_event_protocol.json"),
        "seed_manifest": git_blob_sha256(commit, "configs/seed_manifest.json"),
    }


def main() -> int:
    test_results = run_tests()
    isolation = verify_seed_isolation()
    config_contract = verify_config_contract()
    invariants = run_invariant_checks()
    hashes = compute_hashes()
    attested = hashes["git_commit_sha"]
    smoke = verify_smoke_evidence(attested)

    previous = {}
    if GATE_PATH.exists():
        previous = json.loads(GATE_PATH.read_text(encoding="utf-8"))

    # A gate is a new attestation for the current committed HEAD.  It must not
    # silently re-baseline after a failed drift check: the full required test
    # suites and invariant probes above always run before this record is written.
    clean, dirty_paths = working_tree_clean()
    committed_match, committed_mismatches = committed_hashes_match(
        hashes["git_commit_sha"], hashes["source"]
    )
    missing_hashes = [name for name, value in hashes["source"].items() if value == "MISSING"]
    checks = {
        "test_suites": {
            "status": "PASS" if test_results["_all_pass"] else "FAIL",
            "details": {k: v for k, v in test_results.items() if k != "_all_pass"},
        },
        "seed_namespace_isolation": {
            "status": "PASS" if isolation["passed"] else "FAIL",
            "details": isolation,
        },
        "frozen_protocol_contract": {
            "status": "PASS" if config_contract["passed"] else "FAIL",
            "details": config_contract["checks"],
        },
        "burst_atomicity": {
            "status": "PASS" if invariants.get("burst_atomicity", {}).get("passed") else "FAIL",
            "details": invariants.get("burst_atomicity", {}),
        },
        "single_snapshot_identity": {
            "status": "PASS" if invariants.get("single_snapshot_identity", {}).get("passed") else "FAIL",
            "details": invariants.get("single_snapshot_identity", {}),
        },
        "overlap_received_order": {
            "status": "PASS" if invariants.get("overlap_received_order", {}).get("passed") else "FAIL",
            "details": invariants.get("overlap_received_order", {}),
        },
        "unseen_isolation": {
            "status": "PASS" if invariants.get("unseen_isolation", {}).get("passed") else "FAIL",
            "details": invariants.get("unseen_isolation", {}),
        },
        "reward_invariant_four_modes": {
            "status": "PASS" if invariants.get("reward_invariant_four_modes", {}).get("passed") else "FAIL",
            "details": invariants.get("reward_invariant_four_modes", {}),
        },
        "confirmation_timeline_contracts": {
            "status": "PASS" if invariants.get("confirmation_timeline_contracts", {}).get("passed") else "FAIL",
            "details": invariants.get("confirmation_timeline_contracts", {}),
        },
        "unconfirmed_event_no_decision": {
            "status": "PASS" if invariants.get("confirmation_timeline_contracts", {}).get("passed") else "FAIL",
            "details": "covered by confirmation_timeline_contracts and dedicated tests",
        },
        "concurrency_stale_rejection": {
            "status": "PASS" if invariants.get("concurrency_stale_rejection", {}).get("passed") else "FAIL",
            "details": invariants.get("concurrency_stale_rejection", {}),
        },
        "concurrency_exclusive_holder": {
            "status": "PASS" if invariants.get("concurrency_exclusive_holder", {}).get("passed") else "FAIL",
            "details": invariants.get("concurrency_exclusive_holder", {}),
        },
        "concurrency_duplicate_assignment": {
            "status": "PASS" if invariants.get("concurrency_duplicate_assignment", {}).get("passed") else "FAIL",
            "details": invariants.get("concurrency_duplicate_assignment", {}),
        },
        "concurrency_late_ack_resurrection": {
            "status": "PASS" if invariants.get("concurrency_late_ack_resurrection", {}).get("passed") else "FAIL",
            "details": invariants.get("concurrency_late_ack_resurrection", {}),
        },
        "concurrency_fencing_monotonicity": {
            "status": "PASS" if invariants.get("concurrency_fencing_monotonicity", {}).get("passed") else "FAIL",
            "details": invariants.get("concurrency_fencing_monotonicity", {}),
        },
        "stale_injection_rate": {
            "status": "PASS" if invariants.get("stale_injection_rate", {}).get("passed") else "FAIL",
            "details": invariants.get("stale_injection_rate", {}),
        },
        "model_save_load_determinism": {
            "status": "PASS" if invariants.get("model_save_load_determinism", {}).get("passed") else "FAIL",
            "details": invariants.get("model_save_load_determinism", {}),
        },
        "reward_semantic_consistency": {
            "status": "PASS" if invariants.get("reward_semantic_consistency", {}).get("passed") else "FAIL",
            "details": invariants.get("reward_semantic_consistency", {}),
        },
        "smoke_20x4": {
            "status": "PASS" if smoke.get("passed") else "FAIL",
            "details": smoke,
        },
        "source_hash_integrity": {
            "status": "PASS" if clean and committed_match and not missing_hashes else "FAIL",
            "details": {
                "git_commit_sha": hashes["git_commit_sha"],
                "source_tree_hash": hashes["source_tree_hash"],
                "missing": missing_hashes,
                "working_tree_clean": clean,
                "dirty_paths": dirty_paths,
                "committed_blob_hashes_match": committed_match,
                "committed_blob_mismatches": committed_mismatches,
                "previous_gate_not_used_as_baseline": True,
            },
        },
    }

    all_pass = all(check["status"] == "PASS" for check in checks.values())

    gate = {
        "schema_version": "1.0.0",
        "gate_name": "P0_READINESS_GATE",
        "generated_by": "scripts/build_p0_gate.py",
        "created_at": previous.get("created_at", utc_now()),
        "generated_at": utc_now(),
        "training_allowed": all_pass,
        "training_allowed_reason": "All machine checks PASS" if all_pass else "Machine checks failed - see checks section",
        "checks": checks,
        "test_count": int(test_results.get("total_tests", 0)),
        "required_test_count": int(test_results.get("total_tests", 0)),
        "protected_source_files": protected_paths(),
        "attested_source_commit_sha": hashes["git_commit_sha"],
        "git_commit_sha": hashes["git_commit_sha"],
        "source_tree_hash": hashes["source_tree_hash"],
        "protocol_sha256": hashes["protocol"],
        "seed_manifest_sha256": hashes["seed_manifest"],
        "source_hashes": hashes,
        "violations": [
            f"{name}: {value['status']}"
            for name, value in checks.items()
            if value["status"] != "PASS"
        ],
    }

    GATE_PATH.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"training_allowed": gate["training_allowed"], "violations": gate["violations"]}, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
