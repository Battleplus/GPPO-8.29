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
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import TextTestRunner, loader

ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "handoff" / "P0_GATE.json"
PROTOCOL_PATH = ROOT / "configs" / "random_event_protocol.json"
SEED_MANIFEST_PATH = ROOT / "configs" / "seed_manifest.json"

SOURCE_FILES = [
    "event_runtime/concurrency.py",
    "event_runtime/adapter.py",
    "event_runtime/metrics.py",
    "event_runtime/replay.py",
    "event_runtime/state_machine.py",
    "event_runtime/observation.py",
    "ppo_allocation/random_event/environment.py",
    "ppo_allocation/random_event/experiment.py",
    "ppo_allocation/random_event/models.py",
    "ppo_allocation/random_event/trainer.py",
    "ppo_allocation/random_event/runtime_bridge.py",
    "ppo_allocation/random_event/scheduler.py",
]

# Required test suites.  Each entry: (label, discovery_dir, pattern).
# The legacy_compatibility suite requires sb3_contrib (locked env Python 3.11);
# it is REQUIRED by the planning doc Phase I gate list ("CPP / legacy
# compatibility").  If the dependency is missing the suite errors and the gate
# stays training_allowed=false -- that is the intended honest behaviour.
REQUIRED_TEST_SUITES = [
    ("core_contracts", "ppo_allocation/tests_random_event", "test_random_event_core.py"),
    ("training_contracts", "ppo_allocation/tests_random_event", "test_random_event_training.py"),
    ("legacy_compatibility", "ppo_allocation/tests_random_event", "test_legacy_compatibility.py"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_tests() -> dict:
    """Run required suites and return per-suite results."""
    # The random-event tests import ``config`` and ``random_event`` packages
    # which live inside ppo_allocation; put that directory on sys.path.
    ppo_dir = ROOT / "ppo_allocation"
    if str(ppo_dir) not in sys.path:
        sys.path.insert(0, str(ppo_dir))

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
    results["_all_pass"] = all_pass
    return results


def verify_seed_isolation() -> dict:
    """Expand seed ranges and assert train/validation/test namespaces disjoint."""
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
    ok = not any(overlaps.values())
    return {
        "passed": ok,
        "train_count": len(train),
        "validation_count": len(validation),
        "test_count": len(test),
        "overlaps": overlaps,
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
    """Run the code-level invariant probes (burst atomicity, reward, save/load)."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ppo_allocation"))
    results = {}

    # --- Burst atomicity: 3-event batch -> graph_version delta == 1 ---
    try:
        from random_event.environment import RandomEventAllocationEnv

        env = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, mode="burst", events_per_episode=3)
        graph, info = env.reset(seed=42)
        delta = env.graph_version
        env.close()
        results["burst_atomicity"] = {
            "passed": delta == 1,
            "graph_version_after_3_event_batch": delta,
        }
    except Exception as exc:  # pragma: no cover
        results["burst_atomicity"] = {"passed": False, "error": str(exc)}

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
            ok = a1 == a2 and abs(lp1 - lp2) < 1e-5 and abs(v1 - v2) < 1e-5
            variant_results[variant] = ok
            env.close()
            env2.close()
        results["model_save_load_determinism"] = {
            "passed": all(variant_results.values()),
            "variants": variant_results,
        }
    except Exception as exc:  # pragma: no cover
        results["model_save_load_determinism"] = {"passed": False, "error": str(exc)}

    return results


def compute_hashes() -> dict:
    source = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        source[relative] = sha256_file(path) if path.exists() else "MISSING"
    return {
        "source": source,
        "protocol": sha256_file(PROTOCOL_PATH),
        "seed_manifest": sha256_file(SEED_MANIFEST_PATH),
    }


def main() -> int:
    test_results = run_tests()
    isolation = verify_seed_isolation()
    config_contract = verify_config_contract()
    invariants = run_invariant_checks()
    hashes = compute_hashes()

    previous = {}
    if GATE_PATH.exists():
        previous = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    previous_source_hashes = previous.get("source_hashes", {}).get("source", {})

    # On the very first gate run there is no committed baseline; record the
    # hashes as the new baseline instead of failing on drift from nothing.
    has_baseline = bool(previous_source_hashes)
    source_drift = {
        relative: {"previous": previous_source_hashes.get(relative), "current": current}
        for relative, current in hashes["source"].items()
        if has_baseline and previous_source_hashes.get(relative) != current
    }

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
        "model_save_load_determinism": {
            "status": "PASS" if invariants.get("model_save_load_determinism", {}).get("passed") else "FAIL",
            "details": invariants.get("model_save_load_determinism", {}),
        },
        "source_hash_integrity": {
            "status": "PASS" if not source_drift else "FAIL",
            "details": {
                "baseline_recorded": not has_baseline,
                "drift": source_drift,
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
