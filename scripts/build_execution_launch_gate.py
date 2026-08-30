#!/usr/bin/env python3
"""Build the source-bound Execution-Preemption V1 launch gate.

The script may authorize a future clean training worktree, but it never creates
an optimizer, loads or writes a checkpoint, or starts any evaluation phase.
"""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_preemption.contract import load_training_contract  # noqa: E402
from execution_preemption.gate import (  # noqa: E402
    GATE_NAME,
    GATE_RELATIVE_PATH,
    REMOTE_SOURCE_REF,
    git_head,
    git_tree,
    git_worktree_changes,
    sha256_file,
    source_hash_inventory,
)


DEFAULT_OUTPUT = ROOT / GATE_RELATIVE_PATH
DEV_ROOT = ROOT / "experiments" / "dynamic_preemption" / "dev_v1"
RUN_COUNT_PATTERN = re.compile(r"Ran\s+(\d+)\s+tests?")


class GateBuildError(RuntimeError):
    """Raised when a mandatory precondition cannot be measured."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateBuildError(f"missing or invalid JSON: {path.relative_to(ROOT)}") from exc
    if not isinstance(value, Mapping):
        raise GateBuildError(f"JSON root is not an object: {path.relative_to(ROOT)}")
    return value


def _remote_source_head() -> str:
    try:
        result = subprocess.run(
            ["git", "ls-remote", "origin", REMOTE_SOURCE_REF],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise GateBuildError("cannot verify the GitHub research branch head") from exc
    fields = result.stdout.strip().split()
    if len(fields) != 2 or fields[1] != REMOTE_SOURCE_REF or len(fields[0]) != 40:
        raise GateBuildError("GitHub research branch returned an invalid head")
    return fields[0]


def _run_test_suite(label: str, cwd: Path, start_dir: str, minimum: int) -> dict[str, object]:
    command = [sys.executable, "-m", "unittest", "discover", "-s", start_dir, "-q"]
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "FAIL", "label": label, "test_count": 0, "error": str(exc)}
    combined = result.stdout + result.stderr
    matches = RUN_COUNT_PATTERN.findall(combined)
    count = int(matches[-1]) if matches else 0
    passed = result.returncode == 0 and count >= minimum
    return {
        "status": "PASS" if passed else "FAIL",
        "label": label,
        "test_count": count,
        "minimum_required": minimum,
        "return_code": result.returncode,
        "failure_output_tail": "" if passed else combined[-4000:],
    }


def _all_false(value: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return all(value.get(key) is False for key in keys)


def _artifact_record(path: Path, passed: bool, details: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "PASS" if passed else "FAIL",
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path) if path.is_file() else "MISSING",
        "details": dict(details),
    }


def _validate_dev_manifest() -> dict[str, object]:
    path = DEV_ROOT / "manifest.json"
    value = _load_json(path)
    inventory = value.get("inventory", [])
    counters: Counter[str] = Counter()
    file_hash_ok = True
    canonical_hash_ok = True
    unique_paths: set[str] = set()
    unique_seeds: set[int] = set()
    if not isinstance(inventory, list):
        inventory = []
    for item in inventory:
        if not isinstance(item, Mapping):
            file_hash_ok = canonical_hash_ok = False
            continue
        relative = str(item.get("path", ""))
        target = DEV_ROOT / relative
        scenario = str(item.get("scenario_id", ""))
        counters[scenario] += 1
        unique_paths.add(relative)
        try:
            unique_seeds.add(int(item["case_seed"]))
        except (KeyError, TypeError, ValueError):
            pass
        if not target.is_file() or sha256_file(target) != item.get("sha256_file"):
            file_hash_ok = False
            continue
        tape = _load_json(target)
        if _canonical_sha256(tape) != item.get("sha256_canonical"):
            canonical_hash_ok = False
    passed = all((
        value.get("status") == "PASS",
        value.get("bank") == "Dynamic-Preemption-Dev",
        value.get("classification") == "development_only_not_held_out",
        value.get("scenario_count") == 10,
        value.get("tape_count") == 200,
        value.get("replayed_tape_count") == 200,
        value.get("decision_count") == 280,
        value.get("invariant_failures") == 0,
        value.get("paired") is True,
        value.get("checkpoint_selection") is False,
        value.get("training_started") is False,
        len(inventory) == 200,
        len(unique_paths) == 200,
        len(unique_seeds) == 200,
        len(counters) == 10,
        set(counters.values()) == {20},
        file_hash_ok,
        canonical_hash_ok,
    ))
    return _artifact_record(path, passed, {
        "scenario_count": len(counters),
        "tape_count": len(inventory),
        "scenario_cardinalities": dict(sorted(counters.items())),
        "unique_path_count": len(unique_paths),
        "unique_seed_count": len(unique_seeds),
        "file_sha256_inventory": file_hash_ok,
        "canonical_sha256_inventory": canonical_hash_ok,
    })


def _validate_allocator_replay() -> dict[str, object]:
    path = DEV_ROOT / "allocator_replay_summary.json"
    value = _load_json(path)
    results = value.get("results", [])
    passed = all((
        value.get("status") == "PASS",
        value.get("bank") == "Dynamic-Preemption-Dev",
        value.get("allocator_count") == 2,
        value.get("tape_count") == 200,
        value.get("allocator_tape_runs") == 400,
        value.get("model_effectiveness_evaluated") is False,
        value.get("checkpoint_selection") is False,
        value.get("training_started") is False,
        isinstance(results, list) and len(results) == 2,
        all(item.get("status") == "PASS" for item in results),
        all(item.get("tape_count") == 200 for item in results),
        all(item.get("invariant_failures") == 0 for item in results),
    ))
    return _artifact_record(path, passed, {
        "allocator_count": value.get("allocator_count"),
        "allocator_tape_runs": value.get("allocator_tape_runs"),
        "invariant_failures": sum(int(item.get("invariant_failures", -1)) for item in results)
        if isinstance(results, list) else -1,
    })


def _validate_graph_smoke() -> dict[str, object]:
    path = DEV_ROOT / "graph_schema_smoke.json"
    value = _load_json(path)
    scales = value.get("scales", [])
    passed = all((
        value.get("status") == "PASS",
        value.get("schema_id") == "execution-preemption-heterograph-v1",
        value.get("uav_counts") == [4, 8, 16, 32],
        value.get("scale_count") == 4,
        value.get("training_started") is False,
        value.get("old_checkpoint_compatible") is False,
        isinstance(scales, list) and len(scales) == 4,
        all(item.get("status") == "PASS" for item in scales),
    ))
    return _artifact_record(path, passed, {"uav_counts": value.get("uav_counts")})


def _validate_adapter_smoke() -> dict[str, object]:
    path = DEV_ROOT / "policy_adapter_smoke.json"
    value = _load_json(path)
    scales = value.get("scales", [])
    passed = all((
        value.get("status") == "PASS",
        value.get("uav_counts") == [4, 8, 16, 32],
        value.get("flat_observation_dimension") == 37976,
        value.get("action_capacity") == 3073,
        _all_false(value, (
            "training_allowed", "training_started", "validation_started",
            "freeze_started", "test_started", "hidden_evaluation_started",
            "checkpoint_loaded", "model_framework_loaded",
        )),
        isinstance(scales, list) and len(scales) == 4,
        all(item.get("status") == "PASS" for item in scales),
        all(item.get("shared_action_space") is True for item in scales),
        all(item.get("noop_enabled") is False for item in scales),
    ))
    return _artifact_record(path, passed, {
        "uav_counts": value.get("uav_counts"),
        "flat_observation_dimension": value.get("flat_observation_dimension"),
        "action_capacity": value.get("action_capacity"),
    })


def _validate_deferred_parity() -> dict[str, object]:
    path = DEV_ROOT / "deferred_transaction_parity.json"
    value = _load_json(path)
    results = value.get("results", [])
    decision_parity = sum(int(item.get("decision_parity_pass_count", 0)) for item in results)
    state_parity = sum(int(item.get("state_sha256_parity_pass_count", 0)) for item in results)
    passed = all((
        value.get("status") == "PASS",
        value.get("allocator_tape_runs") == 400,
        value.get("graph_version_increment_per_atomic_batch") == 1,
        value.get("live_runtime_mutated_before_batch_commit") is False,
        decision_parity == 400,
        state_parity == 400,
        _all_false(value, (
            "training_allowed", "training_started", "validation_started",
            "freeze_started", "test_started", "hidden_evaluation_started",
            "checkpoint_loaded", "model_framework_loaded",
        )),
    ))
    return _artifact_record(path, passed, {
        "decision_parity_pass_count": decision_parity,
        "state_sha256_parity_pass_count": state_parity,
        "live_runtime_mutated_before_batch_commit": value.get(
            "live_runtime_mutated_before_batch_commit"
        ),
    })


def _validate_training_contract_smoke() -> dict[str, object]:
    path = DEV_ROOT / "training_contract_smoke.json"
    value = _load_json(path)
    contract = value.get("contract", {})
    passed = all((
        value.get("status") == "PASS",
        value.get("classification") == "training_precondition_contract_smoke_not_model_evidence",
        value.get("legacy_checkpoint_compatible") is False,
        value.get("legacy_evidence_reusable") is False,
        value.get("checkpoint_selection") is False,
        value.get("source_bound_launch_gate_created") is False,
        _all_false(value, (
            "training_allowed", "training_started", "validation_started",
            "freeze_started", "test_started", "hidden_evaluation_started",
        )),
        isinstance(contract, Mapping),
        contract.get("contract_id") == "execution-preemption-training-v1",
        contract.get("training_allowed") is False,
        contract.get("learned_run_count") == 36,
        contract.get("checkpoint_count") == 72,
    ))
    return _artifact_record(path, passed, {
        "contract_sha256": contract.get("canonical_sha256") if isinstance(contract, Mapping) else None,
        "learned_run_count": contract.get("learned_run_count") if isinstance(contract, Mapping) else None,
        "checkpoint_count": contract.get("checkpoint_count") if isinstance(contract, Mapping) else None,
    })


def _validate_framework_smoke() -> dict[str, object]:
    path = DEV_ROOT / "framework_rollout_smoke.json"
    value = _load_json(path)
    runs = value.get("runs", [])
    safety_zero = isinstance(runs, list) and all(
        item.get("mask_violations") == 0
        and item.get("resource_conflicts") == 0
        and item.get("stale_command_resurrections") == 0
        and item.get("energy_safety_violations") == 0
        for item in runs
    )
    passed = all((
        value.get("status") == "PASS",
        value.get("classification") == "framework_rollout_smoke_not_training_or_model_evidence",
        value.get("gppo_backend") == "repository_native_pytorch_relation_aware",
        value.get("torch_tensor_conversion") == "PASS",
        value.get("gym_reset_step_contract") == "PASS",
        value.get("run_count") == 4,
        isinstance(runs, list) and len(runs) == 4,
        all(item.get("status") == "PASS" for item in runs),
        safety_zero,
        value.get("optimizer_created") is False,
        value.get("optimizer_step_count") == 0,
        _all_false(value, (
            "training_allowed", "training_started", "validation_started",
            "freeze_started", "test_started", "hidden_evaluation_started",
            "checkpoint_loaded", "checkpoint_written", "model_weights_loaded",
        )),
        isinstance(value.get("pyg_health"), Mapping),
        value["pyg_health"].get("required_for_current_gppo") is False,
    ))
    return _artifact_record(path, passed, {
        "run_count": value.get("run_count"),
        "optimizer_step_count": value.get("optimizer_step_count"),
        "safety_zero": safety_zero,
        "pyg_health": value.get("pyg_health"),
    })


def _validate_training_runner_smoke() -> dict[str, object]:
    path = DEV_ROOT / "training_runner_smoke.json"
    value = _load_json(path)
    methods = value.get("methods", [])
    passed = all((
        value.get("status") == "PASS",
        value.get("classification")
        == "tiny_training_runner_smoke_not_model_effectiveness_evidence",
        value.get("learned_method_count") == 4,
        value.get("accepted_decision_steps_per_smoke_run") == 2,
        value.get("checkpoint_steps_per_smoke_run") == [1, 2],
        value.get("real_optimizer_updates") is True,
        value.get("fresh_temporary_output_per_run") is True,
        value.get("legacy_checkpoint_loaded") is False,
        value.get("old_campaign_reused") is False,
        value.get("formal_training_allowed") is False,
        _all_false(value, (
            "formal_training_started", "validation_started", "freeze_started",
            "test_started", "hidden_evaluation_started", "checkpoint_selection",
        )),
        isinstance(methods, list) and len(methods) == 4,
        all(item.get("status") == "PASS" for item in methods),
        all(item.get("optimizer_step_count") == 2 for item in methods),
        all(item.get("checkpoint_file_sha_verified") is True for item in methods),
        all(item.get("same_seed_state_determinism") is True for item in methods),
    ))
    return _artifact_record(path, passed, {
        "learned_method_count": value.get("learned_method_count"),
        "real_optimizer_updates": value.get("real_optimizer_updates"),
        "formal_training_started": value.get("formal_training_started"),
    })


def _check(status: bool, details: Mapping[str, Any]) -> dict[str, object]:
    return {"status": "PASS" if status else "FAIL", "details": dict(details)}


def build_gate() -> dict[str, object]:
    source = git_head(ROOT)
    source_tree = git_tree(ROOT, source)
    dirty = git_worktree_changes(ROOT)
    remote_source = _remote_source_head()
    protected_hashes = source_hash_inventory(ROOT, source)
    contract_path = ROOT / "configs" / "execution_training_contract_v1.json"
    contract = load_training_contract(contract_path)

    tests_legacy = _run_test_suite(
        "legacy_minimum_validation_required",
        ROOT / "ppo_allocation",
        "tests_random_event",
        130,
    )
    tests_execution = _run_test_suite(
        "execution_preemption_required",
        ROOT,
        "tests_execution_preemption",
        111,
    )
    artifact_builders: dict[str, Callable[[], dict[str, object]]] = {
        "development_bank_10x20": _validate_dev_manifest,
        "allocator_replay_400": _validate_allocator_replay,
        "graph_schema_4_8_16_32": _validate_graph_smoke,
        "policy_adapter_4_8_16_32": _validate_adapter_smoke,
        "deferred_transaction_parity_400": _validate_deferred_parity,
        "training_contract_smoke": _validate_training_contract_smoke,
        "framework_rollout_smoke": _validate_framework_smoke,
        "training_runner_smoke": _validate_training_runner_smoke,
    }
    artifact_checks = {name: builder() for name, builder in artifact_builders.items()}

    hidden_path = ROOT / "experiments" / "dynamic_preemption" / "hidden_v1"
    training_path = ROOT / "experiments" / "dynamic_preemption" / "training_v1"
    checks: dict[str, dict[str, object]] = {
        "source_is_clean_before_attestation": _check(not dirty, {"dirty_paths": list(dirty)}),
        "github_source_binding": _check(source == remote_source, {
            "local_head_sha": source,
            "remote_source_commit_sha": remote_source,
            "remote_source_ref": REMOTE_SOURCE_REF,
        }),
        "protected_source_inventory": _check(bool(protected_hashes), {
            "protected_file_count": len(protected_hashes),
            "source_tree_sha": source_tree,
        }),
        "legacy_gate_and_evidence_not_reused": _check(True, {
            "new_gate_path": GATE_RELATIVE_PATH,
            "legacy_gate_path": "handoff/P0_GATE.json",
            "legacy_gate_read": False,
            "legacy_smoke_read": False,
            "legacy_checkpoint_read": False,
        }),
        "training_contract_frozen": _check(
            contract.status == "FROZEN_FOR_SOURCE_ATTESTATION"
            and contract.training_allowed is False
            and contract.learned_run_count == 36
            and contract.checkpoint_count == 72,
            {
                "contract_id": contract.contract_id,
                "contract_sha256": contract.canonical_sha256,
                "contract_status": contract.status,
                "training_allowed_in_static_contract": contract.training_allowed,
                "learned_run_count": contract.learned_run_count,
                "checkpoint_count": contract.checkpoint_count,
            },
        ),
        "legacy_required_tests_130": tests_legacy,
        "execution_preemption_required_tests": tests_execution,
        "hidden_bank_not_generated": _check(not hidden_path.exists(), {
            "path": hidden_path.relative_to(ROOT).as_posix(),
            "exists": hidden_path.exists(),
        }),
        "formal_training_output_not_created": _check(not training_path.exists(), {
            "path": training_path.relative_to(ROOT).as_posix(),
            "exists": training_path.exists(),
        }),
        **artifact_checks,
    }
    violations = [name for name, check in checks.items() if check.get("status") != "PASS"]
    return {
        "schema_version": 1,
        "gate_name": GATE_NAME,
        "generated_by": "scripts/build_execution_launch_gate.py",
        "classification": "source_bound_pretraining_launch_evidence_not_model_effectiveness_evidence",
        "training_allowed": not violations,
        "violations": violations,
        "attested_source_commit_sha": source,
        "attested_source_tree_sha": source_tree,
        "remote_source_ref": REMOTE_SOURCE_REF,
        "remote_source_commit_sha": remote_source,
        "allowed_evidence_paths": [GATE_RELATIVE_PATH],
        "protected_source_file_count": len(protected_hashes),
        "protected_source_sha256": protected_hashes,
        "required_test_count": int(tests_legacy.get("test_count", 0)),
        "execution_preemption_test_count": int(tests_execution.get("test_count", 0)),
        "total_test_count": int(tests_legacy.get("test_count", 0))
        + int(tests_execution.get("test_count", 0)),
        "checks": checks,
        "formal_check": "execution_preemption.gate._check_execution_launch_gate",
        "formal_training_started": False,
        "validation_started": False,
        "freeze_started": False,
        "test_started": False,
        "hidden_evaluation_started": False,
        "checkpoint_selection": False,
    }


def write_gate(path: Path, gate: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(gate, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output != DEFAULT_OUTPUT.resolve():
        raise GateBuildError(f"gate output must be {GATE_RELATIVE_PATH}")
    gate = build_gate()
    write_gate(output, gate)
    print(json.dumps({
        "training_allowed": gate["training_allowed"],
        "violations": gate["violations"],
        "attested_source_commit_sha": gate["attested_source_commit_sha"],
        "output": GATE_RELATIVE_PATH,
    }, ensure_ascii=False, indent=2))
    return 0 if gate["training_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
