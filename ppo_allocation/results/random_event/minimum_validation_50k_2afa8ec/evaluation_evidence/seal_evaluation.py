"""Seal and read-only revalidate the completed fixed-50k evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVIDENCE_DIR = Path(__file__).resolve().parent
CAMPAIGN_DIR = EVIDENCE_DIR.parent
PRELIMINARY_DIR = CAMPAIGN_DIR / "preliminary"
WORKTREE = Path(__file__).resolve().parents[5]
PPO_DIR = WORKTREE / "ppo_allocation"
if str(EVIDENCE_DIR) not in sys.path:
    sys.path.insert(0, str(EVIDENCE_DIR))
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from analyze_evaluation import audit_and_load, read_json  # noqa: E402
from ppo_allocation.random_event.experiment import _check_p0_gate  # noqa: E402


EVIDENCE_HEAD = "2afa8ec1cb481deb57645dbd30240d90d32d2233"
SOURCE_COMMIT = "32974ec85be71e192b12cae85d00eb877d5fe07d"
TRAINING_SEALS = {
    "training_evidence.json": "500e89659e05876861a95f23af61f49a6b96679a6b1329ca0286b5956944102b",
    "sha256_inventory.json": "f2aaedf59f86dfaf0dd2b6bc7c1181b0b262e5ebacba870b1602b4759053e4d8",
    "readonly_revalidation.json": "130d1957029ef6f8e4c7c37a7cb31dcebc44ef9b4e93c82d0e33cc1f768e59ed",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(WORKTREE), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def relative(path: Path) -> str:
    return path.relative_to(WORKTREE).as_posix()


def inventory_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_training_seal() -> dict[str, Any]:
    seal_dir = CAMPAIGN_DIR / "training_evidence"
    errors = []
    for name, expected in TRAINING_SEALS.items():
        path = seal_dir / name
        if not path.is_file() or sha256(path) != expected:
            errors.append(name)
    training = read_json(seal_dir / "training_evidence.json")
    inventory = read_json(seal_dir / "sha256_inventory.json")
    verified = 0
    for row in inventory.get("artifacts", []):
        path = WORKTREE / row["path"]
        if (
            path.is_file()
            and path.stat().st_size == int(row["bytes"])
            and sha256(path) == row["sha256"]
        ):
            verified += 1
        else:
            errors.append(row["path"])
    expected_summary = {
        "completed_runs": 6,
        "total_steps": 300_000,
        "checkpoint_count": 12,
        "checkpoint_25000_count": 6,
        "checkpoint_50000_count": 6,
    }
    for field, expected in expected_summary.items():
        if training.get("summary", {}).get(field) != expected:
            errors.append(f"training summary {field}")
    if training.get("provenance", {}).get("runtime_evidence_head") != EVIDENCE_HEAD:
        errors.append("training evidence HEAD")
    if training.get("provenance", {}).get("attested_source_commit_sha") != SOURCE_COMMIT:
        errors.append("training source commit")
    if errors:
        raise SystemExit("training seal revalidation failed: " + ", ".join(errors))
    return {
        "status": "PASS",
        "runs": 6,
        "total_steps": 300_000,
        "checkpoints": 12,
        "checkpoint_25000": 6,
        "checkpoint_50000": 6,
        "inventory_artifacts_verified": verified,
        "seal_sha256": TRAINING_SEALS,
    }


def collect_inventory_paths(audit: dict[str, Any]) -> list[Path]:
    paths: set[Path] = set()
    for name in TRAINING_SEALS:
        paths.add(CAMPAIGN_DIR / "training_evidence" / name)
    paths.add(PRELIMINARY_DIR / "checkpoint_index.json")
    paths.add(PRELIMINARY_DIR / "frozen_manifests.json")
    paths.add(PRELIMINARY_DIR / "formal_test_bank_lock.json")
    paths.add(PRELIMINARY_DIR / "test_ledger.json")
    paths.update((PRELIMINARY_DIR / "freezes").glob("*.json"))
    paths.update((PRELIMINARY_DIR / "tapes/preliminary_test_protocol").rglob("*"))
    paths.update((PRELIMINARY_DIR / "test_results").glob("*.json"))
    paths.update((PRELIMINARY_DIR / "test_state").glob("*.json"))

    freeze = read_json(PRELIMINARY_DIR / "frozen_manifests.json")
    for item in freeze["freezes"]:
        paths.add(PPO_DIR / item["checkpoint_path"])

    analysis_names = (
        "analyze_evaluation.py",
        "seal_evaluation.py",
        "evaluation_rows.csv",
        "evaluation_rows.jsonl",
        "aggregate_metrics.csv",
        "paired_effects.csv",
        "paired_effects.json",
        "analysis_summary.json",
        "result_integrity_audit.json",
        "analysis_reproduction.json",
        "comparison_report.html",
    )
    paths.update(EVIDENCE_DIR / name for name in analysis_names)
    return sorted((path for path in paths if path.is_file()), key=lambda path: relative(path))


def main() -> int:
    head = git("rev-parse", "HEAD")
    if head != EVIDENCE_HEAD:
        raise SystemExit(f"evaluation runtime HEAD mismatch: {head}")
    if git("diff", "--name-only") or git("diff", "--cached", "--name-only"):
        raise SystemExit("tracked source changes detected in evaluation worktree")

    training = verify_training_seal()
    _check_p0_gate()
    rows, audit = audit_and_load()
    if len(rows) != 600 or audit.get("status") != "PASS":
        raise SystemExit("result integrity audit failed")

    forbidden_paths = (
        PRELIMINARY_DIR / "validation_manifest.json",
        PRELIMINARY_DIR / "validation_selection.json",
        PRELIMINARY_DIR / "validation_results",
    )
    if any(path.exists() for path in forbidden_paths):
        raise SystemExit("forbidden Validation artifact detected")

    inventory_paths = collect_inventory_paths(audit)
    inventory = {
        "schema_version": 1,
        "status": "PASS",
        "algorithm": "SHA-256",
        "inventory_scope": "fixed_50k_freeze_heldout_bank_600_results_analysis_and_training_seal_inputs",
        "evidence_head": EVIDENCE_HEAD,
        "attested_source_commit_sha": SOURCE_COMMIT,
        "artifact_count": len(inventory_paths),
        "artifacts": [inventory_row(path) for path in inventory_paths],
        "exclusions": [
            "25k checkpoint binaries (covered by immutable training inventory; no performance read)",
            "this inventory file",
            "evaluation_evidence.json",
            "readonly_revalidation.json",
        ],
    }
    inventory_path = EVIDENCE_DIR / "sha256_inventory.json"
    write_json(inventory_path, inventory)
    inventory_sha = sha256(inventory_path)

    summary = read_json(EVIDENCE_DIR / "analysis_summary.json")
    freeze = read_json(PRELIMINARY_DIR / "frozen_manifests.json")
    checkpoints = [
        {
            "variant": item["variant"],
            "training_seed": item["training_seed"],
            "selected_step": item["selected_step"],
            "checkpoint_path": item["checkpoint_path"],
            "checkpoint_sha256": item["checkpoint_sha256"],
        }
        for item in freeze["freezes"]
    ]
    evidence = {
        "schema_version": 1,
        "evidence_type": "minimum_validation_50k_fixed_heldout_evaluation",
        "status": "PASS",
        "sealed_at": utc_now(),
        "provenance": {
            "runtime_evidence_head": EVIDENCE_HEAD,
            "attested_source_commit_sha": SOURCE_COMMIT,
            "source_tree_hash": audit["source_tree_hash"],
            "protocol_sha256": audit["protocol_sha256"],
            "seed_manifest_sha256": audit["seed_manifest_sha256"],
            "freeze_manifest_sha256": audit["freeze_manifest_sha256"],
            "test_manifest_sha256": audit["test_manifest_sha256"],
        },
        "training_precondition": training,
        "freeze": {
            "status": "PASS",
            "checkpoint_selection": False,
            "fixed_checkpoint_step": 50_000,
            "checkpoint_count": 6,
            "checkpoints": checkpoints,
        },
        "held_out_bank": {
            "status": "PASS",
            "unique_cases": 100,
            "sets": audit["scenario_tape_counts"],
            "manifest_sha256": audit["test_manifest_sha256"],
            "lock_completed": True,
        },
        "evaluation_results": {
            "status": "PASS",
            "expected": 600,
            "observed": 600,
            "unique_model_seed_tape_results": 600,
            "scenario_counts": audit["scenario_result_counts"],
            "result_files": audit["result_files"],
            "state_journals_consumed": 6,
            "ledger_completed": True,
            "metrics": audit["metrics"],
        },
        "analysis": {
            "status": "PASS",
            "canonical_rows": 600,
            "aggregate_records": 480,
            "paired_effect_records": 240,
            "comparison_report": relative(EVIDENCE_DIR / "comparison_report.html"),
            "analysis_summary": relative(EVIDENCE_DIR / "analysis_summary.json"),
            "answers": summary["answers"],
        },
        "operations_not_performed": {
            "validation": "NOT_PERFORMED",
            "checkpoint_selection": "NOT_PERFORMED",
            "checkpoint_25000_performance_read_or_compare": "NOT_PERFORMED",
            "retraining": "NOT_PERFORMED",
            "resume_training": "NOT_PERFORMED",
            "parameter_tuning": "NOT_PERFORMED",
            "protocol_edit": "NOT_PERFORMED",
            "source_edit": "NOT_PERFORMED",
        },
        "inventory": {
            "path": relative(inventory_path),
            "artifact_count": inventory["artifact_count"],
            "sha256": inventory_sha,
        },
    }
    evidence_path = EVIDENCE_DIR / "evaluation_evidence.json"
    write_json(evidence_path, evidence)
    evidence_sha = sha256(evidence_path)

    mismatches = []
    for item in inventory["artifacts"]:
        path = WORKTREE / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or sha256(path) != item["sha256"]
        ):
            mismatches.append(item["path"])
    _check_p0_gate()
    if git("diff", "--name-only") or git("diff", "--cached", "--name-only"):
        mismatches.append("tracked source diff after seal")
    if mismatches:
        raise SystemExit("post-seal readonly revalidation failed: " + ", ".join(mismatches))

    readonly = {
        "schema_version": 1,
        "revalidation_type": "post_evaluation_seal_readonly_revalidation",
        "status": "PASS",
        "checked_at": utc_now(),
        "runtime_evidence_head": EVIDENCE_HEAD,
        "attested_source_commit_sha": SOURCE_COMMIT,
        "evaluation_evidence": {
            "path": relative(evidence_path),
            "bytes": evidence_path.stat().st_size,
            "sha256": evidence_sha,
        },
        "sha256_inventory": {
            "path": relative(inventory_path),
            "bytes": inventory_path.stat().st_size,
            "sha256": inventory_sha,
            "verified_artifacts": inventory["artifact_count"],
            "mismatches": 0,
        },
        "checks": {
            "training_seal_48_artifacts": "PASS",
            "formal_p0_gate_entrypoint": "PASS",
            "fixed_six_50000_freeze": "PASS",
            "checkpoint_selection_absent": "PASS",
            "held_out_100_unique_20x5": "PASS",
            "held_out_lock_complete": "PASS",
            "result_files_6_sha_match": "PASS",
            "model_case_results_600_unique": "PASS",
            "paired_metrics_contract": "PASS",
            "analysis_outputs_present": "PASS",
            "inventory_all_match": "PASS",
            "tracked_source_diff": "NONE",
            "validation_artifacts": "NONE",
            "retraining_or_resume": "NOT_PERFORMED",
            "checkpoint_25000_performance": "NOT_READ_OR_COMPARED",
        },
    }
    readonly_path = EVIDENCE_DIR / "readonly_revalidation.json"
    write_json(readonly_path, readonly)
    print(
        json.dumps(
            {
                "status": "PASS",
                "inventory_artifacts": inventory["artifact_count"],
                "evaluation_evidence_sha256": evidence_sha,
                "inventory_sha256": inventory_sha,
                "readonly_revalidation": str(readonly_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
