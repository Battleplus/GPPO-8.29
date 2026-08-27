"""Verify and byte-rebuild the committed evaluation evidence in a clean worktree."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


EVIDENCE_DIR = Path(__file__).resolve().parent
CAMPAIGN_DIR = EVIDENCE_DIR.parent
PRELIMINARY_DIR = CAMPAIGN_DIR / "preliminary"
WORKTREE = Path(__file__).resolve().parents[5]
PPO_DIR = WORKTREE / "ppo_allocation"
ARCHIVE_MANIFEST = EVIDENCE_DIR / "archive_sha256_manifest.json"
BASE_EVIDENCE_HEAD = "2afa8ec1cb481deb57645dbd30240d90d32d2233"
SOURCE_COMMIT = "32974ec85be71e192b12cae85d00eb877d5fe07d"
EXPECTED_METRICS = (
    "event_success_rate",
    "legal_coverage_rate",
    "recovery_delay",
    "cumulative_uncovered_time",
    "normalized_distance",
    "load_gap",
    "switch_count",
    "episode_return",
    "communication_bytes",
    "inference_latency_ms",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(WORKTREE), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def run_checked(args: list[str], *, cwd: Path = WORKTREE) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def allowed_archive_path(path: str) -> bool:
    prefix = "ppo_allocation/results/random_event/minimum_validation_50k_2afa8ec/"
    if not path.startswith(prefix):
        return False
    relative = path[len(prefix):]
    if relative.startswith("evaluation_evidence/"):
        return True
    if relative.startswith("training_evidence/") and relative.endswith(".json"):
        return True
    exact = {
        "preliminary/checkpoint_index.json",
        "preliminary/training_summary.json",
        "preliminary/frozen_manifests.json",
        "preliminary/formal_test_bank_lock.json",
        "preliminary/test_ledger.json",
    }
    if relative in exact:
        return True
    return any(
        relative.startswith(folder) and relative.endswith(".json")
        for folder in (
            "preliminary/freezes/",
            "preliminary/tapes/preliminary_test_protocol/",
            "preliminary/test_results/",
            "preliminary/test_state/",
        )
    )


def verify_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for item in manifest.get("artifacts", []):
        path = WORKTREE / item["path"]
        if not path.is_file():
            errors.append(f"missing:{item['path']}")
        elif path.stat().st_size != int(item["bytes"]):
            errors.append(f"bytes:{item['path']}")
        elif sha256(path) != item["sha256"]:
            errors.append(f"sha256:{item['path']}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate-reference-worktree",
        type=Path,
        required=True,
        help="clean worktree checked out at the formal evidence HEAD, used to execute _check_p0_gate()",
    )
    args = parser.parse_args(argv)
    gate_reference = args.gate_reference_worktree.resolve()
    errors: list[str] = []
    head = git("rev-parse", "HEAD")
    ancestry = subprocess.run(
        ["git", "-C", str(WORKTREE), "merge-base", "--is-ancestor", BASE_EVIDENCE_HEAD, head],
        check=False,
    ).returncode
    if ancestry != 0:
        errors.append(f"base evidence HEAD is not an ancestor:{BASE_EVIDENCE_HEAD}")
    if git("status", "--porcelain"):
        errors.append("worktree not clean before verification")

    changed = set(git("diff", "--name-only", f"{BASE_EVIDENCE_HEAD}..{head}").splitlines())
    if not changed or any(not allowed_archive_path(path) for path in changed):
        errors.append("commit contains a path outside the evidence-only allowlist")
    if any(path.lower().endswith((".pt", ".pth", ".ckpt")) for path in changed):
        errors.append("checkpoint binary committed")
    if any("/preliminary/runs/" in path or path.startswith("training_control/") for path in changed):
        errors.append("training output/control committed")

    manifest = read_json(ARCHIVE_MANIFEST)
    manifest_paths = {str(item["path"]) for item in manifest.get("artifacts", [])}
    archive_manifest_relative = ARCHIVE_MANIFEST.relative_to(WORKTREE).as_posix()
    if changed != manifest_paths | {archive_manifest_relative}:
        errors.append("commit path set does not equal archive manifest path set")
    errors.extend(verify_manifest(manifest))

    training = read_json(CAMPAIGN_DIR / "training_evidence/training_evidence.json")
    if training.get("status") != "PASS":
        errors.append("training seal status")
    if training.get("provenance", {}).get("runtime_evidence_head") != BASE_EVIDENCE_HEAD:
        errors.append("training evidence HEAD")
    if training.get("provenance", {}).get("attested_source_commit_sha") != SOURCE_COMMIT:
        errors.append("training source provenance")
    expected_summary = {
        "completed_runs": 6,
        "total_steps": 300_000,
        "checkpoint_count": 12,
        "checkpoint_25000_count": 6,
        "checkpoint_50000_count": 6,
    }
    for key, value in expected_summary.items():
        if training.get("summary", {}).get(key) != value:
            errors.append(f"training summary:{key}")

    candidates = {
        (item["variant"], int(item["seed"])): item["sha256"]
        for item in training.get("fixed_50000_evaluation_candidates", [])
    }
    freeze_path = PRELIMINARY_DIR / "frozen_manifests.json"
    freeze = read_json(freeze_path)
    freezes = {
        (item["variant"], int(item["training_seed"])): item
        for item in freeze.get("freezes", [])
    }
    expected_keys = {(variant, seed) for variant in ("PPO-MLP", "GPPO-Adaptive") for seed in (1101, 2202, 3303)}
    if set(candidates) != expected_keys or set(freezes) != expected_keys:
        errors.append("fixed candidate/freeze 2x3 matrix")
    for key in expected_keys:
        item = freezes.get(key, {})
        if int(item.get("selected_step", 0)) != 50_000 or item.get("checkpoint_sha256") != candidates.get(key):
            errors.append(f"fixed 50k SHA provenance:{key}")
    if freeze.get("checkpoint_selection") is not False:
        errors.append("checkpoint selection detected")

    test_manifest_path = PRELIMINARY_DIR / "tapes/preliminary_test_protocol/manifest.json"
    test_manifest = read_json(test_manifest_path)
    entries = test_manifest.get("entries", [])
    entry_by_id = {item["tape_id"]: item for item in entries}
    set_counts = collections.Counter(item["set_name"] for item in entries)
    if len(entries) != 100 or len(entry_by_id) != 100:
        errors.append("held-out bank not 100 unique")
    if set_counts != {f"Test-{name}": 20 for name in ("Single", "Sequential", "Overlap", "Burst", "Unseen")}:
        errors.append("held-out scenario balance")
    manifest_sha = sha256(test_manifest_path)
    lock = read_json(PRELIMINARY_DIR / "formal_test_bank_lock.json")
    ledger = read_json(PRELIMINARY_DIR / "test_ledger.json")
    if lock.get("completed") is not True or ledger.get("completed") is not True:
        errors.append("held-out lock/ledger incomplete")
    if lock.get("test_manifest_sha256") != manifest_sha or ledger.get("test_manifest_sha256") != manifest_sha:
        errors.append("held-out manifest SHA provenance")

    result_count = 0
    composites: set[tuple[str, int, str]] = set()
    for ledger_item in ledger.get("entries", {}).values():
        result_path = PPO_DIR / ledger_item["result_path"]
        if not result_path.is_file() or sha256(result_path) != ledger_item.get("result_sha"):
            errors.append(f"result file SHA:{ledger_item.get('result_path')}")
            continue
        payload = read_json(result_path)
        variant = ledger_item["variant"]
        seed = int(ledger_item["training_seed"])
        if len(payload.get("results", [])) != 100:
            errors.append(f"result count:{variant}/{seed}")
        for item in payload.get("results", []):
            tape_id = item["tape_id"]
            episode = item["episode"]
            if tape_id not in entry_by_id:
                errors.append(f"unknown tape:{variant}/{seed}/{tape_id}")
            if any(metric not in episode for metric in EXPECTED_METRICS):
                errors.append(f"frozen metric missing:{variant}/{seed}/{tape_id}")
            result_count += 1
            composites.add((variant, seed, tape_id))
    if result_count != 600 or len(composites) != 600:
        errors.append(f"600-result completeness:{result_count}/{len(composites)}")

    evidence = read_json(EVIDENCE_DIR / "evaluation_evidence.json")
    if evidence.get("status") != "PASS" or evidence.get("evaluation_results", {}).get("observed") != 600:
        errors.append("evaluation evidence summary")

    if errors:
        raise SystemExit("ARCHIVE PRECHECK FAIL\n" + "\n".join(f"- {error}" for error in errors))

    rebuild_output = run_checked([sys.executable, str(EVIDENCE_DIR / "analyze_evaluation.py"), "--archive-mode"])
    errors.extend(verify_manifest(manifest))
    reference_head = subprocess.run(
        ["git", "-C", str(gate_reference), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    reference_status = subprocess.run(
        ["git", "-C", str(gate_reference), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if reference_head != BASE_EVIDENCE_HEAD:
        errors.append(f"gate reference HEAD mismatch:{reference_head}")
    if reference_status:
        errors.append("gate reference worktree is not clean")
    if errors:
        raise SystemExit("GATE REFERENCE PRECHECK FAIL\n" + "\n".join(f"- {error}" for error in errors))
    gate_output = run_checked(
        [
            sys.executable,
            "-c",
            "from ppo_allocation.random_event.experiment import _check_p0_gate; "
            "_check_p0_gate(); print('FORMAL_CHECK_P0_GATE=PASS')",
        ],
        cwd=gate_reference,
    )
    if git("status", "--porcelain"):
        errors.append("worktree dirty after byte-identical report rebuild")
    if errors:
        raise SystemExit("ARCHIVE POSTCHECK FAIL\n" + "\n".join(f"- {error}" for error in errors))

    result = {
        "schema_version": 1,
        "status": "PASS",
        "evaluation_evidence_commit": head,
        "base_evidence_head": BASE_EVIDENCE_HEAD,
        "attested_source_commit_sha": SOURCE_COMMIT,
        "checks": {
            "clean_before": "PASS",
            "evidence_only_diff": "PASS",
            "checkpoint_binaries_committed": 0,
            "archive_artifacts_sha256": f"{manifest['artifact_count']}/{manifest['artifact_count']} PASS",
            "training_seal": "6/6 runs; 300000 steps; 12/12 checkpoints PASS",
            "freeze": "six fixed 50k checkpoints; no selection PASS",
            "held_out_bank": "100 unique; five scenarios x20 PASS",
            "model_case_results": "600/600 unique PASS",
            "report_rebuild": "byte-identical PASS",
            "formal_p0_gate_at_base_evidence_head": "PASS",
            "archive_head_gate_policy": (
                "not a formal runtime HEAD; immutable gate whitelist intentionally excludes evaluation archive paths"
            ),
            "clean_after": "PASS",
        },
        "test_manifest_sha256": manifest_sha,
        "archive_manifest_sha256": sha256(ARCHIVE_MANIFEST),
        "analysis_rebuild_stdout": rebuild_output,
        "gate_stdout": gate_output,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
