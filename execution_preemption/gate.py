"""Source-bound launch gate for Execution-Preemption V1.

This module deliberately does not reuse the legacy minimum-validation gate.
The new experiment changes state, action, graph and reward semantics and must
therefore carry its own source attestation and evidence-only ancestry rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


GATE_NAME = "EXECUTION_PREEMPTION_V1_LAUNCH_GATE"
GATE_RELATIVE_PATH = (
    "experiments/dynamic_preemption/evidence_v1/"
    "EXECUTION_PREEMPTION_V1_GATE.json"
)
REMOTE_SOURCE_REF = "refs/heads/research/execution-preemption-v1"

PROTECTED_PREFIXES = (
    "execution_preemption/",
    "tests_execution_preemption/",
    "experiments/dynamic_preemption/dev_v1/",
)
PROTECTED_EXACT_PATHS = frozenset({
    "README.md",
    "configs/execution_preemption_v1.json",
    "configs/execution_graph_v1.json",
    "configs/execution_policy_adapter_v1.json",
    "configs/execution_training_contract_v1.json",
    "docs/ALLOCATION_BOUNDARY_V1_ZH.md",
    "docs/EXECUTION_PREEMPTION_CONCLUSION_ZH.md",
    "docs/EXECUTION_PREEMPTION_PROGRESS_ZH.md",
    "docs/EXECUTION_PREEMPTION_V1_PROTOCOL_ZH.md",
    "docs/EXECUTION_BASELINES_V1_ZH.md",
    "docs/EXECUTION_TRAINING_CONTRACT_V1_ZH.md",
    "docs/POLICY_ADAPTER_V1_ZH.md",
})
PROTECTED_SCRIPT_PREFIXES = (
    "scripts/build_execution_",
    "scripts/generate_dynamic_preemption_",
    "scripts/replay_dynamic_preemption_",
    "scripts/run_dynamic_preemption_",
    "scripts/validate_execution_",
)


class LaunchGateError(RuntimeError):
    """Raised when the source-bound launch gate fails closed."""


@dataclass(frozen=True, slots=True)
class LaunchGateValidation:
    attested_source_commit_sha: str
    runtime_head_sha: str
    runtime_mode: str
    source_is_runtime_ancestor: bool
    evidence_only_descendant: bool
    worktree_clean: bool
    protected_diff: tuple[str, ...]
    allowed_evidence_diff: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "PASS",
            "attested_source_commit_sha": self.attested_source_commit_sha,
            "runtime_head_sha": self.runtime_head_sha,
            "runtime_mode": self.runtime_mode,
            "source_is_runtime_ancestor": self.source_is_runtime_ancestor,
            "evidence_only_descendant": self.evidence_only_descendant,
            "worktree_clean": self.worktree_clean,
            "protected_diff": list(self.protected_diff),
            "allowed_evidence_diff": list(self.allowed_evidence_diff),
        }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str, text: bool = True) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=text,
            encoding="utf-8" if text else None,
            errors="replace" if text else None,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LaunchGateError(f"git command failed: {' '.join(args)}") from exc
    return result.stdout.strip() if text else result.stdout


def git_head(root: Path) -> str:
    return str(_git(root, "rev-parse", "HEAD"))


def git_tree(root: Path, commit: str) -> str:
    return str(_git(root, "rev-parse", f"{commit}^{{tree}}"))


def git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        capture_output=True,
    )
    if result.returncode not in (0, 1):
        raise LaunchGateError("git ancestry check failed")
    return result.returncode == 0


def git_changed_paths(root: Path, source: str, runtime: str) -> tuple[str, ...]:
    if source == runtime:
        return ()
    output = str(_git(root, "diff", "--name-only", "--diff-filter=ACDMRTUXB", source, runtime))
    return tuple(sorted(line for line in output.splitlines() if line))


def git_worktree_changes(root: Path) -> tuple[str, ...]:
    output = _git(root, "status", "--porcelain", "-z", "--untracked-files=all", text=False)
    assert isinstance(output, bytes)
    entries = output.decode("utf-8", errors="replace").split("\0")
    paths: list[str] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        if not entry:
            index += 1
            continue
        status = entry[:2]
        path = entry[3:]
        if status[0] in "RC" and index + 1 < len(entries):
            index += 1
            path = entries[index]
        paths.append(path.replace("\\", "/"))
        index += 1
    return tuple(sorted(set(paths)))


def is_protected_source_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return (
        normalized in PROTECTED_EXACT_PATHS
        or normalized.startswith(PROTECTED_PREFIXES)
        or normalized.startswith(PROTECTED_SCRIPT_PREFIXES)
    )


def protected_paths_at_commit(root: Path, commit: str) -> tuple[str, ...]:
    output = str(_git(root, "ls-tree", "-r", "--name-only", commit))
    return tuple(sorted(path for path in output.splitlines() if is_protected_source_path(path)))


def git_blob_sha256(root: Path, commit: str, path: str) -> str:
    payload = _git(root, "show", f"{commit}:{path}", text=False)
    assert isinstance(payload, bytes)
    return sha256_bytes(payload)


def source_hash_inventory(root: Path, commit: str) -> dict[str, str]:
    return {
        path: git_blob_sha256(root, commit, path)
        for path in protected_paths_at_commit(root, commit)
    }


def disk_hash_inventory(root: Path, paths: Sequence[str]) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path in paths:
        target = root / path
        inventory[path] = sha256_file(target) if target.is_file() else "MISSING"
    return inventory


def allowed_evidence_path(path: str) -> bool:
    return path.replace("\\", "/") == GATE_RELATIVE_PATH


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchGateError(message)


def load_gate(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchGateError(f"launch gate is missing or invalid: {path}") from exc
    _expect(isinstance(value, Mapping), "launch gate root must be an object")
    return value


def validate_runtime_against_gate(
    gate: Mapping[str, Any],
    *,
    root: Path,
    require_fully_clean: bool = False,
) -> LaunchGateValidation:
    _expect(gate.get("schema_version") == 1, "launch gate schema drift")
    _expect(gate.get("gate_name") == GATE_NAME, "launch gate name drift")
    _expect(gate.get("generated_by") == "scripts/build_execution_launch_gate.py",
            "launch gate generator drift")
    _expect(gate.get("training_allowed") is True, "training is not allowed")
    _expect(gate.get("violations") == [], "launch gate contains violations")
    _expect(gate.get("formal_training_started") is False,
            "formal training must not have started in launch evidence")
    _expect(gate.get("validation_started") is False, "validation must not have started")
    _expect(gate.get("freeze_started") is False, "freeze must not have started")
    _expect(gate.get("test_started") is False, "test must not have started")
    _expect(gate.get("hidden_evaluation_started") is False,
            "hidden evaluation must not have started")

    checks = gate.get("checks")
    _expect(isinstance(checks, Mapping) and bool(checks), "launch gate checks missing")
    for name, check in checks.items():
        _expect(isinstance(check, Mapping), f"invalid check record: {name}")
        _expect(check.get("status") == "PASS", f"launch gate check is not PASS: {name}")

    source = str(gate.get("attested_source_commit_sha", ""))
    _expect(len(source) == 40, "attested source SHA missing")
    _expect(gate.get("remote_source_ref") == REMOTE_SOURCE_REF, "remote source ref drift")
    _expect(gate.get("remote_source_commit_sha") == source, "remote source binding mismatch")
    _expect(git_tree(root, source) == gate.get("attested_source_tree_sha"),
            "attested source tree mismatch")

    expected_hashes = gate.get("protected_source_sha256")
    _expect(isinstance(expected_hashes, Mapping) and bool(expected_hashes),
            "protected source inventory missing")
    actual_source_hashes = source_hash_inventory(root, source)
    _expect(dict(expected_hashes) == actual_source_hashes,
            "protected committed source hashes do not match attestation")
    runtime = git_head(root)
    ancestor = git_is_ancestor(root, source, runtime)
    _expect(ancestor, "attested source is not an ancestor of runtime HEAD")
    diff = git_changed_paths(root, source, runtime)
    protected_diff = tuple(path for path in diff if is_protected_source_path(path))
    unexpected_diff = tuple(path for path in diff if not allowed_evidence_path(path))
    _expect(not protected_diff, f"protected source diff detected: {protected_diff}")
    _expect(not unexpected_diff, f"non-evidence descendant diff detected: {unexpected_diff}")

    dirty = git_worktree_changes(root)
    dirty_non_evidence = tuple(path for path in dirty if not allowed_evidence_path(path))
    _expect(not dirty_non_evidence, f"non-evidence worktree changes detected: {dirty_non_evidence}")
    if require_fully_clean:
        _expect(not dirty, f"training worktree must be clean: {dirty}")

    runtime_mode = "SOURCE_HEAD" if runtime == source else "EVIDENCE_DESCENDANT"
    evidence_only = runtime != source and bool(diff) and not unexpected_diff
    return LaunchGateValidation(
        attested_source_commit_sha=source,
        runtime_head_sha=runtime,
        runtime_mode=runtime_mode,
        source_is_runtime_ancestor=ancestor,
        evidence_only_descendant=evidence_only,
        worktree_clean=not dirty,
        protected_diff=protected_diff,
        allowed_evidence_diff=diff,
    )


def check_execution_launch_gate(
    *,
    root: Path | None = None,
    gate_path: Path | None = None,
    require_fully_clean: bool = False,
) -> LaunchGateValidation:
    repository = (root or Path(__file__).resolve().parents[1]).resolve()
    path = (gate_path or repository / GATE_RELATIVE_PATH).resolve()
    return validate_runtime_against_gate(
        load_gate(path),
        root=repository,
        require_fully_clean=require_fully_clean,
    )


def _check_execution_launch_gate(*, require_fully_clean: bool = False) -> dict[str, object]:
    """Formal training-entry guard; raises before any optimizer is created."""
    return check_execution_launch_gate(require_fully_clean=require_fully_clean).to_dict()


__all__ = [
    "GATE_NAME",
    "GATE_RELATIVE_PATH",
    "LaunchGateError",
    "LaunchGateValidation",
    "REMOTE_SOURCE_REF",
    "_check_execution_launch_gate",
    "allowed_evidence_path",
    "check_execution_launch_gate",
    "disk_hash_inventory",
    "git_blob_sha256",
    "git_changed_paths",
    "git_head",
    "git_is_ancestor",
    "git_tree",
    "git_worktree_changes",
    "is_protected_source_path",
    "load_gate",
    "protected_paths_at_commit",
    "sha256_file",
    "source_hash_inventory",
    "validate_runtime_against_gate",
]
