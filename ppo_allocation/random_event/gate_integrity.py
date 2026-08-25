from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Mapping

SMOKE_NAMESPACE_PREFIX = "ppo_allocation/results/random_event/minimum_validation_p0_smoke_"


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def git_blob_sha256(root: Path, commit: str, relative: str) -> str:
    data = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=root)
    return hashlib.sha256(data).hexdigest()


def is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def source_to_runtime_changed_paths(root: Path, attested_source: str, runtime_head: str) -> list[str]:
    output = subprocess.check_output(
        ["git", "diff", "--name-only", f"{attested_source}..{runtime_head}"],
        cwd=root,
        text=True,
    )
    return [line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()]


def smoke_namespace(attested_source: str) -> str:
    return f"{SMOKE_NAMESPACE_PREFIX}{attested_source[:8]}"


def is_allowed_evidence_path(path: str, attested_source: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    if normalized == "handoff/P0_GATE.json":
        return True
    return normalized.startswith(smoke_namespace(attested_source) + "/") and normalized.endswith(".json")


def is_evidence_only_descendant(root: Path, attested_source: str, runtime_head: str) -> bool:
    if runtime_head == attested_source:
        return True
    if not is_ancestor(root, attested_source, runtime_head):
        return False
    changed = source_to_runtime_changed_paths(root, attested_source, runtime_head)
    return bool(changed) and all(is_allowed_evidence_path(path, attested_source) for path in changed)


def verify_protected_tree_against_attested_source(
    root: Path,
    attested_source: str,
    expected_hashes: Mapping[str, str],
) -> None:
    if not expected_hashes:
        raise ValueError("missing protected source attestation hashes")
    for relative, expected in expected_hashes.items():
        committed = git_blob_sha256(root, attested_source, relative)
        disk_path = root / relative
        if not disk_path.is_file():
            raise ValueError(f"protected file missing from disk: {relative}")
        disk = hashlib.sha256(disk_path.read_bytes()).hexdigest()
        if committed != expected or disk != expected:
            raise ValueError(f"protected hash mismatch: {relative}")


def source_tree_hash(expected_hashes: Mapping[str, str]) -> str:
    payload = "".join(
        f"{relative}:{digest}\n"
        for relative, digest in sorted(expected_hashes.items())
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_runtime_attestation(
    root: Path,
    attested_source: str,
    runtime_head: str,
    expected_hashes: Mapping[str, str],
) -> list[str]:
    if not is_evidence_only_descendant(root, attested_source, runtime_head):
        raise ValueError("runtime HEAD is not a valid evidence-only descendant")
    changed = source_to_runtime_changed_paths(root, attested_source, runtime_head)
    verify_protected_tree_against_attested_source(root, attested_source, expected_hashes)
    return changed


def smoke_metadata_matches_attested(metadata: Mapping[str, object], attested_source: str) -> bool:
    return metadata.get("git_commit") == attested_source
