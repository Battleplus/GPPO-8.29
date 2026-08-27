"""Build the SHA-256 manifest for the Git evidence-only archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EVIDENCE_DIR = Path(__file__).resolve().parent
CAMPAIGN_DIR = EVIDENCE_DIR.parent
PRELIMINARY_DIR = CAMPAIGN_DIR / "preliminary"
WORKTREE = Path(__file__).resolve().parents[5]
OUTPUT = EVIDENCE_DIR / "archive_sha256_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(WORKTREE).as_posix()


def main() -> int:
    paths: set[Path] = set()
    paths.update((CAMPAIGN_DIR / "training_evidence").glob("*.json"))
    paths.update(
        PRELIMINARY_DIR / name
        for name in (
            "checkpoint_index.json",
            "training_summary.json",
            "frozen_manifests.json",
            "formal_test_bank_lock.json",
            "test_ledger.json",
        )
    )
    paths.update((PRELIMINARY_DIR / "freezes").glob("*.json"))
    paths.update((PRELIMINARY_DIR / "tapes/preliminary_test_protocol").rglob("*.json"))
    paths.update((PRELIMINARY_DIR / "test_results").glob("*.json"))
    paths.update((PRELIMINARY_DIR / "test_state").glob("*.json"))
    paths.update(
        path
        for path in EVIDENCE_DIR.iterdir()
        if path.is_file() and path != OUTPUT and path.suffix.lower() != ".pyc"
    )
    ordered = sorted(paths, key=relative)
    lock = json.loads(
        (PRELIMINARY_DIR / "formal_test_bank_lock.json").read_text(encoding="utf-8")
    )
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "created_at": lock["completed_at"],
        "purpose": "Git evidence-only archive without checkpoint binaries",
        "parent_evidence_head": "2afa8ec1cb481deb57645dbd30240d90d32d2233",
        "attested_source_commit_sha": "32974ec85be71e192b12cae85d00eb877d5fe07d",
        "algorithm": "SHA-256",
        "artifact_count": len(ordered),
        "artifacts": [
            {
                "path": relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in ordered
        ],
        "explicit_exclusions": [
            "all *.pt checkpoint binaries",
            "preliminary/runs training outputs",
            "training_control",
            "validation artifacts",
            "Python cache files",
        ],
        "checkpoint_policy": (
            "The six 50k SHA-256 values are cross-bound by the sealed training evidence, "
            "checkpoint index, formal Freeze, Test ledger and result files; binaries remain external."
        ),
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "artifact_count": len(ordered), "output": str(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
