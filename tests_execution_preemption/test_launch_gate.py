from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from execution_preemption.gate import (
    GATE_NAME,
    GATE_RELATIVE_PATH,
    LaunchGateError,
    REMOTE_SOURCE_REF,
    allowed_evidence_path,
    git_head,
    git_tree,
    is_protected_source_path,
    source_hash_inventory,
    validate_runtime_against_gate,
)


def _git(root: Path, *args: str) -> str:
    command = ["git"]
    if args and args[0] == "commit":
        command.extend(["-c", "commit.gpgsign=false"])
    command.extend(args)
    return subprocess.run(
        command,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


class LaunchGatePathTests(unittest.TestCase):
    def test_protected_paths_cover_new_stack_not_legacy_gate(self) -> None:
        self.assertTrue(is_protected_source_path("execution_preemption/runtime.py"))
        self.assertTrue(is_protected_source_path("tests_execution_preemption/test_launch_gate.py"))
        self.assertTrue(is_protected_source_path(
            "experiments/dynamic_preemption/dev_v1/manifest.json"
        ))
        self.assertTrue(is_protected_source_path("scripts/build_execution_launch_gate.py"))
        self.assertFalse(is_protected_source_path("handoff/P0_GATE.json"))
        self.assertFalse(is_protected_source_path(
            "ppo_allocation/results/random_event/minimum_validation_old/checkpoint.pt"
        ))

    def test_evidence_whitelist_is_exact(self) -> None:
        self.assertTrue(allowed_evidence_path(GATE_RELATIVE_PATH))
        self.assertFalse(allowed_evidence_path("handoff/P0_GATE.json"))
        self.assertFalse(allowed_evidence_path(
            "experiments/dynamic_preemption/evidence_v1/model.pt"
        ))
        self.assertFalse(allowed_evidence_path(
            "experiments/dynamic_preemption/evidence_v1/extra.json"
        ))


class LaunchGateRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        _git(self.root, "init", "-q")
        _git(self.root, "config", "user.email", "gate-test@example.invalid")
        _git(self.root, "config", "user.name", "Gate Test")
        (self.root / "execution_preemption").mkdir()
        (self.root / "execution_preemption" / "runtime.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        (self.root / "README.md").write_text("source\n", encoding="utf-8")
        _git(self.root, "add", ".")
        _git(self.root, "commit", "-q", "-m", "source")
        self.source = git_head(self.root)
        self.gate = {
            "schema_version": 1,
            "gate_name": GATE_NAME,
            "generated_by": "scripts/build_execution_launch_gate.py",
            "training_allowed": True,
            "violations": [],
            "formal_training_started": False,
            "validation_started": False,
            "freeze_started": False,
            "test_started": False,
            "hidden_evaluation_started": False,
            "checks": {"tests": {"status": "PASS"}},
            "attested_source_commit_sha": self.source,
            "attested_source_tree_sha": git_tree(self.root, self.source),
            "remote_source_ref": REMOTE_SOURCE_REF,
            "remote_source_commit_sha": self.source,
            "protected_source_sha256": source_hash_inventory(self.root, self.source),
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_gate(self) -> Path:
        path = self.root / GATE_RELATIVE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.gate, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def test_source_head_passes_with_only_untracked_gate(self) -> None:
        self._write_gate()
        result = validate_runtime_against_gate(self.gate, root=self.root)
        self.assertEqual(result.runtime_mode, "SOURCE_HEAD")
        self.assertFalse(result.evidence_only_descendant)
        self.assertFalse(result.worktree_clean)
        with self.assertRaisesRegex(LaunchGateError, "must be clean"):
            validate_runtime_against_gate(
                self.gate, root=self.root, require_fully_clean=True
            )

    def test_clean_evidence_only_descendant_passes(self) -> None:
        self._write_gate()
        _git(self.root, "add", GATE_RELATIVE_PATH)
        _git(self.root, "commit", "-q", "-m", "evidence")
        result = validate_runtime_against_gate(
            self.gate, root=self.root, require_fully_clean=True
        )
        self.assertEqual(result.runtime_mode, "EVIDENCE_DESCENDANT")
        self.assertTrue(result.evidence_only_descendant)
        self.assertTrue(result.worktree_clean)
        self.assertEqual(result.allowed_evidence_diff, (GATE_RELATIVE_PATH,))

    def test_protected_disk_drift_fails_closed(self) -> None:
        self._write_gate()
        (self.root / "execution_preemption" / "runtime.py").write_text(
            "VALUE = 2\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(LaunchGateError, "worktree changes"):
            validate_runtime_against_gate(self.gate, root=self.root)

    def test_non_evidence_descendant_fails_closed(self) -> None:
        self._write_gate()
        _git(self.root, "add", GATE_RELATIVE_PATH)
        _git(self.root, "commit", "-q", "-m", "evidence")
        (self.root / "notes.txt").write_text("not evidence\n", encoding="utf-8")
        _git(self.root, "add", "notes.txt")
        _git(self.root, "commit", "-q", "-m", "extra")
        with self.assertRaisesRegex(LaunchGateError, "non-evidence descendant"):
            validate_runtime_against_gate(self.gate, root=self.root)

    def test_gate_tampering_fails_closed(self) -> None:
        self.gate["training_allowed"] = False
        with self.assertRaisesRegex(LaunchGateError, "not allowed"):
            validate_runtime_against_gate(self.gate, root=self.root)


if __name__ == "__main__":
    unittest.main()
