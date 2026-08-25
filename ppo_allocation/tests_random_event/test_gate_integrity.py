from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from random_event import gate_integrity


class GateIntegrityTests(unittest.TestCase):
    def test_source_head_and_valid_evidence_descendant_pass(self):
        source = "a" * 40
        self.assertTrue(gate_integrity.is_evidence_only_descendant(Path("."), source, source))
        with patch.object(gate_integrity, "is_ancestor", return_value=True), patch.object(
            gate_integrity,
            "source_to_runtime_changed_paths",
            return_value=["handoff/P0_GATE.json"],
        ):
            self.assertTrue(
                gate_integrity.is_evidence_only_descendant(Path("."), source, "b" * 40)
            )

    def test_invalid_ancestry_fails(self):
        with patch.object(gate_integrity, "is_ancestor", return_value=False):
            self.assertFalse(
                gate_integrity.is_evidence_only_descendant(
                    Path("."), "a" * 40, "b" * 40
                )
            )

    def test_protected_paths_fail(self):
        source = "a" * 40
        for path in (
            "source.py",
            "configs/protocol.json",
            "tests/test_x.py",
            "scripts/run.py",
        ):
            with self.subTest(path=path), patch.object(
                gate_integrity, "is_ancestor", return_value=True
            ), patch.object(
                gate_integrity,
                "source_to_runtime_changed_paths",
                return_value=[path],
            ):
                self.assertFalse(
                    gate_integrity.is_evidence_only_descendant(
                        Path("."), source, "b" * 40
                    )
                )

    def test_whitelist_and_smoke_provenance_are_strict(self):
        source = "a" * 40
        self.assertTrue(gate_integrity.is_allowed_evidence_path("handoff/P0_GATE.json", source))
        self.assertTrue(
            gate_integrity.is_allowed_evidence_path(
                "ppo_allocation/results/random_event/"
                "minimum_validation_p0_smoke_aaaaaaaa/smoke_summary.json",
                source,
            )
        )
        self.assertFalse(
            gate_integrity.is_allowed_evidence_path(
                "ppo_allocation/results/random_event/smoke_old/smoke_summary.json",
                source,
            )
        )
        self.assertTrue(gate_integrity.smoke_metadata_matches_attested({"git_commit": source}, source))
        self.assertFalse(
            gate_integrity.smoke_metadata_matches_attested({"git_commit": "b" * 40}, source)
        )

    def test_disk_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.py"
            path.write_bytes(b"changed\n")
            expected = hashlib.sha256(b"source\n").hexdigest()
            with patch.object(gate_integrity, "git_blob_sha256", return_value=expected):
                with self.assertRaises(ValueError):
                    gate_integrity.verify_protected_tree_against_attested_source(
                        root, "a" * 40, {"source.py": expected}
                    )

    def test_source_tree_hash_is_stable_and_ordered(self):
        hashes = {"b.py": "2" * 64, "a.py": "1" * 64}
        expected = hashlib.sha256(b"a.py:" + b"1" * 64 + b"\nb.py:" + b"2" * 64 + b"\n").hexdigest()
        self.assertEqual(gate_integrity.source_tree_hash(hashes), expected)


if __name__ == "__main__":
    unittest.main()
