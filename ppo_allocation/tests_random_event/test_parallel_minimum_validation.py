from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ppo_allocation.random_event.parallel import (
    FORMAL_BUDGET,
    FORMAL_CHECKPOINTS,
    FORMAL_SEEDS,
    FORMAL_VARIANTS,
    aggregate_progress,
    format_progress,
    validate_worker_isolation,
    worker_specs,
)
from ppo_allocation.random_event.progress import read_progress, write_progress


class ParallelMinimumValidationTests(unittest.TestCase):
    def test_formal_contract(self):
        self.assertEqual(FORMAL_VARIANTS, ("PPO-MLP", "GPPO-NoGate", "GPPO-Adaptive"))
        self.assertEqual(FORMAL_SEEDS, (1101, 2202, 3303))
        self.assertEqual(FORMAL_BUDGET, 50_000)
        self.assertEqual(FORMAL_CHECKPOINTS, (25_000, 50_000))
        self.assertEqual(len(worker_specs(Path("/tmp/campaign"))), 9)

    def test_worker_paths_are_unique(self):
        validate_worker_isolation(worker_specs(Path("/tmp/campaign")))

    def test_aggregate_uses_exact_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = worker_specs(Path(tmp))
            for index, spec in enumerate(specs):
                write_progress(spec.progress_path, {
                    "variant": spec.variant, "seed": spec.seed,
                    "status": "done" if index == 0 else "running",
                    "total_steps": index * 1000, "target_steps": FORMAL_BUDGET,
                    "checkpoint_count": 1 if index == 0 else 0,
                    "steps_per_second": 2.5,
                })
            snapshot = aggregate_progress(specs)
            self.assertEqual(snapshot["total_steps"], sum(i * 1000 for i in range(9)))
            self.assertEqual(snapshot["target_steps"], 9 * FORMAL_BUDGET)
            self.assertEqual(snapshot["completed_runs"], 1)
            self.assertIn("50000", format_progress(snapshot))

    def test_progress_io_is_atomic_and_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            self.assertTrue(write_progress(path, {"total_steps": 17}))
            self.assertEqual(read_progress(path)["total_steps"], 17)

    def test_progress_failure_is_best_effort(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blocked"
            path.mkdir()
            self.assertFalse(write_progress(path, {"total_steps": 1}))

    def test_no_duplicate_worker_paths(self):
        specs = worker_specs(Path("/tmp/campaign"))
        self.assertEqual(len({str(s.stdout_path) for s in specs}), 9)
        self.assertEqual(len({str(s.stderr_path) for s in specs}), 9)
        self.assertEqual(len({str(s.progress_path) for s in specs}), 9)
        self.assertEqual(len({str(s.checkpoint_dir) for s in specs}), 9)


if __name__ == "__main__":
    unittest.main()
