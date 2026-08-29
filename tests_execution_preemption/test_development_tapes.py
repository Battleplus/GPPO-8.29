from __future__ import annotations

from collections import Counter
import unittest

from execution_preemption.tapes import (
    CASES_PER_SCENARIO,
    SCENARIO_CATALOG,
    build_development_bank,
    build_development_tape,
    replay_tape,
    tape_sha256,
)


class DevelopmentTapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tapes = build_development_bank()

    def test_frozen_cardinality_and_unique_seeds(self) -> None:
        self.assertEqual(len(self.tapes), 200)
        counts = Counter(item["scenario_id"] for item in self.tapes)
        self.assertEqual(set(counts), {item["id"] for item in SCENARIO_CATALOG})
        self.assertTrue(all(count == CASES_PER_SCENARIO for count in counts.values()))
        self.assertEqual(len({item["case_seed"] for item in self.tapes}), 200)

    def test_generation_is_byte_stable(self) -> None:
        first = build_development_tape("urgent_at_40", 0)
        second = build_development_tape("urgent_at_40", 0)
        self.assertEqual(tape_sha256(first), tape_sha256(second))

    def test_every_tape_replays_with_expected_decisions_and_invariants(self) -> None:
        decision_count = 0
        for tape in self.tapes:
            runtime, decisions = replay_tape(tape)
            runtime.validate_invariants()
            decision_count += len(decisions)
        self.assertEqual(decision_count, 280)


if __name__ == "__main__":
    unittest.main()
