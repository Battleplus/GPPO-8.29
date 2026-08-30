#!/usr/bin/env python3
"""Run exactly one fresh formal Execution-Preemption V1 training worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_preemption.contract import LEARNED_METHODS, TRAINING_SCALES, TRAINING_SEEDS  # noqa: E402
from execution_preemption.training import (  # noqa: E402
    TrainingRunConfig,
    formal_run_relative_path,
    train_run,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=LEARNED_METHODS)
    parser.add_argument("--seed", required=True, type=int, choices=TRAINING_SEEDS)
    parser.add_argument("--uav-count", required=True, type=int, choices=TRAINING_SCALES)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    config = TrainingRunConfig(
        method_id=args.method,
        policy_seed=args.seed,
        uav_count=args.uav_count,
    )
    output = args.output_root.resolve() / formal_run_relative_path(
        args.method, args.seed, args.uav_count
    )
    report = train_run(config, output)
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "method_id": report["method_id"],
        "policy_seed": report["policy_seed"],
        "uav_count": report["uav_count"],
        "accepted_decision_steps": report["accepted_decision_steps"],
        "checkpoint_steps": report["checkpoint_steps"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
