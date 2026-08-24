"""Run exactly one fresh minimum-validation model×seed worker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PPO_ROOT = ROOT / "ppo_allocation"
if str(PPO_ROOT) not in sys.path:
    sys.path.insert(0, str(PPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from random_event.phase_j import (  # noqa: E402
    PreliminaryProtocol,
    _check_p0_gate_strict,
    _developer_attestation,
    train_single_run,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("seed", type=int)
    parser.add_argument("run_dir")
    parser.add_argument("--developer", action="store_true")
    parser.add_argument("--budget", type=int, default=50_000)
    parser.add_argument("--checkpoint-interval", type=int, default=25_000)
    args = parser.parse_args(argv)
    protocol = PreliminaryProtocol(
        budget=args.budget,
        checkpoint_interval=args.checkpoint_interval,
    )
    gate = _developer_attestation() if args.developer else _check_p0_gate_strict()
    result = train_single_run(
        Path(args.run_dir),
        variant=args.variant,
        seed=args.seed,
        protocol=protocol,
        gate=gate,
    )
    (Path(args.run_dir) / "run_result.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
