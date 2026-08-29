"""Build the immutable 10x20 Dynamic-Preemption-Dev paired tape bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution_preemption.tapes import (  # noqa: E402
    CASES_PER_SCENARIO,
    CONTRACT_ID,
    DEV_BANK,
    SCENARIO_CATALOG,
    build_development_bank,
    replay_tape,
    tape_sha256,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    path.write_bytes((data + "\n").encode("utf-8"))


def build(output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to reuse non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tapes = build_development_bank()
    inventory: list[dict[str, object]] = []
    decision_count = 0
    for tape in tapes:
        _, decisions = replay_tape(tape)
        decision_count += len(decisions)
        relative = Path("tapes") / tape["scenario_id"] / f"{tape['tape_id']}.json"
        path = output_dir / relative
        _write_json(path, tape)
        inventory.append({
            "path": relative.as_posix(),
            "sha256_canonical": tape_sha256(tape),
            "sha256_file": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scenario_id": tape["scenario_id"],
            "case_index": tape["case_index"],
            "case_seed": tape["case_seed"],
        })
    catalog = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "bank": DEV_BANK,
        "classification": "development_only_not_held_out",
        "scenario_count": len(SCENARIO_CATALOG),
        "cases_per_scenario": CASES_PER_SCENARIO,
        "case_count": len(tapes),
        "scenarios": list(SCENARIO_CATALOG),
    }
    _write_json(output_dir / "scenario_catalog.json", catalog)
    manifest = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "bank": DEV_BANK,
        "status": "PASS",
        "classification": "development_only_not_held_out",
        "paired": True,
        "scenario_count": len(SCENARIO_CATALOG),
        "tape_count": len(tapes),
        "replayed_tape_count": len(tapes),
        "decision_count": decision_count,
        "invariant_failures": 0,
        "training_started": False,
        "checkpoint_selection": False,
        "inventory": inventory,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "dynamic_preemption" / "dev_v1",
    )
    args = parser.parse_args()
    manifest = build(args.output_dir.resolve())
    print(json.dumps({
        "status": manifest["status"],
        "scenario_count": manifest["scenario_count"],
        "tape_count": manifest["tape_count"],
        "decision_count": manifest["decision_count"],
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
