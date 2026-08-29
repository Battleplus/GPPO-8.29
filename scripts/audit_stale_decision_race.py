"""Audit an event arriving between inference and action submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ppo_allocation.random_event.environment import (  # noqa: E402
    ActionSubmission,
    RandomEventAllocationEnv,
)
from ppo_allocation.random_event.events import EventTape  # noqa: E402
from ppo_allocation.random_event.models import FairPPOMLP, GraphActorCritic  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="ppo_allocation directory containing the frozen manifest and six 50k checkpoints",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=_REPO_ROOT / "experiments/extreme_scenarios/results_20260827",
    )
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    frozen_path = checkpoint_root / "results/random_event/minimum_validation_50k_2afa8ec/preliminary/frozen_manifests.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    tape_path = result_root / "tapes/atomic_triple_shock/atomic_triple_shock-C00.json"
    tape = EventTape.from_json(tape_path.read_bytes())

    rows = []
    for family in ("PPO-MLP", "GPPO-Adaptive"):
        item = next(
            record for record in frozen["freezes"]
            if record["variant"] == family and int(record["training_seed"]) == 1101
        )
        checkpoint = checkpoint_root / item["checkpoint_path"]
        if sha256_file(checkpoint) != item["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint SHA mismatch: {checkpoint}")
        if family == "PPO-MLP":
            model, _ = FairPPOMLP.load(checkpoint, map_location="cpu")
        else:
            model, _ = GraphActorCritic.load(checkpoint, map_location="cpu")
        model.eval()

        env = RandomEventAllocationEnv(
            initial_seed=tape.initial_seed,
            event_seed=tape.event_seed,
            mode=tape.mode,
            events_per_episode=len(tape.events),
            event_tape=tape,
            max_decisions=150,
        )
        env.reset(seed=tape.initial_seed)
        stale_context = env.begin_decision()
        stale_action, _, _, _ = model.act(stale_context.graph, deterministic=True)
        before = {
            "time": env.current_time,
            "graph_version": env.graph_version,
            "decision_version": env.decision_version,
            "decision_step": env.decision_step,
            "event_index": env.next_event_index,
        }
        next_observed = tape.events[3].observed_at
        env.advance_time(next_observed - env.current_time)
        after_arrival = {
            "time": env.current_time,
            "graph_version": env.graph_version,
            "decision_version": env.decision_version,
            "decision_step": env.decision_step,
            "event_index": env.next_event_index,
        }
        _, reward, terminated, truncated, stale_info = env.submit_action(
            ActionSubmission.from_decision(stale_action, stale_context)
        )
        decision_step_after_stale = env.decision_step
        fresh_context = env.begin_decision()
        fresh_action, _, _, _ = model.act(fresh_context.graph, deterministic=True)
        _, fresh_reward, _, _, fresh_info = env.submit_action(
            ActionSubmission.from_decision(fresh_action, fresh_context)
        )
        row = {
            "family": family,
            "training_seed": 1101,
            "checkpoint_sha256": item["checkpoint_sha256"],
            "before_inflight_event": before,
            "after_inflight_event": after_arrival,
            "stale_action": stale_action,
            "stale_submission_rejected": bool(stale_info.get("stale_decision")),
            "stale_reward": reward,
            "stale_terminated": terminated,
            "stale_truncated": truncated,
            "decision_step_after_stale": decision_step_after_stale,
            "stale_consumed_decision_step": decision_step_after_stale != before["decision_step"],
            "fresh_action": fresh_action,
            "fresh_submission_rejected": bool(fresh_info.get("stale_decision")),
            "fresh_reward": fresh_reward,
            "final_decision_step": env.decision_step,
        }
        # Exactly one step may be consumed: the fresh action, never the stale one.
        row["pass"] = bool(
            row["stale_submission_rejected"]
            and reward == 0.0
            and not row["fresh_submission_rejected"]
            and env.decision_step == before["decision_step"] + 1
            and after_arrival["graph_version"] > before["graph_version"]
        )
        rows.append(row)
        env.close()

    payload = {
        "schema_version": 1,
        "classification": "system_invariant_audit_not_model_effect_evaluation",
        "scenario": "event_arrives_between_inference_and_submission",
        "status": "PASS" if all(row["pass"] for row in rows) else "FAIL",
        "rows": rows,
    }
    output = result_root / "stale_decision_race_audit.json"
    output.write_bytes(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
