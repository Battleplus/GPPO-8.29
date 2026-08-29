"""Build deterministic Gym/Torch rollout smoke evidence without training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium  # noqa: E402
import numpy  # noqa: E402
import torch  # noqa: E402

from execution_preemption.framework import flat_to_torch, hetero_to_torch  # noqa: E402
from execution_preemption.gym_env import ExecutionPreemptionGymEnv  # noqa: E402
from execution_preemption.policy_models import ExecutionGPPOAdaptive, ExecutionPPOMLP  # noqa: E402


TAPE_ROOT = ROOT / "experiments" / "dynamic_preemption" / "dev_v1" / "tapes"
DEFAULT_OUTPUT = (
    ROOT / "experiments" / "dynamic_preemption" / "dev_v1"
    / "framework_rollout_smoke.json"
)


def _load_tape(scenario: str) -> dict:
    path = next((TAPE_ROOT / scenario).glob(f"{scenario}-00-*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _pyg_health() -> dict[str, object]:
    try:
        version = importlib.metadata.version("torch-geometric")
    except importlib.metadata.PackageNotFoundError:
        return {
            "installed": False,
            "version": None,
            "native_import_pass": False,
            "classification": "NOT_INSTALLED_OPTIONAL",
            "required_for_current_gppo": False,
        }
    command = [
        sys.executable,
        "-c",
        "from torch_geometric.data import HeteroData; print('PASS')",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    native_pass = result.returncode == 0 and "PASS" in result.stdout
    combined = (result.stdout + result.stderr).lower()
    if native_pass:
        classification = "PASS_OPTIONAL"
    elif "numpy.dtype size changed" in combined or "binary incompatibility" in combined:
        classification = "HOST_NUMPY_PANDAS_BINARY_INCOMPATIBILITY_OPTIONAL"
    else:
        classification = "HOST_IMPORT_FAILURE_OPTIONAL"
    return {
        "installed": True,
        "version": version,
        "native_import_pass": native_pass,
        "classification": classification,
        "required_for_current_gppo": False,
    }


def _model(policy: str):
    torch.manual_seed(20260829)
    if policy == "ppo_mlp_framework_smoke_v1":
        return ExecutionPPOMLP(hidden_dim=16).eval()
    if policy == "gppo_adaptive_framework_smoke_v1":
        return ExecutionGPPOAdaptive(hidden_dim=16, layers=1).eval()
    raise ValueError(policy)


def _run(policy: str, scenario: str) -> dict[str, object]:
    tape = _load_tape(scenario)
    env = ExecutionPreemptionGymEnv(tape, allocator_id=policy)
    _, info = env.reset(seed=tape["case_seed"])
    model = _model(policy)
    actions: list[int] = []
    rewards: list[float] = []
    mask_violations = 0
    terminated = False
    while not terminated:
        if policy.startswith("ppo_"):
            action, _, _, diagnostics = model.act(flat_to_torch(info["flat_observation"]))
        else:
            action, _, _, diagnostics = model.act(hetero_to_torch(info["hetero_observation"]))
        if not bool(env.action_masks()[action]):
            mask_violations += 1
            raise RuntimeError("framework policy selected a masked action")
        _, reward, terminated, truncated, info = env.step(action)
        if truncated:
            raise RuntimeError("framework smoke unexpectedly truncated")
        actions.append(action)
        rewards.append(float(reward))
        if not 0.0 <= diagnostics["pre_mask_invalid_probability"] <= 1.0:
            raise RuntimeError("invalid probability diagnostic")
    metrics = info["episode_metrics"]
    return {
        "policy_id": policy,
        "scenario_id": scenario,
        "action_count": len(actions),
        "actions": actions,
        "reward_sequence": rewards,
        "reward_sum": float(sum(rewards)),
        "mask_violations": mask_violations,
        "final_runtime_sha256": info["live_runtime_sha256"],
        "graph_version": env.runtime.graph_version,
        "accepted_decision_count": metrics["accepted_decision_count"],
        "resource_conflicts": metrics["resource_conflicts"],
        "stale_command_resurrections": metrics["stale_command_resurrections"],
        "energy_safety_violations": metrics["energy_safety_violations"],
        "status": "PASS",
    }


def build_report() -> dict[str, object]:
    torch.use_deterministic_algorithms(True)
    policies = (
        "ppo_mlp_framework_smoke_v1",
        "gppo_adaptive_framework_smoke_v1",
    )
    scenarios = ("execution_uav_destroyed", "simultaneous_p1")
    runs = [_run(policy, scenario) for policy in policies for scenario in scenarios]
    parameter_counts = {
        policy: sum(parameter.numel() for parameter in _model(policy).parameters())
        for policy in policies
    }
    return {
        "schema_version": 1,
        "status": "PASS",
        "classification": "framework_rollout_smoke_not_training_or_model_evidence",
        "python_version": ".".join(str(item) for item in sys.version_info[:3]),
        "torch_version": torch.__version__,
        "gymnasium_version": gymnasium.__version__,
        "numpy_version": numpy.__version__,
        "pyg_health": _pyg_health(),
        "gppo_backend": "repository_native_pytorch_relation_aware",
        "torch_tensor_conversion": "PASS",
        "gym_reset_step_contract": "PASS",
        "model_parameter_counts": parameter_counts,
        "run_count": len(runs),
        "runs": runs,
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "model_weights_loaded": False,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "training_allowed": False,
        "training_started": False,
        "validation_started": False,
        "freeze_started": False,
        "test_started": False,
        "hidden_evaluation_started": False,
    }


def write_report(path: Path, report: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = write_report(args.output.resolve(), build_report())
    print(json.dumps({"status": "PASS", "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
