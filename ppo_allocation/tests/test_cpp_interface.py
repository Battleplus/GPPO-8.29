import json
import subprocess
import sys
from pathlib import Path

PPO_DIR = Path(__file__).resolve().parents[1]
if str(PPO_DIR) in sys.path:
    sys.path.remove(str(PPO_DIR))
sys.path.insert(0, str(PPO_DIR))

from reallocation_service import reallocate_from_preallocation


def test_cpp_style_interface_smoke():
    model_path = "ppo_allocation/results/models/run_20260605_210049/maskable_ppo_uav_task_allocation.zip"
    preallocation_path = "ppo_allocation/scenarios/output_template.json"
    output_path = "ppo_allocation/results/tmp_cpp_interface_output.json"

    out = reallocate_from_preallocation(
        model_path=model_path,
        preallocation_path=preallocation_path,
        event={"event_type": "UAV_DAMAGE", "uav_id": 1},
        output_path=output_path,
    )

    assert Path(out).exists()
    payload = json.loads(Path(out).read_text(encoding="utf-8"))
    assert payload["event"]
    assert "region_assignments" in payload
    assert "uav_tasks" in payload


def test_cli_bridge_smoke(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "model_path": "ppo_allocation/results/models/run_20260605_210049/maskable_ppo_uav_task_allocation.zip",
                "preallocation_path": "ppo_allocation/scenarios/output_template.json",
                "event": {"event_type": "UAV_DAMAGE", "uav_id": 1},
                "output_path": str(tmp_path / "bridge_output.json"),
                "deterministic": True,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "ppo_allocation/cpp_bridge.py", "--request-file", str(request_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["success"] is True
    assert Path(payload["output_path"]).exists()
