"""循环测试：基于初始场景连续注入突发事件 → PPO重分配，共 5 轮。

每轮的初始态势 = 上一轮 PPO 分配后的结果。
输出每轮的 before/after 快照、分配 JSON 和累积对比图。
"""

from pathlib import Path
import json
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from env.uav_env import UAVTaskAllocationEnv
from env.uav import UAV
from env.region import Region
from env.target import Target
from env.event import Event
from config import (
    Weather, SensorType, TaskType, TargetType, EventType,
    ActionCode, NO_UAV, NO_TARGET, NUM_REGIONS,
)
from apply import _inject_random_event, build_action_summary
from preallocation_adapter import adapt
from policy.action_repair import repair_action
from utils.visualization import save_comparison_figure, save_loop_overview_figure, save_loop_animation_html
from utils.logger import save_json

ACTION_NAMES = {
    ActionCode.KEEP: "KEEP", ActionCode.U0: "U0", ActionCode.U1: "U1",
    ActionCode.U2: "U2", ActionCode.U3: "U3", ActionCode.NO_UAV: "NO_UAV",
}

OUTPUT_DIR = Path("results/loop_test8")
MODEL_PATH = "results/models/run_20260605_210049/maskable_ppo_uav_task_allocation.zip"



def _sensor_short(sensor):
    return "EO" if sensor == SensorType.EO else "SAR"


def print_state(env, title: str):
    """打印当前态势摘要。"""
    print(f"\n  [{title}]")
    for rid in range(NUM_REGIONS):
        r = env.regions[rid]
        uid = r.assigned_uav
        w = "Sunny" if r.weather == Weather.SUNNY else "Rainy"
        if uid != NO_UAV and uid in env.uavs and env.uavs[uid].alive:
            u = env.uavs[uid]
            print(f"    R{rid}({w}) ← U{uid}({_sensor_short(u.sensor)})  "
                  f"task={u.task.name}  regions={sorted(u.regions)}")
        else:
            print(f"    R{rid}({w}) ← UNASSIGNED")
    for uid, u in env.uavs.items():
        if u.task == TaskType.TRACK:
            print(f"    U{uid}: TRACK T{u.target_id}")
        elif not u.alive:
            print(f"    U{uid}: DEAD")


def run_loop_test(model_path: str = MODEL_PATH, n_rounds: int = 5):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    env = UAVTaskAllocationEnv(random_event_mode=False)
    rng = np.random.default_rng()

    # 加载初始场景：从预分配输出适配
    prealloc_path = "scenarios/output_template.json"
    prealloc = json.loads(Path(prealloc_path).read_text(encoding="utf-8"))
    scenario = adapt(prealloc)

    env.decision_step = 0
    for rid_str, r in scenario["regions"].items():
        rid = int(rid_str)
        env.regions[rid] = Region(
            rid=rid, center_x=r["center_x"], center_y=r["center_y"],
            weather=Weather(r["weather"]), assigned_uav=r.get("assigned_uav", NO_UAV),
            need_reassign=r.get("need_reassign", False), priority=r.get("priority", 1.0),
        )
    for uid_str, u in scenario["uavs"].items():
        uid = int(uid_str)
        env.uavs[uid] = UAV(
            uid=uid, x=u["x"], y=u["y"], sensor=SensorType(u["sensor"]),
            alive=u.get("alive", True), sensor_failed=u.get("sensor_failed", False),
            task=TaskType(u["task"]), regions=set(u.get("regions", [])),
            target_id=u.get("target_id", NO_TARGET),
        )
    for tid_str, t in scenario.get("targets", {}).items():
        tid = int(tid_str)
        env.targets[tid] = Target(
            tid=tid, target_type=TargetType(t.get("target_type", 1)),
            x=t["x"], y=t["y"], region=t.get("region", 0),
            movable=t.get("movable", False), discovered=t.get("discovered", False),
            tracked=t.get("tracked", False), destroyed=t.get("destroyed", False),
            tracker_id=t.get("tracker_id", NO_TARGET),
        )
    env.current_event = Event(EventType.REGION_VACANCY, [], description="(initialized)")
    env._get_obs()

    print("=" * 60)
    print(f"Initial state loaded from {prealloc_path} (adapted)")
    print(f"  AOI: {scenario.get('scenario_name', 'N/A')}")
    print(f"  description: {scenario.get('description', '')}")
    print_state(env, "初始态势")

    model = MaskablePPO.load(model_path, env=env, device="cpu")

    # 记录全程的初始快照
    initial_snapshot = env.snapshot()

    round_records = []

    for step in range(1, n_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  Round {step}/{n_rounds}")
        print(f"{'='*60}")

        # ----- 1. 记录旧分配 -----
        old_assignments = {rid: r.assigned_uav for rid, r in env.regions.items()}

        # ----- 2. 注入突发事件 -----
        event = _inject_random_event(env, rng)
        env.current_event = event
        print(f"  Event: {event.description}")

        before = env.snapshot()

        # ----- 3. PPO 重分配 -----
        masks = get_action_masks(env)
        raw_action, _ = model.predict(env._get_obs(), deterministic=True, action_masks=masks)
        repaired_action, _ = repair_action(env, raw_action)
        env._execute_action(repaired_action)

        after = env.snapshot()
        assignment_json = env.export_assignment_json()
        action_summary = build_action_summary(old_assignments, repaired_action)

        # ----- 4. 输出结果 -----
        print(action_summary)
        print_state(env, f"第{step}轮 PPO 分配后")

        # ----- 5. 保存本轮文件 -----
        round_dir = OUTPUT_DIR / f"round_{step:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        save_json(round_dir / "assignment.json", assignment_json)
        save_json(round_dir / "snapshots.json", {
            "round": step,
            "before": before, "after": after,
            "event": event.description,
            "action_summary": action_summary,
            "raw_action": [int(x) for x in raw_action],
            "repaired_action": [int(x) for x in repaired_action],
        })
        save_comparison_figure(
            before, after, event.description, action_summary,
            str(round_dir / "before_after.png")
        )

        round_records.append({
            "round": step,
            "event": event.description,
            "action_summary": action_summary,
            "snapshot": after,
        })

        print(f"  → saved to {round_dir}")

        # ----- 6. 检查是否还能继续 -----
        alive_count = sum(1 for u in env.uavs.values() if u.alive)
        assignable = sum(1 for u in env.uavs.values()
                         if u.alive and u.task != TaskType.TRACK)
        if assignable == 0:
            print(f"\n  [STOP] 没有可分配搜索任务的无人机，在第 {step} 轮终止")
            break

    # ----- 保存汇总 -----
    summary = {
        "scenario": "example_target_discovered.json",
        "n_rounds_completed": len(round_records),
        "initial_snapshot": initial_snapshot,
        "final_snapshot": env.snapshot(),
        "final_assignment": env.export_assignment_json(),
        "rounds": round_records,
    }
    save_json(OUTPUT_DIR / "summary.json", summary)
    save_loop_overview_figure(initial_snapshot, round_records, str(OUTPUT_DIR / "overview.png"))
    save_loop_animation_html(initial_snapshot, round_records, str(OUTPUT_DIR / "animation.html"))
    print(f"\n{'='*60}")
    print(f"  Done. {len(round_records)} rounds completed.")
    print(f"  Results saved to {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_loop_test()
