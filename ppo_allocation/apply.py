"""模型应用与结果导出模块。

流程：
  1. 从 JSON 加载合理的初始任务分配态势
  2. 程序随机注入一个突发事件（区域空缺/无人机损毁/天气恶化/目标发现/目标摧毁）
  3. PPO 根据突发事件重新分配任务
  4. 导出分配结果 JSON、before/after 快照、对比图（含事件和动作详情）

运行方式：
    # 使用 JSON 场景 + 随机注入事件（默认）
    python apply.py

    # 指定场景文件
    python apply.py --scenario scenarios/xxx.json

    # 完全随机生成
    python apply.py --random

输出：
    results/eval/ppo_assignment_output.json
    results/eval/ppo_before_after_snapshots.json
    results/eval/ppo_before_after.png
    results/eval/ppo_before_after.gif
"""

from pathlib import Path
import json
import argparse
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
from utils.visualization import save_before_after_animation, save_comparison_figure
from utils.logger import save_json
from policy.action_repair import repair_action

# 动作码 → 可读名称
ACTION_NAMES = {
    ActionCode.KEEP:   "KEEP",
    ActionCode.U0:     "U0",
    ActionCode.U1:     "U1",
    ActionCode.U2:     "U2",
    ActionCode.U3:     "U3",
    ActionCode.NO_UAV: "NO_UAV",
}

# 事件类型 → 可读名称
EVENT_NAMES = {
    EventType.REGION_VACANCY:     "区域空缺",
    # EventType.WEATHER_INVALID:    "天气恶化",  # 已注释：SAR 全天候，天气不影响
    EventType.UAV_DAMAGE:         "无人机损毁",
    EventType.TARGET_DESTROYED:   "目标被摧毁",
}


def load_scenario(env: UAVTaskAllocationEnv, scenario_path: str) -> dict:
    """从 JSON 文件加载初始态势到环境（只含状态，不含事件）。

    Args:
        env:          已构造的环境对象
        scenario_path: JSON 场景文件路径

    Returns:
        dict: 场景原始数据
    """
    scenario_path = Path(scenario_path)
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))

    env.decision_step = 0

    # 加载区域
    env.regions.clear()
    for rid_str, r in scenario["regions"].items():
        rid = int(rid_str)
        env.regions[rid] = Region(
            rid=rid,
            center_x=r["center_x"],
            center_y=r["center_y"],
            weather=Weather(r["weather"]),
            assigned_uav=r.get("assigned_uav", NO_UAV),
            need_reassign=r.get("need_reassign", False),
            priority=r.get("priority", 1.0),
        )

    # 加载无人机
    env.uavs.clear()
    for uid_str, u in scenario["uavs"].items():
        uid = int(uid_str)
        env.uavs[uid] = UAV(
            uid=uid,
            x=u["x"],
            y=u["y"],
            sensor=SensorType(u["sensor"]),
            alive=u.get("alive", True),
            sensor_failed=u.get("sensor_failed", False),
            task=TaskType(u["task"]),
            regions=set(u.get("regions", [])),
            target_id=u.get("target_id", NO_TARGET),
        )

    # 加载目标
    env.targets.clear()
    for tid_str, t in scenario["targets"].items():
        tid = int(tid_str)
        env.targets[tid] = Target(
            tid=tid,
            target_type=TargetType(t["target_type"]),
            x=t["x"],
            y=t["y"],
            region=t.get("region", 0),
            movable=t.get("movable", False),
            discovered=t.get("discovered", False),
            tracked=t.get("tracked", False),
            destroyed=t.get("destroyed", False),
            tracker_id=t.get("tracker_id", NO_UAV),
        )

    # 初始事件为空（后续随机注入）
    env.current_event = Event(
        event_type=EventType.REGION_VACANCY,
        affected_regions=[],
        description="(no event yet)",
    )

    return scenario


# =========================
# 随机事件注入
# =========================

def _inject_random_event(env: UAVTaskAllocationEnv, rng: np.random.Generator) -> Event:
    """随机选择一个可行的突发事件并应用到环境中。

    扫描 5 种事件类型，收集所有可执行的候选事件，
    随机选择一个应用，修改环境状态，返回事件对象。

    Args:
        env: 环境对象（已加载初始态势）
        rng: numpy 随机数生成器

    Returns:
        Event: 实际应用的事件对象（失败时返回占位事件）
    """
    candidates = []

    # ---- 候选 1：无人机损毁 ----
    damageable = [uid for uid, u in env.uavs.items()
                  if u.alive and len(u.regions) > 0]
    if damageable:
        uid = int(rng.choice(damageable))
        region_list = list(env.uavs[uid].regions)
        candidates.append((
            lambda uid=uid, rl=region_list: _apply_uav_damage(env, uid, rl),
            f"U{uid}无人机损毁，其负责的{len(region_list)}个区域出现空缺",
        ))

    # # ---- 候选 3：天气恶化（EO 区域变雨天） ----
    # # 已注释：全部使用 SAR 传感器，天气不影响传感器有效性
    # eo_regions = [rid for rid, r in env.regions.items()
    #               if r.assigned_uav != NO_UAV
    #               and env.uavs[r.assigned_uav].sensor == SensorType.EO
    #               and env.uavs[r.assigned_uav].alive
    #               and r.weather == Weather.SUNNY]
    # if eo_regions:
    #     rid = int(rng.choice(eo_regions))
    #     candidates.append((
    #         lambda rid=rid: _apply_weather_invalid(env, rid),
    #         f"R{rid}区域天气恶化为雨天，EO传感器U{env.regions[rid].assigned_uav}失效",
    #     ))

    # ---- 候选 4：搜索无人机发现目标（转入跟踪） ----
    # 只允许无人机发现其搜索区域内的目标
    searchers = [uid for uid, u in env.uavs.items()
                 if u.alive and u.task == TaskType.SEARCH and len(u.regions) > 0]
    undiscovered = [tid for tid, t in env.targets.items()
                    if not t.discovered and not t.destroyed]
    valid_pairs = [(uid, tid) for uid in searchers for tid in undiscovered
                   if env.targets[tid].region in env.uavs[uid].regions]
    if valid_pairs:
        uid, tid = valid_pairs[int(rng.choice(len(valid_pairs)))]
        region_list = list(env.uavs[uid].regions)
        candidates.append((
            lambda uid=uid, tid=tid, rl=region_list: _apply_target_discovered(env, uid, tid, rl),
            f"U{uid}在搜索中发现T{tid}，转入TRACK，其{len(region_list)}个搜索区域出现空缺",
        ))

    # ---- 候选 5：目标被摧毁（释放跟踪无人机） ----
    tracked_targets = [tid for tid, t in env.targets.items()
                       if t.tracked and not t.destroyed and t.tracker_id != NO_UAV]
    if tracked_targets:
        tid = int(rng.choice(tracked_targets))
        tracker = env.targets[tid].tracker_id
        candidates.append((
            lambda tid=tid, tracker=tracker: _apply_target_destroyed(env, tid, tracker),
            f"T{tid}被摧毁，U{tracker}被释放，可重新加入搜索",
        ))

    # 如果没有候选（极端情况），不做任何事
    if not candidates:
        return Event(EventType.REGION_VACANCY, [], description="No valid event possible")

    # 随机选择一个候选事件执行
    idx = int(rng.choice(len(candidates)))
    return candidates[idx][0]()


def _apply_region_vacancy(env, rid: int) -> Event:
    """清空指定区域的分配，产生空缺事件。"""
    env._clear_region_assignment(rid)
    env.regions[rid].need_reassign = True
    return Event(EventType.REGION_VACANCY, [rid],
                 description=f"R{rid}区域搜索无人机离开，出现空缺，需PPO重新分配")


def _apply_uav_damage(env, uid: int, affected_regions: list) -> Event:
    """损毁指定无人机，释放其所有区域。"""
    u = env.uavs[uid]
    u.alive = False
    u.task = TaskType.IDLE
    u.target_id = NO_TARGET
    for rid in affected_regions:
        env._clear_region_assignment(rid)
        env.regions[rid].need_reassign = True
    return Event(EventType.UAV_DAMAGE, affected_regions, damaged_uav=uid,
                 description=f"U{uid}损毁，其{len(affected_regions)}个区域出现空缺，需PPO重新分配")


# def _apply_weather_invalid(env, rid: int) -> Event:
#     """指定区域天气恶化为雨天，EO 传感器失效。"""
#     # 已注释：全部使用 SAR 传感器，天气不影响传感器有效性
#     old_uid = env.regions[rid].assigned_uav
#     env.regions[rid].weather = Weather.RAINY
#     env._clear_region_assignment(rid)
#     env.regions[rid].need_reassign = True
#     return Event(EventType.WEATHER_INVALID, [rid],
#                  weather_disabled_uav=old_uid,
#                  description=f"R{rid}天气恶化为雨天，U{old_uid}(EO)失效，需PPO重新分配")


def _apply_target_discovered(env, uid: int, tid: int, affected_regions: list) -> Event:
    """搜索无人机发现目标，转入跟踪，释放搜索区域。"""
    u = env.uavs[uid]
    t = env.targets[tid]
    t.discovered = True
    t.tracked = True
    t.tracker_id = uid
    u.regions.clear()
    u.task = TaskType.TRACK
    u.target_id = tid
    for rid in affected_regions:
        env.regions[rid].assigned_uav = NO_UAV
        env.regions[rid].need_reassign = True
    return Event(EventType.REGION_VACANCY, affected_regions,
                 description=f"U{uid}发现T{tid}({_target_type_name(t.target_type)})，转入TRACK，"
                             f"其{len(affected_regions)}个搜索区域出现空缺，需PPO重新分配")


def _apply_target_destroyed(env, tid: int, tracker: int) -> Event:
    """目标被摧毁，释放跟踪无人机。"""
    t = env.targets[tid]
    t.destroyed = True
    t.tracked = False
    t.tracker_id = NO_UAV
    u = env.uavs[tracker]
    if u.alive:
        u.task = TaskType.IDLE
        u.target_id = NO_TARGET
    return Event(EventType.TARGET_DESTROYED, [], released_uav=tracker,
                 description=f"T{tid}被摧毁，U{tracker}释放，可重新加入搜索，需PPO重新分配")


def _target_type_name(tt):
    return "COMMAND" if tt == TargetType.COMMAND else "CAR"


# =========================
# 动作摘要
# =========================

def build_action_summary(old_assignments: dict, repaired_action) -> str:
    """构建 PPO 动作的可读摘要。"""
    lines = ["Region   PPO决策         变化"]
    lines.append("-" * 42)
    for rid in range(NUM_REGIONS):
        final_code = int(repaired_action[rid])
        final_name = ACTION_NAMES.get(final_code, f"??({final_code})")
        old_uid = old_assignments.get(rid, NO_UAV)
        old_name = f"U{old_uid}" if old_uid != NO_UAV else "UNASSIGNED"

        if final_code == ActionCode.KEEP:
            change = "(不变)"
        elif final_code == ActionCode.NO_UAV:
            change = f"{old_name} → UNASSIGNED"
        else:
            new_name = f"U{final_code - 1}"
            change = f"{old_name} → {new_name}"

        lines.append(f"  R{rid}    {final_name:<8}    {change}")
    return "\n".join(lines)


# =========================
# 主流程
# =========================

def run_apply(
    model_path: str,
    output_dir: str = "results/eval",
    deterministic: bool = True,
    scenario_path: str = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = UAVTaskAllocationEnv(random_event_mode=False)

    if scenario_path:
        # ===== 自定义 JSON 场景 =====
        scenario = load_scenario(env, scenario_path)
        env._get_obs()  # 初始化观测（确保观测维度一致）
        print(f"Loaded scenario: {scenario.get('scenario_name', 'N/A')}")
        print(f"  {scenario.get('description', '')}\n")

        # 打印初始分配
        print("Initial task allocation:")
        for rid in range(NUM_REGIONS):
            r = env.regions[rid]
            uid = r.assigned_uav
            u = env.uavs[uid] if uid != NO_UAV else None
            # w = "Sunny" if r.weather == Weather.SUNNY else "Rainy"  # 已注释：SAR 全天候
            if u:
                print(f"  R{rid} ← U{uid}({_sensor_short(u.sensor)})")
            else:
                print(f"  R{rid} ← UNASSIGNED")
        print()

        # ===== 随机注入突发事件 =====
        rng = np.random.default_rng()
        event = _inject_random_event(env, rng)
        env.current_event = event
        print(f"Injected event: {event.description}\n")

        # 事件后的状态（before = 事件后、PPO 前）
        old_assignments = {rid: r.assigned_uav for rid, r in env.regions.items()}
        before = env.snapshot()

        # ===== PPO 重分配 =====
        model = MaskablePPO.load(model_path, env=env, device="cpu")
        masks = get_action_masks(env)
        raw_action, _ = model.predict(env._get_obs(), deterministic=deterministic, action_masks=masks)

        repaired_action, _ = repair_action(env, raw_action)
        env._execute_action(repaired_action)

        after = env.snapshot()
        assignment_json = env.export_assignment_json()
        reward = 0.0

    else:
        # ===== 纯随机场景 =====
        obs, info = env.reset()
        event_text = info["event"].description
        old_assignments = {rid: r.assigned_uav for rid, r in env.regions.items()}
        before = env.snapshot()

        model = MaskablePPO.load(model_path, env=env, device="cpu")
        masks = get_action_masks(env)
        raw_action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)

        obs, reward, _, _, info = env.step(raw_action)
        repaired_action = info["repaired_action"]
        after = env.snapshot()
        assignment_json = env.export_assignment_json()

    # ===== 输出摘要 =====
    event_text = env.current_event.description
    action_summary = build_action_summary(old_assignments, repaired_action)

    print(f"{'='*55}")
    print(f"  Event: {event_text}")
    print(f"{'='*55}")
    print(action_summary)
    print(f"{'='*55}\n")

    # ===== 导出文件 =====
    json_path = output_dir / "ppo_assignment_output.json"
    gif_path = output_dir / "ppo_before_after.gif"
    png_path = output_dir / "ppo_before_after.png"
    snapshot_path = output_dir / "ppo_before_after_snapshots.json"

    save_json(json_path, assignment_json)
    save_json(snapshot_path, {
        "before": before, "after": after, "reward": reward,
        "event": event_text, "action_summary": action_summary,
        "raw_action": [int(x) for x in raw_action],
        "repaired_action": [int(x) for x in repaired_action],
    })
    save_comparison_figure(before, after, event_text, action_summary, str(png_path))
    save_before_after_animation(before, after, event_text, action_summary, str(gif_path))

    print(f"Saved assignment JSON       → {json_path}")
    print(f"Saved before/after snapshots → {snapshot_path}")
    print(f"Saved comparison figure      → {png_path}")
    print(f"Saved animation              → {gif_path}")


def _sensor_short(sensor):
    return "EO" if sensor == SensorType.EO else "SAR"


def main():
    parser = argparse.ArgumentParser(description="Apply PPO to scenario + random event.")
    parser.add_argument("--scenario", type=str, default=None,
                        help="JSON scenario file (clean initial state).")
    parser.add_argument("--random", action="store_true",
                        help="Use fully random environment (ignores scenario).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible event injection.")
    parser.add_argument("--output-dir", type=str, default="results/eval")
    args = parser.parse_args()

    model_path = "results\\models\\run_20260602_220416\\maskable_ppo_uav_task_allocation_50000_steps.zip"

    scenario_path = args.scenario
    if args.random:
        scenario_path = None
    elif scenario_path is None:
        default_scenario = Path("scenarios/example_target_discovered.json")
        if default_scenario.exists():
            scenario_path = str(default_scenario)

    run_apply(model_path, args.output_dir, scenario_path=scenario_path)


if __name__ == "__main__":
    main()
