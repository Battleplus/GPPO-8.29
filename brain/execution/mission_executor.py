"""Mission execution layer — drives Isaac platforms through the full mission pipeline.

Architecture
------------
* One ``MissionExecutor`` owns the Isaac step loop.
* The Brain FSM does planning: MILP once for recon, once for strike;
  PPO for dynamic task reallocation during recon.
* This layer consumes ``FormationPlan`` / ``Route`` waypoints, advances each
  platform one physics step per frame, and fires Brain events at milestones.

Layering
--------
::

   ┌─────────────────────────────────────────────┐
   │  MILP        → 任务分配（侦察/打击各1次）     │
   │  PPO         → 动态重分配（事件驱动，0-N次）   │
   ├─────────────────────────────────────────────┤
   │  MPPI        → 全局航路（集结→子区 / 阵位）    │
   │  search_planner → 巡航轨迹（跑道形/8字形等）   │
   ├─────────────────────────────────────────────┤
   │  DRL (drl_env) → 局部避障（轨迹点之间实时推理） │
   └─────────────────────────────────────────────┘

PlatformRunner phases
---------------------
FORMATION_TRANSIT    MPPI waypoint following (staging → cell / → strike position)
PATROL               search_planner waypoints, DRL between each adjacent pair
DONE / LOST
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

from ..core.events import MissionEvent

logger = logging.getLogger(__name__)

_PROJECT = Path(__file__).resolve().parents[3]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# ── Optional imports ──────────────────────────────────────────────────────
_DRL_AVAILABLE = False
try:
    from drl_env.core import (  # type: ignore[import-not-found]
        build_observation,
        build_obstacle_list_from_scene,
        MAX_SPEED,
        SENSE_RANGE,
    )
    from drl_env.numpy_policy import NumpyPolicy  # type: ignore[import-not-found]
    _DRL_AVAILABLE = True
except ImportError:
    pass

_SEARCH_PLANNER_AVAILABLE = False
try:
    from search_planner.planner import plan as search_plan  # type: ignore[import-not-found]
    from search_planner.config import PlannerConfig  # type: ignore[import-not-found]
    _SEARCH_PLANNER_AVAILABLE = True
except ImportError:
    pass

_ACS_AVAILABLE = False
try:
    from scenes import air_combat_scene as _acs  # type: ignore[import-not-found]
    _ACS_AVAILABLE = True
except ImportError:
    _acs = None  # type: ignore[assignment]


# ── Platform state machine ────────────────────────────────────────────────

class _Phase(Enum):
    FORMATION_TRANSIT = auto()   # following MPPI waypoints toward cell/target
    PATROL            = auto()   # following search_planner waypoints, DRL between each
    TRACK             = auto()   # hold target contact / designation
    STRIKE_TRANSIT    = auto()   # following MPPI waypoints toward target point
    ATTACK            = auto()
    DONE              = auto()
    LOST              = auto()


class PlatformRunner:
    """Per-platform state machine — no extra threads, stepped every frame."""

    WAYPOINT_RADIUS = 15.0   # scene units

    def __init__(
        self,
        platform: Any,
        transit_waypoints: list[list[float]] | None = None,
        phase: _Phase = _Phase.FORMATION_TRANSIT,
        policy: Any | None = None,
        obstacles: list[dict] | None = None,
    ) -> None:
        self.platform = platform
        # Transit waypoints (MPPI output)
        self.transit_wps: list[np.ndarray] = [
            np.array(wp, dtype=float) for wp in (transit_waypoints or [])
        ]
        self.transit_index = 0
        # Patrol waypoints (search_planner output, populated after transit done)
        self.patrol_wps: list[np.ndarray] = []
        self.patrol_index = 0
        self.phase = phase
        self.policy = policy
        self.obstacles = obstacles or []

    # ── per-frame step ───────────────────────────────────────────────────

    def step(self, dt: float, scene: Any) -> None:
        if self.phase in (_Phase.DONE, _Phase.LOST):
            return
        if getattr(self.platform, "destroyed", False):
            self.phase = _Phase.LOST
            return

        if self.phase in (_Phase.FORMATION_TRANSIT, _Phase.STRIKE_TRANSIT):
            self._step_transit(dt, scene)
        elif self.phase == _Phase.PATROL:
            self._step_patrol(dt, scene)
        elif self.phase in (_Phase.TRACK, _Phase.ATTACK):
            return

    # ── transit: follow MPPI waypoint by waypoint ────────────────────────

    def _step_transit(self, dt: float, scene: Any) -> None:
        if self.transit_index >= len(self.transit_wps):
            self.phase = _Phase.DONE
            return
        reached = self._move_toward(self.transit_wps[self.transit_index], dt, scene)
        if reached:
            self.transit_index += 1

    # ── patrol: follow search_planner waypoints, DRL between each pair ───

    def _step_patrol(self, dt: float, scene: Any) -> None:
        if self.patrol_index >= len(self.patrol_wps):
            self.phase = _Phase.DONE
            return
        # Use DRL for the leg between current pos → next patrol waypoint
        if _DRL_AVAILABLE and self.policy is not None:
            self._step_drl(dt, scene)
        else:
            # Fallback: simple approach without DRL
            self._move_toward(self.patrol_wps[self.patrol_index], dt, scene)

    # ── DRL local obstacle avoidance ─────────────────────────────────────

    def _step_drl(self, dt: float, scene: Any) -> None:
        target_wp = self.patrol_wps[self.patrol_index]
        pos = np.array(self.platform.position, dtype=float)
        vel = np.array(self.platform.velocity, dtype=float)

        obs = build_observation(
            drone_pos=pos, drone_vel=vel, target_pos=target_wp,
            obstacles=self.obstacles, sense_range=SENSE_RANGE, max_speed=MAX_SPEED,
        )
        action_norm = self.policy.forward(obs)
        accel_cmd = action_norm * self.policy.max_accel
        self._apply_accel(accel_cmd, dt, scene)

        if float(np.linalg.norm(target_wp - pos)) < self.WAYPOINT_RADIUS:
            self.patrol_index += 1

    # ── simple waypoint approach (no DRL) ────────────────────────────────

    def _move_toward(self, target_wp: np.ndarray, dt: float, scene: Any) -> bool:
        """Return True when arrived at target_wp."""
        mm = self.platform.motion_model
        pos = np.array(self.platform.position, dtype=float)
        vel = np.array(self.platform.velocity, dtype=float)
        to_wp = target_wp - pos
        dist = float(np.linalg.norm(to_wp))
        if dist < self.WAYPOINT_RADIUS:
            return True
        desired_vel = (to_wp / dist) * mm.max_speed * min(1.0, dist / 80.0)
        accel_cmd = (desired_vel - vel) / max(dt, 1e-3)
        self._apply_accel(accel_cmd, dt, scene)
        return False

    def _apply_accel(self, accel_cmd: np.ndarray, dt: float, scene: Any) -> None:
        mm = self.platform.motion_model
        an = float(np.linalg.norm(accel_cmd))
        if an > mm.max_accel:
            accel_cmd *= mm.max_accel / an
        mm.state.velocity += accel_cmd * dt
        sn = float(np.linalg.norm(mm.state.velocity))
        if sn > mm.max_speed:
            mm.state.velocity *= mm.max_speed / sn
        mm.state.position += mm.state.velocity * dt
        if _ACS_AVAILABLE and scene is not None:
            clearance = _acs._air_platform_clearance_units(
                scene.config, self.platform.spec, scene.meters_per_unit)
            mm.state.position = _acs._clamp_air_platform_above_terrain(
                mm.state.position, scene.map_size_units,
                scene.terrain_visual_height_units, clearance)
            mm._update_heading_attitude(dt, accel_cmd=accel_cmd)
            _acs._set_root_pose(self.platform.root_prim, mm.state.position, mm.state)
            _acs._spin_rotors(self.platform.rotor_prims, getattr(scene, "tactical_time_s", 0.0))


# ── Mission-level execution worker ────────────────────────────────────────

class FlightExecutionWorker:
    """Own the simulation step loop and exchange plans/events via queues."""

    DT = 1.0 / 30.0

    def __init__(
        self,
        brain: Any,
        environment: Any,
        *,
        command_queue: queue.Queue | None = None,
        event_queue: queue.Queue | None = None,
        drl_policy_path: str | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.brain = brain
        self.env = environment
        self.command_queue = command_queue or queue.Queue()
        self.event_queue = event_queue or queue.Queue()
        self.max_steps = max_steps
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._runners: dict[str, PlatformRunner] = {}
        self._runner_phase: dict[str, str] = {}
        self._completed: set[tuple[str, str]] = set()
        self._policy: Any | None = None
        self._obstacles: list[dict] = []

        if drl_policy_path and _DRL_AVAILABLE:
            p = Path(drl_policy_path)
            if p.exists():
                self._policy = NumpyPolicy(str(p))

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="FlightExecutionWorker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def submit_plan(self, plan_type: str, plan: Any) -> None:
        self.command_queue.put({"type": "PLAN", "plan_type": plan_type, "plan": plan})

    def next_event(self, timeout: float | None = None) -> MissionEvent | None:
        try:
            return self.event_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self) -> None:
        try:
            scene = self._ensure_scene()
            if _DRL_AVAILABLE and _ACS_AVAILABLE:
                self._obstacles = build_obstacle_list_from_scene(scene)
            step = 0
            while not self._stop.is_set():
                if self.max_steps is not None and step >= self.max_steps:
                    break
                step += 1
                self._drain_commands(scene)
                scene = self.env.step(self.DT)
                self._step_runners(scene)
        except Exception as exc:
            logger.exception("FlightExecutionWorker failed")
            self.event_queue.put(
                MissionEvent.execution_failed(
                    str(exc), source="FlightExecutionWorker"
                )
            )

    def _ensure_scene(self) -> Any:
        scene = getattr(self.env, "scene", None)
        if scene is not None:
            return scene
        initializer = getattr(self.env, "initialize", None)
        if callable(initializer):
            return initializer()
        raise RuntimeError("Execution environment has no scene")

    def _drain_commands(self, scene: Any) -> None:
        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                return
            if command.get("type") == "STOP":
                self.stop()
                return
            if command.get("type") == "PLAN":
                self._load_plan(
                    scene,
                    command.get("plan_type", ""),
                    command.get("plan"),
                )

    def _load_plan(self, scene: Any, plan_type: str, plan: Any) -> None:
        if plan is None:
            return
        phase = "recon" if plan_type == "recon" else "strike"
        for route in getattr(plan, "routes", []):
            pid = str(route.platform)
            platform = self._platform_by_id(scene, pid)
            if platform is None:
                continue
            runner_phase = (
                _Phase.FORMATION_TRANSIT
                if phase == "recon"
                else _Phase.STRIKE_TRANSIT
            )
            self._runners[pid] = PlatformRunner(
                platform=platform,
                transit_waypoints=route.waypoints,
                phase=runner_phase,
                policy=self._policy if phase == "recon" else None,
                obstacles=self._obstacles,
            )
            self._runner_phase[pid] = phase
            self._completed.discard((pid, phase))

    @staticmethod
    def _platform_by_id(scene: Any, pid: str) -> Any | None:
        for platform in getattr(scene, "platforms", []):
            if str(getattr(platform, "entity_id", "")) == str(pid):
                return platform
        return None

    def _step_runners(self, scene: Any) -> None:
        time_scale = float(
            getattr(scene, "config", {})
            .get("simulation", {})
            .get("time_scale", 1.0)
        )
        for pid, runner in list(self._runners.items()):
            runner.step(self.DT * time_scale, scene)
            phase = self._runner_phase.get(pid, "")
            if runner.phase == _Phase.LOST:
                key = (pid, "lost")
                if key not in self._completed:
                    self._completed.add(key)
                    self.event_queue.put(
                        MissionEvent.platform_lost(
                            {"platform_id": pid}, source="FlightExecutionWorker"
                        )
                    )
                continue
            if runner.phase != _Phase.DONE:
                continue
            key = (pid, phase)
            if key in self._completed:
                continue
            self._completed.add(key)
            route = self._route_metadata(pid)
            if phase == "recon":
                self.event_queue.put(
                    MissionEvent.recon_cell_done(
                        {"platform_id": pid, **route},
                        source="FlightExecutionWorker",
                    )
                )
            elif phase == "strike":
                payload = {"platform_id": pid, **route}
                self.event_queue.put(
                    MissionEvent.strike_position_reached(
                        payload, source="FlightExecutionWorker"
                    )
                )
                self.event_queue.put(
                    MissionEvent.attack_finished(
                        {**payload, "success": True, "destroyed": True},
                        source="FlightExecutionWorker",
                    )
                )

    def _route_metadata(self, pid: str) -> dict[str, Any]:
        for plan in (
            getattr(self.brain.context, "recon_formation_plan", None),
            getattr(self.brain.context, "action_formation_plan", None),
        ):
            for route in getattr(plan, "routes", []) if plan is not None else []:
                if str(getattr(route, "platform", "")) == str(pid):
                    return {
                        "target_id": str(getattr(route, "target_id", "")),
                        "position_id": str(getattr(route, "position_id", "")),
                        "metadata": dict(getattr(route, "metadata", {}) or {}),
                    }
        return {}


# ── Main executor ─────────────────────────────────────────────────────────

class MissionExecutor:
    """Drive Isaac sim through the full pipeline:

    1. brain.start()  → MILP recon alloc + MPPI recon plan (once)
    2. Formation transit → patrol (search_planner + DRL)
    3. Target detection / platform loss → PPO realloc (event-driven)
    4. Recon done → MILP strike alloc + MPPI strike plan (once)
    5. Strike transit → mission complete
    """

    DT = 1.0 / 30.0

    def __init__(
        self, brain: Any, environment: Any,
        drl_policy_path: str | None = None,
        max_steps: int = 100_000,
    ) -> None:
        self.brain = brain
        self.env = environment
        self.max_steps = max_steps
        self._policy: Any | None = None
        self._obstacles: list[dict] = []
        self._runners: dict[str, PlatformRunner] = {}
        # Recon bookkeeping
        self._recon_detections: list[dict] = []
        self._recon_platform_ids: set[str] = set()
        self._recon_done_ids: set[str] = set()
        # Strike bookkeeping
        self._strike_platform_ids: set[str] = set()
        self._strike_done_ids: set[str] = set()
        self._strike_target_by_pid: dict[str, str] = {}
        # each recon runner's cell centre km, for search_planner
        self._cell_km: dict[str, tuple[float, float]] = {}

        if drl_policy_path and _DRL_AVAILABLE:
            p = Path(drl_policy_path)
            if p.exists():
                self._policy = NumpyPolicy(str(p))
                logger.info("DRL policy loaded: %s", p)
            else:
                logger.warning("DRL policy not found: %s", p)

    # ── Main loop ────────────────────────────────────────────────────────

    def run(self) -> None:
        from brain.core.events import MissionEvent
        from brain.core.states import MissionState
        from brain.integration.context_sync import sync_context_from_air_combat_scene

        state = self.brain.start()
        logger.info("[Executor] brain.start() → %s", state.value)

        scene = self.env.scene
        if scene is None:
            raise RuntimeError("Isaac scene not initialized")

        if _DRL_AVAILABLE and _ACS_AVAILABLE:
            self._obstacles = build_obstacle_list_from_scene(scene)

        step = 0
        while step < self.max_steps:
            step += 1
            scene = self.env.step(self.DT)
            sync_context_from_air_combat_scene(self.brain.context, scene)
            self._check_platform_losses(scene)

            current = self.brain.current_state

            if current == MissionState.RECON_PLAN_READY:
                self._setup_recon_runners(scene)
                state = self.brain.dispatch(MissionEvent.recon_plan_dispatched())
                logger.info("[Executor] recon plan dispatched → %s", state.value)
                continue

            if current == MissionState.RECON_EXECUTING:
                self._step_runners(scene)
                self._check_recon_contacts(scene)
                if self._all_recon_done():
                    state = self.brain.dispatch(MissionEvent.recon_finished())
                    logger.info("[Executor] recon finished → %s", state.value)
                continue

            if current == MissionState.WAIT_RECON_RESULT:
                payload = {
                    "detections": list(self._recon_detections),
                    "confirmed": [
                        d["truth_id"] for d in self._recon_detections
                        if d.get("confidence", 0.0) >= 0.7
                    ],
                }
                state = self.brain.dispatch(MissionEvent.recon_result_received(data=payload))
                logger.info("[Executor] recon result delivered (%d detections) → %s",
                            len(self._recon_detections), state.value)
                continue

            if current == MissionState.ACTION_PLAN_READY:
                self._setup_strike_runners(scene)
                state = self.brain.dispatch(MissionEvent.action_plan_dispatched())
                logger.info("[Executor] strike plan dispatched → %s", state.value)
                continue

            if current == MissionState.ACTION_EXECUTING:
                self._step_runners(scene)
                if self._all_strike_done():
                    state = self.brain.dispatch(MissionEvent.action_finished())
                    logger.info("[Executor] strike finished → %s", state.value)
                continue

            if current.is_terminal:
                logger.info("[Executor] terminal: %s", current.value)
                break

            self._step_runners(scene)

        logger.info("[Executor] exit step=%d state=%s", step, self.brain.current_state.value)

    # ── Setup ────────────────────────────────────────────────────────────

    def _platform_by_id(self, scene: Any, pid: str) -> Any | None:
        for p in scene.platforms:
            if str(p.entity_id) == pid:
                return p
        return None

    def _setup_recon_runners(self, scene: Any) -> None:
        self._runners.clear()
        self._recon_platform_ids.clear()
        self._recon_done_ids.clear()
        self._recon_detections.clear()
        self._cell_km.clear()

        plan = self.brain.context.recon_formation_plan
        if plan is None:
            logger.error("[Executor] no recon formation plan")
            return

        routes = getattr(plan, "routes", [])
        for route in routes:
            pid = str(route.platform)
            plat = self._platform_by_id(scene, pid)
            if plat is None:
                continue
            cell_km = route.metadata.get("cell_center_km")
            if cell_km:
                self._cell_km[pid] = (float(cell_km[0]), float(cell_km[1]))
            runner = PlatformRunner(
                platform=plat,
                transit_waypoints=route.waypoints,
                phase=_Phase.FORMATION_TRANSIT,
                policy=self._policy,
                obstacles=self._obstacles,
            )
            self._runners[pid] = runner
            self._recon_platform_ids.add(pid)

        logger.info("[Executor] %d recon runners: %s", len(self._runners), list(self._runners))

    def _setup_strike_runners(self, scene: Any) -> None:
        self._runners.clear()
        self._strike_platform_ids.clear()
        self._strike_done_ids.clear()
        self._strike_target_by_pid.clear()

        plan = self.brain.context.action_formation_plan
        if plan is None:
            logger.error("[Executor] no action formation plan")
            return

        for route in getattr(plan, "routes", []):
            pid = str(route.platform)
            plat = self._platform_by_id(scene, pid)
            if plat is None:
                continue
            runner = PlatformRunner(
                platform=plat,
                transit_waypoints=route.waypoints,
                phase=_Phase.STRIKE_TRANSIT,
                obstacles=self._obstacles,
            )
            self._runners[pid] = runner
            self._strike_platform_ids.add(pid)
            self._strike_target_by_pid[pid] = str(getattr(route, "target_id", ""))

        logger.info("[Executor] %d strike runners: %s", len(self._runners), list(self._runners))

    # ── Per-frame stepping ───────────────────────────────────────────────

    def _step_runners(self, scene: Any) -> None:
        ts = float(scene.config.get("simulation", {}).get("time_scale", 45.0))

        for pid, runner in list(self._runners.items()):
            runner.step(self.DT * ts, scene)

            # Recon: transit done → start patrol (search_planner waypoints + DRL)
            if pid in self._recon_platform_ids:
                if runner.phase == _Phase.DONE and not runner.patrol_wps:
                    # Transit arrived at cell — generate patrol waypoints
                    self._start_patrol(runner, pid, scene)
                if runner.phase == _Phase.DONE and runner.patrol_wps:
                    # Patrol completed — mark done
                    self._recon_done_ids.add(pid)
                    logger.info("[Executor] %s patrol complete", pid)

            if pid in self._strike_platform_ids:
                if runner.phase == _Phase.DONE and pid not in self._strike_done_ids:
                    self._strike_done_ids.add(pid)
                    target_id = self._strike_target_by_pid.get(pid, "")
                    if target_id:
                        result = self.brain.handle_attack_finished({
                            "platform_id": pid,
                            "target_id": target_id,
                            "success": True,
                            "destroyed": True,
                        })
                        if not result.success:
                            logger.warning(
                                "[Executor] attack completion handling failed for %s/%s: %s",
                                pid, target_id, result.reason,
                            )
                    logger.info("[Executor] %s strike arrival", pid)

            if runner.phase == _Phase.LOST:
                if pid in self._recon_platform_ids:
                    self._recon_done_ids.add(pid)
                if pid in self._strike_platform_ids:
                    self._strike_done_ids.add(pid)

    def _start_patrol(self, runner: PlatformRunner, pid: str, scene: Any) -> None:
        """Call search_planner to generate patrol waypoints for this cell."""
        cell_km = self._cell_km.get(pid)
        if cell_km is None or not _SEARCH_PLANNER_AVAILABLE:
            logger.warning("[Executor] %s: no cell info or search_planner missing → skip patrol", pid)
            self._recon_done_ids.add(pid)
            return

        try:
            cfg = PlannerConfig(
                area_center_km=cell_km,
                area_width_km=25.0,
                area_height_km=25.0,
                pattern="racetrack",
                angle_deg=30.0,
                altitude_agl_m=5000.0,
                cruise_speed_mps=200.0,
                map_size_km=float(scene.map_size_units) * float(scene.meters_per_unit) / 1000.0,
                meters_per_unit=float(scene.meters_per_unit),
            )
            result = search_plan(cfg)
            runner.patrol_wps = [
                np.array([wp.x, wp.y, wp.z], dtype=float)
                for wp in result.waypoints
            ]
            runner.patrol_index = 0
            runner.phase = _Phase.PATROL
            logger.info(
                "[Executor] %s patrol started — %d waypoints (pattern=%s, collisions %d→%d)",
                pid, len(runner.patrol_wps), cfg.pattern,
                result.stats["collision_count_before"],
                result.stats["collision_count_after"],
            )
        except Exception:
            logger.exception("[Executor] %s search_planner failed", pid)
            self._recon_done_ids.add(pid)

    # ── Contacts & losses ────────────────────────────────────────────────

    def _check_recon_contacts(self, scene: Any) -> None:
        for contact in getattr(scene, "contacts", []):
            tid = str(contact.get("target_id", ""))
            if any(d["truth_id"] == tid for d in self._recon_detections):
                continue
            conf = float(contact.get("detection_probability", 0.8))
            platform_id = str(contact.get("platform_id", ""))
            self._recon_detections.append({
                "truth_id": tid,
                "confidence": conf,
                "platform_id": platform_id,
            })
            result = self.brain.handle_target_detected({
                "target_id": tid,
                "platform_id": platform_id,
                "confidence": conf,
            })
            if not result.success:
                logger.warning(
                    "[Executor] target event handling failed for %s: %s",
                    tid, result.reason,
                )
            logger.info("[Executor] target detected: %s (conf=%.2f)", tid, conf)

    def _check_platform_losses(self, scene: Any) -> None:
        for platform in scene.platforms:
            pid = str(platform.entity_id)
            if not getattr(platform, "destroyed", False):
                continue
            runner = self._runners.get(pid)
            if runner is None or runner.phase == _Phase.LOST:
                continue
            runner.phase = _Phase.LOST
            logger.warning("[Executor] %s destroyed — PPO realloc", pid)
            result = self.brain.report_platform_loss(pid)
            if result.success:
                logger.info("[Executor] PPO realloc OK, rebuilding runners")
                self._setup_recon_runners(scene)
            else:
                logger.error("[Executor] PPO realloc failed: %s", result.reason)

    # ── Completion checks ────────────────────────────────────────────────

    def _all_recon_done(self) -> bool:
        if not self._recon_platform_ids:
            return False
        return self._recon_platform_ids == self._recon_done_ids

    def _all_strike_done(self) -> bool:
        if not self._strike_platform_ids:
            return False
        return self._strike_platform_ids == self._strike_done_ids
