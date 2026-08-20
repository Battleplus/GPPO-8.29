"""Lifecycle wrapper for the existing Isaac air-combat environment."""

from __future__ import annotations

from typing import Any, Callable


class IsaacAirCombatEnvironment:
    """Create, step, reset, and close the repository's Isaac scene.

    Isaac imports are intentionally lazy so Brain unit tests and planning-only
    processes can import this module without an Isaac Sim installation.
    Factories may be injected for tests or an embedding application.
    """

    def __init__(
        self,
        scene_config: dict[str, Any] | None = None,
        *,
        headless: bool = False,
        app_config: dict[str, Any] | None = None,
        app_factory: Callable[[dict[str, Any]], Any] | None = None,
        stage_factory: Callable[[], Any] | None = None,
        scene_factory: Callable[[Any, dict[str, Any]], Any] | None = None,
        timeline_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.scene_config = dict(scene_config or {})
        self.headless = bool(headless)
        self.app_config = {
            "headless": self.headless,
            "renderer": "HydraStorm",
            "width": 1280,
            "height": 720,
            **(app_config or {}),
        }
        self._app_factory = app_factory
        self._stage_factory = stage_factory
        self._scene_factory = scene_factory
        self._timeline_factory = timeline_factory
        self.app: Any | None = None
        self.stage: Any | None = None
        self.scene: Any | None = None
        self.timeline: Any | None = None
        self.tactical_time_s = 0.0
        self.frame_count = 0

    @property
    def initialized(self) -> bool:
        return self.scene is not None

    def initialize(self):
        """Initialize SimulationApp, USD stage, scene, and timeline once."""
        if self.scene is not None:
            return self.scene

        if self._app_factory is None:
            from isaacsim import SimulationApp
            self._app_factory = SimulationApp
        self.app = self._app_factory(dict(self.app_config))

        if self._stage_factory is None:
            import omni.usd
            from pxr import Usd

            def default_stage_factory():
                stage = omni.usd.get_context().get_stage()
                if not stage:
                    stage = Usd.Stage.CreateInMemory()
                    omni.usd.get_context().set_stage(stage)
                return stage

            self._stage_factory = default_stage_factory
        self.stage = self._stage_factory()

        if self._scene_factory is None:
            from scenes.air_combat_scene import create_scene
            self._scene_factory = create_scene
        self.scene = self._scene_factory(
            self.stage, dict(self.scene_config)
        )

        if self._timeline_factory is None:
            import omni.timeline
            self._timeline_factory = (
                omni.timeline.get_timeline_interface
            )
        self.timeline = self._timeline_factory()
        if self.timeline is not None and hasattr(self.timeline, "play"):
            self.timeline.play()

        self.tactical_time_s = 0.0
        self.frame_count = 0
        return self.scene

    def step(self, dt: float = 1.0 / 30.0):
        """Advance tactical scene state and one Isaac application frame."""
        scene = self.initialize()
        time_scale = float(
            scene.config.get("simulation", {}).get("time_scale", 45.0)
        )
        tactical_dt = max(1e-6, float(dt)) * time_scale
        self.tactical_time_s += tactical_dt
        scene.update(
            tactical_dt=tactical_dt,
            tactical_time_s=self.tactical_time_s,
        )
        if self.app is not None and hasattr(self.app, "update"):
            self.app.update()
        self.frame_count += 1
        return scene

    def reset(self):
        """Reset tactical counters while keeping the initialized Isaac stage."""
        scene = self.initialize()
        self.tactical_time_s = 0.0
        self.frame_count = 0
        if hasattr(scene, "update"):
            scene.update(tactical_dt=1e-6, tactical_time_s=0.0)
        return scene

    def is_running(self) -> bool:
        if self.app is None:
            return False
        checker = getattr(self.app, "is_running", None)
        return bool(checker()) if callable(checker) else True

    def close(self) -> None:
        if self.timeline is not None and hasattr(self.timeline, "stop"):
            self.timeline.stop()
        if self.app is not None and hasattr(self.app, "close"):
            self.app.close()
        self.scene = None
        self.stage = None
        self.timeline = None
        self.app = None
