from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MotionState:
    position: np.ndarray
    velocity: np.ndarray
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


class BaseMotionModel:
    def __init__(self, initial_position: np.ndarray, max_speed: float) -> None:
        self.state = MotionState(
            position=np.array(initial_position, dtype=float),
            velocity=np.zeros(3, dtype=float),
        )
        self.max_speed = max(0.1, float(max_speed))

    def step(self, dt: float, target_position: np.ndarray, target_velocity: np.ndarray) -> MotionState:
        raise NotImplementedError

    def _update_heading_attitude(self, dt: float, accel_cmd: np.ndarray | None = None) -> None:
        vel = self.state.velocity
        if np.linalg.norm(vel[:2]) > 1e-4:
            self.state.yaw_deg = math.degrees(math.atan2(float(vel[1]), float(vel[0])))
        speed_xy = float(np.linalg.norm(vel[:2]))
        climb = float(vel[2])
        if accel_cmd is None:
            accel_cmd = np.zeros(3, dtype=float)
        forward_pitch = max(-18.0, min(18.0, -math.degrees(math.atan2(climb, max(1.0, speed_xy)))))
        lateral_roll = max(-22.0, min(22.0, float(accel_cmd[1]) * 2.2))
        alpha = min(1.0, max(0.0, float(dt) * 4.0))
        self.state.pitch_deg = self.state.pitch_deg * (1.0 - alpha) + forward_pitch * alpha
        self.state.roll_deg = self.state.roll_deg * (1.0 - alpha) + lateral_roll * alpha


class GroundKinematicMotionModel(BaseMotionModel):
    def __init__(self, initial_position: np.ndarray, max_speed: float, max_turn_rate_deg: float = 28.0) -> None:
        super().__init__(initial_position, max_speed=max_speed)
        self.max_turn_rate_deg = float(max_turn_rate_deg)

    def step(self, dt: float, target_position: np.ndarray, target_velocity: np.ndarray) -> MotionState:
        dt = max(1e-3, float(dt))
        target_position = np.array(target_position, dtype=float)
        delta = target_position - self.state.position
        delta[2] = 0.0
        dist = float(np.linalg.norm(delta[:2]))
        if dist > 1e-6:
            desired_yaw = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
            yaw_error = ((desired_yaw - self.state.yaw_deg + 180.0) % 360.0) - 180.0
            yaw_step = max(-self.max_turn_rate_deg * dt, min(self.max_turn_rate_deg * dt, yaw_error))
            self.state.yaw_deg += yaw_step
            heading = np.array(
                [math.cos(math.radians(self.state.yaw_deg)), math.sin(math.radians(self.state.yaw_deg)), 0.0],
                dtype=float,
            )
            step_len = min(dist, self.max_speed * dt)
            new_position = self.state.position + heading * step_len
            new_position[2] = target_position[2]
            self.state.velocity = (new_position - self.state.position) / dt
            self.state.position = new_position
        else:
            self.state.velocity = np.zeros(3, dtype=float)
        self.state.pitch_deg = max(-6.0, min(6.0, -float(np.linalg.norm(self.state.velocity[:2])) * 0.08))
        self.state.roll_deg = 0.0
        return self.state


class TrackedDynamicMotionModel(BaseMotionModel):
    def __init__(self, initial_position: np.ndarray, max_speed: float, max_accel: float, drag: float = 0.35) -> None:
        super().__init__(initial_position, max_speed=max_speed)
        self.max_accel = max(0.1, float(max_accel))
        self.drag = max(0.0, float(drag))

    def step(self, dt: float, target_position: np.ndarray, target_velocity: np.ndarray) -> MotionState:
        dt = max(1e-3, float(dt))
        pos_error = np.array(target_position, dtype=float) - self.state.position
        pos_error[2] = 0.0
        vel_error = np.array(target_velocity, dtype=float) - self.state.velocity
        vel_error[2] = 0.0
        accel_cmd = 0.65 * pos_error + 1.05 * vel_error - self.drag * self.state.velocity
        accel_cmd[2] = 0.0
        accel_norm = float(np.linalg.norm(accel_cmd[:2]))
        if accel_norm > self.max_accel:
            accel_cmd *= self.max_accel / accel_norm
        self.state.velocity = self.state.velocity + accel_cmd * dt
        self.state.velocity[2] = 0.0
        speed = float(np.linalg.norm(self.state.velocity[:2]))
        if speed > self.max_speed:
            self.state.velocity *= self.max_speed / speed
        self.state.position = self.state.position + self.state.velocity * dt
        self.state.position[2] = float(target_position[2])
        self._update_heading_attitude(dt, accel_cmd=accel_cmd)
        self.state.pitch_deg = max(-8.0, min(8.0, -speed * 0.18))
        self.state.roll_deg = max(-12.0, min(12.0, float(accel_cmd[1]) * 3.8))
        return self.state


class SurfaceVesselDynamicMotionModel(BaseMotionModel):
    def __init__(
        self,
        initial_position: np.ndarray,
        max_speed: float,
        max_accel: float,
        drag: float = 0.45,
        current_xy: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        super().__init__(initial_position, max_speed=max_speed)
        self.max_accel = max(0.1, float(max_accel))
        self.drag = max(0.0, float(drag))
        self.current = np.array([float(current_xy[0]), float(current_xy[1]), 0.0], dtype=float)

    def step(self, dt: float, target_position: np.ndarray, target_velocity: np.ndarray) -> MotionState:
        dt = max(1e-3, float(dt))
        pos_error = np.array(target_position, dtype=float) - self.state.position
        pos_error[2] = 0.0
        vel_error = np.array(target_velocity, dtype=float) - self.state.velocity
        vel_error[2] = 0.0
        accel_cmd = 0.48 * pos_error + 0.95 * vel_error - self.drag * (self.state.velocity - self.current)
        accel_cmd[2] = 0.0
        accel_norm = float(np.linalg.norm(accel_cmd[:2]))
        if accel_norm > self.max_accel:
            accel_cmd *= self.max_accel / accel_norm
        self.state.velocity = self.state.velocity + accel_cmd * dt
        self.state.velocity[2] = 0.0
        speed = float(np.linalg.norm(self.state.velocity[:2]))
        if speed > self.max_speed:
            self.state.velocity *= self.max_speed / speed
        self.state.position = self.state.position + self.state.velocity * dt
        self.state.position[2] = float(target_position[2])
        self._update_heading_attitude(dt, accel_cmd=accel_cmd)
        self.state.roll_deg = max(-5.0, min(5.0, float(accel_cmd[1]) * 1.4))
        self.state.pitch_deg = max(-3.0, min(3.0, float(accel_cmd[0]) * 0.9))
        return self.state


class SubsurfaceDynamicMotionModel(BaseMotionModel):
    def __init__(
        self,
        initial_position: np.ndarray,
        max_speed: float,
        max_accel: float,
        buoyancy_gain: float = 1.1,
        vertical_damping: float = 1.8,
        horizontal_drag: float = 0.35,
    ) -> None:
        super().__init__(initial_position, max_speed=max_speed)
        self.max_accel = max(0.1, float(max_accel))
        self.buoyancy_gain = float(buoyancy_gain)
        self.vertical_damping = float(vertical_damping)
        self.horizontal_drag = float(horizontal_drag)

    def step(self, dt: float, target_position: np.ndarray, target_velocity: np.ndarray) -> MotionState:
        dt = max(1e-3, float(dt))
        target_position = np.array(target_position, dtype=float)
        target_velocity = np.array(target_velocity, dtype=float)
        pos_error = target_position - self.state.position
        vel_error = target_velocity - self.state.velocity
        accel_cmd = np.zeros(3, dtype=float)
        accel_cmd[:2] = 0.55 * pos_error[:2] + 1.05 * vel_error[:2] - self.horizontal_drag * self.state.velocity[:2]
        accel_cmd[2] = self.buoyancy_gain * pos_error[2] + 0.9 * vel_error[2] - self.vertical_damping * self.state.velocity[2]
        accel_norm = float(np.linalg.norm(accel_cmd))
        if accel_norm > self.max_accel:
            accel_cmd *= self.max_accel / accel_norm
        self.state.velocity = self.state.velocity + accel_cmd * dt
        speed = float(np.linalg.norm(self.state.velocity))
        if speed > self.max_speed:
            self.state.velocity *= self.max_speed / speed
        self.state.position = self.state.position + self.state.velocity * dt
        self._update_heading_attitude(dt, accel_cmd=accel_cmd)
        return self.state


class KinematicMotionModel(BaseMotionModel):
    def __init__(self, initial_position: np.ndarray, max_speed: float) -> None:
        super().__init__(initial_position, max_speed=max_speed)

    def step(self, dt: float, target_position: np.ndarray, target_velocity: np.ndarray) -> MotionState:
        dt = max(1e-3, float(dt))
        delta = np.array(target_position, dtype=float) - self.state.position
        dist = float(np.linalg.norm(delta))
        if dist > 1e-6:
            direction = delta / dist
            step_len = min(dist, self.max_speed * dt)
            new_position = self.state.position + direction * step_len
            self.state.velocity = (new_position - self.state.position) / dt
            self.state.position = new_position
        else:
            self.state.velocity = np.zeros(3, dtype=float)
        self._update_heading_attitude(dt)
        return self.state


class DynamicPointMassModel(BaseMotionModel):
    def __init__(
        self,
        initial_position: np.ndarray,
        max_speed: float,
        max_accel: float,
        position_gain: float = 1.2,
        velocity_gain: float = 1.6,
    ) -> None:
        super().__init__(initial_position, max_speed=max_speed)
        self.max_accel = max(0.1, float(max_accel))
        self.position_gain = float(position_gain)
        self.velocity_gain = float(velocity_gain)

    def step(self, dt: float, target_position: np.ndarray, target_velocity: np.ndarray) -> MotionState:
        dt = max(1e-3, float(dt))
        target_position = np.array(target_position, dtype=float)
        target_velocity = np.array(target_velocity, dtype=float)

        pos_error = target_position - self.state.position
        vel_error = target_velocity - self.state.velocity
        accel_cmd = self.position_gain * pos_error + self.velocity_gain * vel_error

        accel_norm = float(np.linalg.norm(accel_cmd))
        if accel_norm > self.max_accel:
            accel_cmd *= self.max_accel / accel_norm

        self.state.velocity = self.state.velocity + accel_cmd * dt
        speed = float(np.linalg.norm(self.state.velocity))
        if speed > self.max_speed:
            self.state.velocity *= self.max_speed / speed

        self.state.position = self.state.position + self.state.velocity * dt
        self._update_heading_attitude(dt, accel_cmd=accel_cmd)
        return self.state


def build_motion_model(model_cfg: dict[str, Any], initial_position: np.ndarray, domain: str = "air") -> BaseMotionModel:
    model_type = str(model_cfg.get("type", "kinematic")).lower()
    max_speed = float(model_cfg.get("max_speed", 18.0))
    domain = str(domain).lower()

    if domain == "ground":
        if model_type == "dynamic":
            return TrackedDynamicMotionModel(
                initial_position=np.array(initial_position, dtype=float),
                max_speed=max_speed,
                max_accel=float(model_cfg.get("max_accel", 2.5)),
                drag=float(model_cfg.get("drag", 0.35)),
            )
        return GroundKinematicMotionModel(
            initial_position=np.array(initial_position, dtype=float),
            max_speed=max_speed,
            max_turn_rate_deg=float(model_cfg.get("max_turn_rate_deg", 28.0)),
        )

    if domain == "surface":
        return SurfaceVesselDynamicMotionModel(
            initial_position=np.array(initial_position, dtype=float),
            max_speed=max_speed,
            max_accel=float(model_cfg.get("max_accel", 1.5)),
            drag=float(model_cfg.get("drag", 0.45)),
            current_xy=tuple(model_cfg.get("current_xy", (0.0, 0.0))),
        )

    if domain == "subsurface":
        return SubsurfaceDynamicMotionModel(
            initial_position=np.array(initial_position, dtype=float),
            max_speed=max_speed,
            max_accel=float(model_cfg.get("max_accel", 1.4)),
            buoyancy_gain=float(model_cfg.get("buoyancy_gain", 1.1)),
            vertical_damping=float(model_cfg.get("vertical_damping", 1.8)),
            horizontal_drag=float(model_cfg.get("horizontal_drag", 0.35)),
        )

    if model_type == "dynamic":
        return DynamicPointMassModel(
            initial_position=np.array(initial_position, dtype=float),
            max_speed=max_speed,
            max_accel=float(model_cfg.get("max_accel", 8.0)),
            position_gain=float(model_cfg.get("position_gain", 1.2)),
            velocity_gain=float(model_cfg.get("velocity_gain", 1.6)),
        )
    return KinematicMotionModel(
        initial_position=np.array(initial_position, dtype=float),
        max_speed=max_speed,
    )
