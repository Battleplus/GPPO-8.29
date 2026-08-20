from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import carb
import numpy as np
from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils

import omni.physx

_ATTACHED_STAGE_IDS: set[int] = set()


def _np3(values: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.array(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)


def _stage_id(stage) -> int:
    cache = UsdUtils.StageCache.Get()
    stage_id = cache.GetId(stage)
    if not stage_id.IsValid():
        cache.Insert(stage)
        stage_id = cache.GetId(stage)
    return int(stage_id.ToLongInt())


def _ensure_stage_attached(stage) -> int:
    stage_id = _stage_id(stage)
    if stage_id not in _ATTACHED_STAGE_IDS:
        psi = omni.physx.get_physx_simulation_interface()
        psi.attach_stage(stage_id)
        _ATTACHED_STAGE_IDS.add(stage_id)
    return stage_id


def _prim_id(prim) -> int:
    return int(PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())))


def _pose_from_prim(prim) -> tuple[np.ndarray, np.ndarray]:
    stage = prim.GetStage()
    xform = UsdGeom.Xformable(prim)
    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    transform = Gf.Transform(matrix)
    translation = transform.GetTranslation()
    rotation = transform.GetRotation().GetQuat()
    return (
        np.array([float(translation[0]), float(translation[1]), float(translation[2])], dtype=float),
        np.array([float(rotation.GetImaginary()[0]), float(rotation.GetImaginary()[1]), float(rotation.GetImaginary()[2]), float(rotation.GetReal())], dtype=float),
    )


def _yaw_from_quat(quat_xyzw: np.ndarray) -> float:
    x, y, z, w = map(float, quat_xyzw.tolist())
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _quat_from_ypr(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Gf.Quatf:
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return Gf.Quatf(w, Gf.Vec3f(x, y, z))


def _vector3(value) -> carb.Float3:
    vec = _np3(value).astype(float).tolist()
    return carb.Float3(vec)


def _rigid_body_pose(prim) -> tuple[np.ndarray, np.ndarray]:
    psi = omni.physx.get_physx_simulation_interface()
    path = str(prim.GetPath())
    try:
        pose = psi.get_rigidbody_transformation(path)
        if pose and pose.get("ret_val", True):
            pos = pose.get("position")
            rot = pose.get("rotation")
            if pos is not None and rot is not None:
                pos_arr = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=float)
                rot_arr = np.array([float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])], dtype=float)
                return pos_arr, rot_arr
    except Exception:
        pass
    return _pose_from_prim(prim)


def _rigid_body_velocity(prim) -> np.ndarray:
    stage = prim.GetStage()
    path = prim.GetPath()
    rb = UsdPhysics.RigidBodyAPI.Get(stage, path)
    if rb:
        try:
            vel = rb.GetVelocityAttr().Get()
            if vel is not None:
                return _np3(vel)
        except Exception:
            pass
    return np.zeros(3, dtype=float)


def _rigid_body_angular_velocity(prim) -> np.ndarray:
    stage = prim.GetStage()
    path = prim.GetPath()
    rb = UsdPhysics.RigidBodyAPI.Get(stage, path)
    if rb:
        try:
            ang = rb.GetAngularVelocityAttr().Get()
            if ang is not None:
                return _np3(ang)
        except Exception:
            pass
    return np.zeros(3, dtype=float)


def apply_force(prim, force: np.ndarray, torque: np.ndarray | None = None, application_pos: np.ndarray | None = None) -> None:
    psi = omni.physx.get_physx_simulation_interface()
    path = str(prim.GetPath())
    force_vec = _vector3(force)
    torque_vec = _vector3(torque if torque is not None else np.zeros(3, dtype=float))
    if application_pos is None:
        application_pos = _rigid_body_pose(prim)[0]
    pos_vec = _vector3(application_pos)
    stage_id = _ensure_stage_attached(prim.GetStage())
    prim_id = _prim_id(prim)

    try:
        psi.apply_force_at_pos(stage_id, prim_id, force_vec, pos_vec)
    except Exception as exc:
        raise RuntimeError(f"apply_force_at_pos failed for {path}: {exc}") from exc

    if float(np.linalg.norm(_np3(torque_vec))) > 1e-8:
        try:
            psi.apply_torque(stage_id, prim_id, torque_vec)
        except Exception as exc:
            raise RuntimeError(f"apply_torque failed for {path}: {exc}") from exc


def apply_torque(prim, torque: np.ndarray) -> None:
    apply_force(prim, np.zeros(3, dtype=float), torque=torque)


@dataclass
class MotionState:
    position: np.ndarray
    velocity: np.ndarray
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


class PhysXDrivenModel:
    def __init__(self, prim, mass: float = 1.0, kinematic: bool = False):
        self.prim = prim
        self.mass = float(mass)
        self.kinematic = bool(kinematic)
        self._setup_rigid_body()

    def _setup_rigid_body(self) -> None:
        rb = UsdPhysics.RigidBodyAPI.Apply(self.prim)
        rb.CreateRigidBodyEnabledAttr(True)
        if self.kinematic:
            try:
                rb.CreateKinematicEnabledAttr(True)
            except Exception:
                pass
        mass_api = UsdPhysics.MassAPI.Apply(self.prim)
        mass_api.CreateMassAttr(self.mass)
        xform = UsdGeom.Xformable(self.prim)
        if not xform.GetTranslateOp():
            xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        if not xform.GetRotateXYZOp():
            xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    def _set_pose(self, position: np.ndarray, yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> None:
        xform = UsdGeom.Xformable(self.prim)
        translate = xform.GetTranslateOp()
        rotate = xform.GetRotateXYZOp()
        if not translate:
            translate = xform.AddTranslateOp()
        if not rotate:
            rotate = xform.AddRotateXYZOp()
        translate.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        rotate.Set(Gf.Vec3f(float(roll_deg), float(pitch_deg), float(yaw_deg)))

    def step(self, dt: float, target_position: np.ndarray, target_velocity: np.ndarray) -> MotionState:
        raise NotImplementedError


class PhysXKinematicTracker(PhysXDrivenModel):
    def __init__(self, prim, mass: float = 1.0, max_speed: float = 18.0):
        super().__init__(prim, mass=mass, kinematic=True)
        self.max_speed = max(0.1, float(max_speed))
        pos, quat = _rigid_body_pose(self.prim)
        self.state = MotionState(
            position=np.array(pos, dtype=float),
            velocity=np.zeros(3, dtype=float),
            yaw_deg=_yaw_from_quat(quat),
            pitch_deg=0.0,
            roll_deg=0.0,
        )

    def step(self, dt: float, target_position: np.ndarray, target_velocity: np.ndarray) -> MotionState:
        dt = max(1e-3, float(dt))
        target_position = _np3(target_position)
        delta = target_position - self.state.position
        dist = float(np.linalg.norm(delta))
        if dist > 1e-6:
            direction = delta / dist
            step_len = min(dist, self.max_speed * dt)
            new_position = self.state.position + direction * step_len
        else:
            new_position = self.state.position.copy()
        self.state.velocity = (new_position - self.state.position) / dt
        self.state.position = new_position
        yaw = math.degrees(math.atan2(float(self.state.velocity[1]), float(self.state.velocity[0]))) if np.linalg.norm(self.state.velocity[:2]) > 1e-4 else 0.0
        pitch = max(-18.0, min(18.0, -math.degrees(math.atan2(float(self.state.velocity[2]), max(1.0, float(np.linalg.norm(self.state.velocity[:2])))))))
        roll = max(-12.0, min(12.0, float(target_velocity[1]) * 1.2))
        self._set_pose(self.state.position, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)
        return self.state


class PhysXGroundVehicle(PhysXDrivenModel):
    def __init__(self, prim, mass: float = 1000.0, max_drive_force: float = 8000.0, wheel_base: float = 3.0):
        super().__init__(prim, mass=mass, kinematic=False)
        self.max_drive_force = float(max_drive_force)
        self.wheel_base = float(wheel_base)

    def step(self, dt: float, target_position: np.ndarray, target_velocity: np.ndarray) -> MotionState:
        pos, quat = _rigid_body_pose(self.prim)
        vel = _rigid_body_velocity(self.prim)
        yaw = math.radians(_yaw_from_quat(quat))
        pos_err = _np3(target_position) - pos
        pos_err[2] = 0.0
        dist = float(np.linalg.norm(pos_err[:2]))
        desired_speed = min(10.0, dist * 0.8)
        desired_heading = math.atan2(float(pos_err[1]), float(pos_err[0])) if dist > 0.5 else yaw
        heading_error = math.atan2(math.sin(desired_heading - yaw), math.cos(desired_heading - yaw))
        desired_turn_rate = 2.5 * heading_error
        track_width = 2.0
        left_speed = desired_speed - desired_turn_rate * track_width / 2.0
        right_speed = desired_speed + desired_turn_rate * track_width / 2.0
        current_speed = float(np.linalg.norm(vel[:2]))
        speed_error = desired_speed - current_speed
        drive_force = np.clip(speed_error * 800.0, -self.max_drive_force, self.max_drive_force)
        yaw_torque = (right_speed - left_speed) * 2500.0
        cy, sy = math.cos(yaw), math.sin(yaw)
        force_world = np.array([cy * drive_force, sy * drive_force, 0.0], dtype=float)
        torque_world = np.array([0.0, 0.0, yaw_torque], dtype=float)
        apply_force(self.prim, force_world, torque_world)
        new_yaw = math.degrees(yaw + desired_turn_rate * dt)
        self._set_pose(pos, yaw_deg=new_yaw, pitch_deg=0.0, roll_deg=0.0)
        return MotionState(position=pos, velocity=vel, yaw_deg=new_yaw, pitch_deg=0.0, roll_deg=0.0)


class PhysXAerialVehicle(PhysXDrivenModel):
    def __init__(self, prim, mass: float = 10.0, max_force: float = 250.0, max_torque: float = 35.0):
        super().__init__(prim, mass=mass, kinematic=False)
        self.max_force = float(max_force)
        self.max_torque = float(max_torque)

    def step(self, dt: float, target_position: np.ndarray, target_velocity: np.ndarray) -> MotionState:
        pos, quat = _rigid_body_pose(self.prim)
        vel = _rigid_body_velocity(self.prim)
        ang = _rigid_body_angular_velocity(self.prim)
        pos_err = _np3(target_position) - pos
        vel_err = _np3(target_velocity) - vel
        kp_xy = 1.9
        kd_xy = 1.25
        kp_z = 18.0
        kd_z = 7.0
        accel_cmd = np.array(
            [
                kp_xy * pos_err[0] + kd_xy * vel_err[0],
                kp_xy * pos_err[1] + kd_xy * vel_err[1],
                kp_z * pos_err[2] + kd_z * vel_err[2] + 9.81,
            ],
            dtype=float,
        )
        force_world = self.mass * accel_cmd
        force_norm = float(np.linalg.norm(force_world))
        if force_norm > self.max_force:
            force_world *= self.max_force / force_norm
        yaw_deg = _yaw_from_quat(quat)
        desired_yaw = math.degrees(math.atan2(float(target_velocity[1]), float(target_velocity[0]))) if np.linalg.norm(target_velocity[:2]) > 1e-4 else yaw_deg
        yaw_error = math.atan2(math.sin(math.radians(desired_yaw - yaw_deg)), math.cos(math.radians(desired_yaw - yaw_deg)))
        torque_world = np.array(
            [
                np.clip(-1.4 * ang[0], -self.max_torque, self.max_torque),
                np.clip(-1.4 * ang[1], -self.max_torque, self.max_torque),
                np.clip(7.0 * yaw_error - 0.6 * ang[2], -self.max_torque, self.max_torque),
            ],
            dtype=float,
        )
        apply_force(self.prim, force_world, torque_world)
        pitch = np.clip(-math.degrees(math.atan2(float(force_world[0]), max(1.0, abs(float(force_world[2]))))), -18.0, 18.0)
        roll = np.clip(math.degrees(math.atan2(float(force_world[1]), max(1.0, abs(float(force_world[2]))))), -20.0, 20.0)
        self._set_pose(pos, yaw_deg=desired_yaw, pitch_deg=pitch, roll_deg=roll)
        return MotionState(position=pos, velocity=vel, yaw_deg=desired_yaw, pitch_deg=pitch, roll_deg=roll)


class PhysXHelicopterVehicle(PhysXAerialVehicle):
    def __init__(self, prim, mass: float = 18.0, max_force: float = 420.0, max_torque: float = 55.0):
        super().__init__(prim, mass=mass, max_force=max_force, max_torque=max_torque)

    def step(self, dt: float, target_position: np.ndarray, target_velocity: np.ndarray) -> MotionState:
        state = super().step(dt, target_position, target_velocity)
        state.pitch_deg *= 0.65
        state.roll_deg *= 0.70
        return state
