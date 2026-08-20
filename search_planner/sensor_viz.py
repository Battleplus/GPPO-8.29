"""Real-time sensor FOV and coverage-map visualization for SAR drones.

Provides:
  - SensorFOV:  two visual layers per drone:
      1. EO triangle — 2D red gradient triangle on ground plane (forward-looking)
      2. SAR swath  — 3D yellow gradient footprint mapped to terrain surface
  - CoverageMap: tracks which ground cells have been scanned.

Usage (per frame):
    fov = SensorFOV(stage, base_path, sensor_spec, terrain_fn, mpu)
    fov.update(drone_pos, drone_yaw_deg)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ──────────────────────────────────────────────────────────────
#  Sensor FOV parameters
# ──────────────────────────────────────────────────────────────


@dataclass
class SensorFOVParams:
    """Convenience wrapper around a sensor's FOV-relevant fields."""

    hfov_deg: float = 65.0
    vfov_deg: float = 50.0
    max_range_km: float = 8.0
    boresight_pitch_deg: float = -32.0
    boresight_yaw_offset_deg: float = 0.0

    @classmethod
    def from_spec(cls, spec) -> "SensorFOVParams":
        return cls(
            hfov_deg=spec.azimuth_fov_deg,
            vfov_deg=spec.elevation_fov_deg,
            max_range_km=spec.max_range_km,
            boresight_pitch_deg=spec.boresight_pitch_deg,
            boresight_yaw_offset_deg=spec.boresight_yaw_offset_deg,
        )


# ──────────────────────────────────────────────────────────────
#  Sensor FOV
# ──────────────────────────────────────────────────────────────


class SensorFOV:
    """Two-layer sensor visualisation per drone.

    Layer 1 — EO triangle: 2D red gradient fan on a horizontal plane,
              showing the forward-looking instantaneous FOV.

    Layer 2 — SAR swath: 3D yellow gradient mesh mapped to the terrain
              surface, showing the SAR scan footprint (top-down ≈ rectangle).
    """

    def __init__(
        self,
        stage,
        base_path: str,
        eo_params: SensorFOVParams,
        sar_params: SensorFOVParams,
        terrain_fn,
        meters_per_unit: float = 100.0,
        map_size_units: float = 3000.0,
        height_scale: float = 10.0,
    ):
        self._stage = stage
        self._base = base_path
        self._eo = eo_params
        self._sar = sar_params
        self._terrain = terrain_fn
        self._mpu = meters_per_unit
        self._map_size = map_size_units
        self._height_scale = height_scale
        self._eo_range = eo_params.max_range_km * 1000.0 / meters_per_unit
        self._sar_range = sar_params.max_range_km * 1000.0 / meters_per_unit
        self._trail: list[tuple[float, float, float, float]] = []  # (x,y,z,yaw)
        self._trail_max = 30  # max trail samples

        self._ensure_paths()

    # ── public API ────────────────────────────────────────────

    def update(self, x: float, y: float, z: float, yaw_deg: float):
        """Recompute and redraw both FOV layers."""
        # ── EO triangle (2D, drone altitude, parallel to ground) ──
        eo_pts = self._compute_eo_triangle(x, y, z, yaw_deg, self._eo, self._eo_range)
        if eo_pts is not None:
            self._draw_eo_triangle(eo_pts)

        # ── SAR trailing swath (distance-based sampling) ──
        min_step = 3.0  # minimum distance between trail samples
        if self._trail:
            last = self._trail[-1]
            dist = math.hypot(x - last[0], y - last[1])
        else:
            dist = float("inf")
        if dist >= min_step:
            self._trail.append((x, y, z, yaw_deg))
            if len(self._trail) > self._trail_max:
                self._trail.pop(0)
        self._draw_sar_trail()

    # ── EO: 2D triangle at drone altitude, parallel to ground ─

    def _compute_eo_triangle(
        self, x, y, z, yaw_deg, p: SensorFOVParams, max_range,
    ) -> list[tuple[float, float, float]] | None:
        """2D triangle at drone altitude: apex at drone nose, base always
        forward and perpendicular to heading, height = 50% of max_range."""
        hh = math.radians(p.hfov_deg / 2.0)
        yaw = math.radians(yaw_deg + p.boresight_yaw_offset_deg)

        fwd_x = math.cos(yaw)
        fwd_y = math.sin(yaw)
        right_x = -fwd_y  # perpendicular, counterclockwise
        right_y = fwd_x

        # Apex slightly ahead of drone, base further forward
        apex_dist = 5.0  # small offset from drone center
        height = max_range * 0.5
        hw = height * math.tan(hh)

        pts = [
            (x + fwd_x * apex_dist, y + fwd_y * apex_dist, z),  # apex
            (x + fwd_x * height + right_x * hw, y + fwd_y * height + right_y * hw, z),  # base-left
            (x + fwd_x * height - right_x * hw, y + fwd_y * height - right_y * hw, z),  # base-right
        ]
        return pts

    def _draw_eo_triangle(self, pts):
        """Flat red triangle, 2D, parallel to ground, at drone altitude."""
        from pxr import Gf, Sdf, UsdGeom

        mesh_path = f"{self._base}/EO_Triangle/Mesh"
        prim = self._stage.GetPrimAtPath(mesh_path)
        if prim and prim.IsValid():
            self._stage.RemovePrim(mesh_path)

        gpts = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts]
        mesh = UsdGeom.Mesh.Define(self._stage, Sdf.Path(mesh_path))
        mesh.CreatePointsAttr(gpts)
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        mesh.CreateSubdivisionSchemeAttr("none")

        primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        primvars.CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, "vertex",
        ).Set([Gf.Vec3f(1.0, 0.08, 0.06)] * 3)
        primvars.CreatePrimvar(
            "displayOpacity", Sdf.ValueTypeNames.FloatArray, "vertex",
        ).Set([0.40, 0.05, 0.05])

    # ── SAR: curved trailing ribbon along flight path ─────────

    def _draw_sar_trail(self):
        """Build a curved ribbon mesh from the trail buffer.  Each trail
        sample produces a left/right cross-section; consecutive sections
        are connected into a triangle strip.  Front (current) = dark,
        back (oldest) = light → visible gradient along the trail."""
        from pxr import Gf, Sdf, UsdGeom

        if len(self._trail) < 2:
            return

        half_w = self._sar_range * 0.20  # half-width of the swath

        gpts = []
        colors = []
        n = len(self._trail)

        for i, (tx, ty, _tz, tyaw) in enumerate(self._trail):
            r_rad = math.radians(tyaw)
            rx = -math.sin(r_rad)
            ry = math.cos(r_rad)

            # Terrain height at each edge point (not just center)
            lx, ly = tx + rx * half_w, ty + ry * half_w
            rx2, ry2 = tx - rx * half_w, ty - ry * half_w
            lz = self._terrain(lx, ly, self._map_size, self._height_scale) + 12.0
            rz = self._terrain(rx2, ry2, self._map_size, self._height_scale) + 12.0

            gpts.append(Gf.Vec3f(float(lx), float(ly), float(lz)))
            gpts.append(Gf.Vec3f(float(rx2), float(ry2), float(rz)))

            # Gradient: front (newest) dark yellow, back (oldest) washed out
            t = i / max(1, n - 1)  # 0=oldest, 1=newest
            r = 1.0
            g = 0.90 - 0.25 * t   # newest=0.65, oldest=0.90
            b = 0.30 - 0.25 * t    # newest=0.05, oldest=0.30
            colors.append(Gf.Vec3f(float(r), float(g), float(b)))
            colors.append(Gf.Vec3f(float(r), float(g), float(b)))

        # Triangle strip: connect consecutive cross-sections
        vi = []
        fc = []
        for i in range(n - 1):
            a = i * 2        # left of section i
            b = i * 2 + 1    # right of section i
            c2 = (i + 1) * 2      # left of section i+1
            d = (i + 1) * 2 + 1   # right of section i+1
            vi.extend([a, b, c2, b, d, c2])
            fc.extend([3, 3])

        mesh_path = f"{self._base}/SAR_Swath/Mesh"
        prim = self._stage.GetPrimAtPath(mesh_path)
        if prim and prim.IsValid():
            self._stage.RemovePrim(mesh_path)

        mesh = UsdGeom.Mesh.Define(self._stage, Sdf.Path(mesh_path))
        mesh.CreatePointsAttr(gpts)
        mesh.CreateFaceVertexCountsAttr(fc)
        mesh.CreateFaceVertexIndicesAttr(vi)
        mesh.CreateSubdivisionSchemeAttr("none")

        pv = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, "vertex").Set(colors)
        # Front (i large) = opaque, back (i=0) = fully transparent
        pv.CreatePrimvar("displayOpacity", Sdf.ValueTypeNames.FloatArray, "vertex").Set(
            [0.45 * (i / max(1, 2 * n - 1)) for i in range(2 * n)]
        )

    # ── paths ────────────────────────────────────────────────

    def _ensure_paths(self):
        from pxr import Sdf, UsdGeom

        parts = self._base.strip("/").split("/")
        for i in range(1, len(parts) + 2):
            p = "/" + "/".join(parts[:i])
            prim = self._stage.GetPrimAtPath(p)
            if not prim or not prim.IsValid():
                UsdGeom.Xform.Define(self._stage, Sdf.Path(p))
        for sub in ("EO_Triangle", "SAR_Swath"):
            p = f"{self._base}/{sub}"
            prim = self._stage.GetPrimAtPath(p)
            if not prim or not prim.IsValid():
                UsdGeom.Xform.Define(self._stage, Sdf.Path(p))


# ──────────────────────────────────────────────────────────────
#  Coverage Map
# ──────────────────────────────────────────────────────────────


@dataclass
class CoverageMap:
    """Tracks which ground cells have been scanned."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution_km: float = 0.5

    grid: np.ndarray = field(init=False)
    cols: int = field(init=False)
    rows: int = field(init=False)

    def __post_init__(self):
        res_units = self.resolution_km * 10.0
        self.cols = max(1, int((self.x_max - self.x_min) / res_units) + 1)
        self.rows = max(1, int((self.y_max - self.y_min) / res_units) + 1)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)

    def mark_footprint(self, pts: list[tuple[float, float, float]]):
        if len(pts) < 3:
            return
        res = self.resolution_km * 10.0
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx_min = max(0, int((min(xs) - self.x_min) / res))
        cx_max = min(self.cols - 1, int((max(xs) - self.x_min) / res + 1))
        cy_min = max(0, int((min(ys) - self.y_min) / res))
        cy_max = min(self.rows - 1, int((max(ys) - self.y_min) / res + 1))
        for cy in range(cy_min, cy_max + 1):
            for cx in range(cx_min, cx_max + 1):
                cell_x = self.x_min + (cx + 0.5) * res
                cell_y = self.y_min + (cy + 0.5) * res
                if _point_in_poly(cell_x, cell_y, xs, ys):
                    self.grid[cy, cx] = 1

    @property
    def coverage_pct(self) -> float:
        return float(np.mean(self.grid)) * 100.0

    def create_prims(self, stage, base_path: str):
        from pxr import Sdf, UsdGeom
        from .visualize import _ensure_path

        bp = f"{base_path}/Coverage"
        _ensure_path(stage, bp)
        self._prims_path = bp
        self._prims_res = self.resolution_km * 10.0
        self._prims_stage = stage

    def update_prims(self):
        if not hasattr(self, "_prims_stage"):
            return
        from pxr import Gf, Sdf, UsdGeom
        from .visualize import _ensure_path

        stage = self._prims_stage
        bp = self._prims_path
        res = self._prims_res
        rows, cols = self.rows, self.cols

        quads = []
        for cy in range(rows):
            for cx in range(cols):
                if self.grid[cy, cx]:
                    x0 = self.x_min + cx * res
                    y0 = self.y_min + cy * res
                    quads.append((x0, y0, x0 + res, y0 + res))

        if len(quads) > 2048:
            quads = quads[:: max(1, len(quads) // 2048)]

        prim = stage.GetPrimAtPath(f"{bp}/Quads")
        if prim and prim.IsValid():
            stage.RemovePrim(f"{bp}/Quads")

        if not quads:
            return

        _ensure_path(stage, f"{bp}/Quads")
        all_pts, all_faces, all_vc = [], [], []
        idx = 0
        for (x0, y0, x1, y1) in quads:
            z = 8.0
            all_pts.extend([
                Gf.Vec3f(float(x0), float(y0), float(z)),
                Gf.Vec3f(float(x1), float(y0), float(z)),
                Gf.Vec3f(float(x1), float(y1), float(z)),
                Gf.Vec3f(float(x0), float(y1), float(z)),
            ])
            all_faces.extend([idx, idx + 1, idx + 2, idx + 3])
            all_vc.append(4)
            idx += 4

        mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(f"{bp}/Quads"))
        mesh.CreatePointsAttr(all_pts)
        mesh.CreateFaceVertexCountsAttr(all_vc)
        mesh.CreateFaceVertexIndicesAttr(all_faces)
        mesh.CreateSubdivisionSchemeAttr("none")
        UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorAttr(
            [Gf.Vec3f(1.0, 0.85, 0.05)]
        )
        UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayOpacityAttr([0.12])


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────


def _point_in_poly(px, py, xs, ys):
    inside = False
    n = len(xs)
    j = n - 1
    for i in range(n):
        if ((ys[i] > py) != (ys[j] > py)) and (
            px < (xs[j] - xs[i]) * (py - ys[i]) / (ys[j] - ys[i]) + xs[i]
        ):
            inside = not inside
        j = i
    return inside
