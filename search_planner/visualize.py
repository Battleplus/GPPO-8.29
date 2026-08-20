"""USD overlay helpers for SAR search path visualization in Isaac Sim."""

from __future__ import annotations

from .area import SearchArea
from .obstacles import MountainObstacle
from .planner import Waypoint


def export_waypoints_usd(
    stage,
    waypoints: list[Waypoint],
    base_path: str = "/World/AirCombat/SAR_Search",
    meters_per_unit: float = 100.0,
    path_color: tuple[float, float, float] | None = None,
    label: str = "",
) -> None:
    """Create waypoint spheres, connecting lines, and direction cones.

    Green/red spheres (safe/near-mountain), polyline, and small cones
    at each waypoint indicating heading direction.
    """
    from pxr import Gf, Sdf, UsdGeom

    _ensure_path(stage, base_path)
    line_rgb = path_color or (0.0, 0.55, 1.0)

    for i, wp in enumerate(waypoints):
        # sphere
        sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(f"{base_path}/Waypoint_{i:04d}"))
        radius = max(0.5, 4.0)
        sphere.CreateRadiusAttr(float(radius))
        UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(float(wp.x), float(wp.y), float(wp.z))
        )
        if path_color:
            color = path_color
        else:
            is_low = wp.z <= wp.terrain_z * 1.02
            color = (1.0, 0.08, 0.04) if is_low else (0.05, 1.0, 0.08)
        UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr(
            [Gf.Vec3f(*color)]
        )

    # connecting polyline
    if len(waypoints) >= 2:
        _make_polyline(
            stage,
            f"{base_path}/PathLine",
            [(wp.x, wp.y, wp.z + 1.5) for wp in waypoints],
            line_rgb,
        )

    # label sphere at start
    if len(waypoints) > 0 and label:
        wp0 = waypoints[0]
        label_s = UsdGeom.Sphere.Define(stage, Sdf.Path(f"{base_path}/StartLabel"))
        label_s.CreateRadiusAttr(8.0)
        UsdGeom.Xformable(label_s.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(float(wp0.x), float(wp0.y), float(wp0.z + 15.0))
        )
        UsdGeom.Gprim(label_s.GetPrim()).CreateDisplayColorAttr(
            [Gf.Vec3f(1.0, 1.0, 1.0)]
        )


def export_search_area_boundary(
    stage,
    area: SearchArea,
    base_path: str = "/World/AirCombat/SAR_Search",
) -> None:
    """Draw a yellow wireframe rectangle high in the air, always visible."""
    from pxr import Sdf, UsdGeom

    _ensure_path(stage, base_path)

    z = 200.0  # well above max visual terrain (~150 units)
    corners = [
        (area.x_min, area.y_min),
        (area.x_max, area.y_min),
        (area.x_max, area.y_max),
        (area.x_min, area.y_max),
        (area.x_min, area.y_min),
    ]
    pts = [(x, y, z) for x, y in corners]
    _make_polyline(stage, f"{base_path}/AreaBoundary", pts, (1.0, 0.85, 0.05))


def export_mountain_overlay(
    stage,
    mountains: list[MountainObstacle],
    base_path: str = "/World/AirCombat/SAR_Search",
) -> None:
    """Draw semi-transparent red cylinders for each mountain obstacle."""
    from pxr import Gf, Sdf, UsdGeom

    _ensure_path(stage, base_path)

    for i, m in enumerate(mountains):
        cyl = UsdGeom.Cylinder.Define(
            stage, Sdf.Path(f"{base_path}/Mountain_{i:02d}")
        )
        cyl.CreateRadiusAttr(float(m.radius_units))
        cyl.CreateHeightAttr(float(m.height_units))
        UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(
                float(m.center_x),
                float(m.center_y),
                float(m.height_units * 0.5),
            )
        )
        UsdGeom.Gprim(cyl.GetPrim()).CreateDisplayColorAttr(
            [Gf.Vec3f(1.0, 0.08, 0.04)]
        )
        UsdGeom.Gprim(cyl.GetPrim()).CreateDisplayOpacityAttr([0.22])


def export_sar_swath(
    stage,
    waypoints: list[Waypoint],
    swath_width_units: float = 50.0,
    base_path: str = "/World/AirCombat/SAR_Search",
    color: tuple[float, float, float] = (0.0, 0.45, 1.0),
) -> None:
    """Draw semi-transparent strips showing SAR ground coverage along the path.

    Each segment between consecutive waypoints gets a quad representing
    the SAR swath projected onto the ground.
    """
    from pxr import Gf, Sdf, UsdGeom

    _ensure_path(stage, base_path)
    half = swath_width_units * 0.5

    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i].x, waypoints[i].y
        x2, y2 = waypoints[i + 1].x, waypoints[i + 1].y
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            continue
        nx = -dy / length * half
        ny = dx / length * half
        z_ground = max(waypoints[i].terrain_z, waypoints[i + 1].terrain_z) + 8.0

        quad = UsdGeom.Mesh.Define(
            stage, Sdf.Path(f"{base_path}/Swath_{i:04d}")
        )
        swath_pts = [
            Gf.Vec3f(float(x1 + nx), float(y1 + ny), float(z_ground)),
            Gf.Vec3f(float(x1 - nx), float(y1 - ny), float(z_ground)),
            Gf.Vec3f(float(x2 - nx), float(y2 - ny), float(z_ground)),
            Gf.Vec3f(float(x2 + nx), float(y2 + ny), float(z_ground)),
        ]
        quad.CreatePointsAttr(swath_pts)
        quad.CreateFaceVertexCountsAttr([4])
        quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        quad.CreateSubdivisionSchemeAttr("none")
        UsdGeom.Gprim(quad.GetPrim()).CreateDisplayColorAttr(
            [Gf.Vec3f(*color)]
        )
        UsdGeom.Gprim(quad.GetPrim()).CreateDisplayOpacityAttr([0.25])


def export_uav_platform(
    stage,
    x: float, y: float, z: float,
    base_path: str = "/World/AirCombat/SAR_Search",
) -> None:
    """Place a simple UAV marker (cross + beacon) at the given position."""
    from pxr import Gf, Sdf, UsdGeom

    _ensure_path(stage, base_path)

    # Beacon sphere
    beacon = UsdGeom.Sphere.Define(stage, Sdf.Path(f"{base_path}/UAV_Beacon"))
    beacon.CreateRadiusAttr(2.0)
    UsdGeom.Xformable(beacon.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(float(x), float(y), float(z))
    )
    UsdGeom.Gprim(beacon.GetPrim()).CreateDisplayColorAttr(
        [Gf.Vec3f(1.0, 1.0, 1.0)]
    )

    # Cross arms
    arm_len = 5.0
    _make_polyline(
        stage, f"{base_path}/UAV_CrossH",
        [(x - arm_len, y, z), (x + arm_len, y, z)],
        (1.0, 1.0, 1.0),
    )
    _make_polyline(
        stage, f"{base_path}/UAV_CrossV",
        [(x, y - arm_len, z), (x, y + arm_len, z)],
        (1.0, 1.0, 1.0),
    )

    # Vertical drop line to the terrain below
    ground_z = z - 5.0  # slightly below
    _make_polyline(
        stage, f"{base_path}/UAV_DropLine",
        [(x, y, z - 2.0), (x, y, ground_z)],
        (1.0, 1.0, 1.0),
    )


# ── internal helpers ──────────────────────────────────────────


def _ensure_path(stage, path: str) -> None:
    from pxr import Sdf, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        UsdGeom.Xform.Define(stage, Sdf.Path(path))


def _make_polyline(
    stage,
    path: str,
    pts: list[tuple[float, float, float]],
    color: tuple[float, float, float],
) -> None:
    from pxr import Gf, Sdf, UsdGeom

    if len(pts) < 2:
        return
    gfx_pts = [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in pts]
    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path(path))
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr([len(gfx_pts)])
    curves.CreatePointsAttr(gfx_pts)
    curves.CreateWidthsAttr([2.0] * len(gfx_pts))
    curves.CreateDisplayColorAttr([Gf.Vec3f(*color)])
