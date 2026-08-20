from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from . import anti_radiation, eo, sar
from .io import save_rgb


PROCESSORS = {
    "eo": eo.process,
    "sar": sar.process,
    "arm": anti_radiation.process,
}


DEFAULT_SENSOR_OFFSETS = {
    "eo": (0.0, 0.0, -0.22),
    "sar": (0.28, 0.0, -0.22),
    "arm": (-0.28, 0.0, -0.22),
}


def normalize_sensor_names(sensor_names: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if sensor_names is None:
        return list(DEFAULT_SENSOR_OFFSETS)
    if isinstance(sensor_names, str):
        raw_names = [name.strip().lower() for name in sensor_names.split(",")]
    else:
        raw_names = [str(name).strip().lower() for name in sensor_names]
    names = [name for name in raw_names if name]
    unknown = [name for name in names if name not in PROCESSORS]
    if unknown:
        raise ValueError(f"Unknown sensor name(s): {', '.join(unknown)}")
    return names


def create_sensor_cameras(
    stage,
    drone_path: str,
    altitude: float,
    coverage: float,
    sensor_names: str | list[str] | tuple[str, ...] | None = None,
    offsets: dict[str, tuple[float, float, float]] | None = None,
    clipping_range: tuple[float, float] = (0.1, 2000.0),
) -> dict[str, str]:
    from pxr import Gf, UsdGeom

    aperture = 20.955
    focal_length = aperture / (2.0 * math.tan(math.atan((coverage * 0.5) / max(1.0, altitude))))
    selected_names = normalize_sensor_names(sensor_names)
    merged_offsets = dict(DEFAULT_SENSOR_OFFSETS)
    if offsets:
        merged_offsets.update(offsets)
    paths = {}
    for name in selected_names:
        offset = merged_offsets[name]
        path = f"{drone_path}/Sensor_{name.upper()}"
        cam = UsdGeom.Camera.Define(stage, path)
        cam.CreateFocalLengthAttr(float(focal_length))
        cam.CreateHorizontalApertureAttr(aperture)
        cam.CreateVerticalApertureAttr(aperture)
        cam.CreateClippingRangeAttr(Gf.Vec2f(float(clipping_range[0]), float(clipping_range[1])))
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(float(offset[0]), float(offset[1]), float(offset[2])))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        paths[name] = path
    print(f"[QL] Sensors ready: {', '.join(name.upper() for name in selected_names)} cameras attached to {drone_path}")
    return paths


def setup_annotators(sensor_paths: dict[str, str], resolution: int | tuple[int, int]):
    import omni.replicator.core as rep

    if isinstance(resolution, int):
        render_resolution = (resolution, resolution)
    else:
        render_resolution = (int(resolution[0]), int(resolution[1]))
    annotators = {}
    for name, path in sensor_paths.items():
        rp = rep.create.render_product(path, resolution=render_resolution)
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rp])
        annotators[name] = annot
    return annotators


def capture(annotators: dict, output_dir: Path, frame_idx: int, process_params: dict[str, dict[str, Any]] | None = None) -> None:
    process_params = process_params or {}
    for name, annot in annotators.items():
        data = annot.get_data()
        if data is None or data.size == 0:
            print(f"[QL][WARN] {name.upper()} sensor returned no image")
            continue
        rgb = data[:, :, :3] if data.shape[2] >= 3 else data
        processed = PROCESSORS[name](rgb, **process_params.get(name, {}))
        save_rgb(processed, output_dir / f"{frame_idx:03d}_{name}.png")
    print(f"[QL] Sensor capture saved: {output_dir} frame={frame_idx:03d}")
