#!/usr/bin/env python3
"""Overlay synchronized multi-UAV mission tracks on an Isaac camera frame."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


COLORS = (
    (25, 225, 255),
    (255, 55, 190),
    (55, 245, 105),
    (255, 215, 35),
)
SHORT_NAMES = ("UAV-1", "UAV-2", "UAV-3", "UAV-4")


def load_rows(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for field in ("x", "y", "z", "vx", "vy", "vz", "time_s"):
                if field in row:
                    row[field] = float(row[field])
            if "frame" in row:
                row["frame"] = int(row["frame"])
            grouped[str(row["platform"])].append(row)
    return dict(grouped)


class CameraProjection:
    def __init__(self, eye, target, width: int, height: int) -> None:
        self.eye = np.asarray(eye, dtype=float)
        forward = np.asarray(target, dtype=float) - self.eye
        self.forward = forward / np.linalg.norm(forward)
        right = np.cross(self.forward, np.array([0.0, 0.0, 1.0]))
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.right, self.forward)
        self.focal_px = width * 24.0 / 20.955
        self.cx = width * 0.5
        self.cy = height * 0.5

    def point(self, xyz) -> tuple[int, int]:
        delta = np.asarray(xyz, dtype=float) - self.eye
        depth = max(1e-6, float(np.dot(delta, self.forward)))
        x = self.cx + self.focal_px * float(np.dot(delta, self.right)) / depth
        y = self.cy - self.focal_px * float(np.dot(delta, self.up)) / depth
        return round(x), round(y)


def font(path: str, size: int):
    return ImageFont.truetype(path, size=size)


def alpha_box(draw, xy, fill, outline=None, width=1, radius=6):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_glow_line(layer, points, color, width: int = 4) -> None:
    if len(points) < 2:
        return
    glow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.line(points, fill=(*color, 150), width=width + 8, joint="curve")
    glow = glow.filter(ImageFilter.GaussianBlur(5))
    layer.alpha_composite(glow)
    ImageDraw.Draw(layer).line(
        points, fill=(*color, 245), width=width, joint="curve"
    )


def triangle(center, previous, size=10):
    dx = center[0] - previous[0]
    dy = center[1] - previous[1]
    angle = math.atan2(dy, dx) if abs(dx) + abs(dy) > 0.1 else 0.0
    result = []
    for offset, radius in ((0.0, size + 3), (2.45, size), (-2.45, size)):
        result.append((
            center[0] + math.cos(angle + offset) * radius,
            center[1] + math.sin(angle + offset) * radius,
        ))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", required=True)
    parser.add_argument("--camera-manifest", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    args = parser.parse_args()

    background = Image.open(args.background).convert("RGB")
    width, height = background.size
    background = ImageEnhance.Contrast(background).enhance(0.92)
    background = ImageEnhance.Brightness(background).enhance(0.72)
    manifest = json.loads(Path(args.camera_manifest).read_text(encoding="utf-8"))
    mission = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    trajectories = load_rows(Path(args.trajectory))
    plans = load_rows(Path(args.plan))
    platforms = list(trajectories)
    frame_count = min(len(rows) for rows in trajectories.values())
    projection = CameraProjection(
        manifest["camera_eye"], manifest["camera_target"], width, height
    )
    output_dir = Path(args.frames_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    bold_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    title_font = font(bold_path, 27)
    body_font = font(font_path, 16)
    small_font = font(font_path, 14)
    label_font = font(bold_path, 15)

    projected_plans: dict[str, list[tuple[int, int]]] = {}
    search_regions: dict[str, list[tuple[int, int]]] = {}
    for platform in platforms:
        projected_plans[platform] = [
            projection.point((row["x"], row["y"], row["z"]))
            for row in plans[platform]
        ]
        search = [row for row in plans[platform] if row["phase"] == "search"]
        xs = [float(row["x"]) for row in search]
        ys = [float(row["y"]) for row in search]
        z = float(search[0]["z"])
        search_regions[platform] = [
            projection.point((min(xs), min(ys), z)),
            projection.point((max(xs), min(ys), z)),
            projection.point((max(xs), max(ys), z)),
            projection.point((min(xs), max(ys), z)),
        ]

    end_frame = frame_count if args.end_frame is None else min(
        frame_count, args.end_frame
    )
    for frame in range(max(0, args.start_frame), end_frame):
        image = background.copy().convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        draw.rectangle((0, 0, width, 66), fill=(7, 13, 18, 235))
        draw.text((24, 15), "多无人机协同任务分配与区域搜索", font=title_font, fill=(245, 248, 250, 255))
        phase = trajectories[platforms[0]][frame]["phase"]
        phase_cn = "并行转场" if phase == "transit" else "目标区域搜索"
        phase_color = (75, 200, 255, 255) if phase == "transit" else (95, 245, 145, 255)
        draw.text((995, 20), f"阶段：{phase_cn}", font=body_font, fill=phase_color)

        for index, platform in enumerate(platforms):
            color = COLORS[index]
            region = search_regions[platform]
            draw.polygon(region, fill=(*color, 24), outline=(*color, 150))
            center_x = round(sum(point[0] for point in region) / len(region))
            center_y = round(sum(point[1] for point in region) / len(region))
            draw.text(
                (center_x - 38, center_y - 10),
                mission["assignments"][index]["task_id"],
                font=small_font,
                fill=(*color, 205),
            )
            draw.line(
                projected_plans[platform], fill=(*color, 90), width=2, joint="curve"
            )

        for index, platform in enumerate(platforms):
            color = COLORS[index]
            rows = trajectories[platform]
            trail = [
                projection.point((row["x"], row["y"], row["z"]))
                for row in rows[: frame + 1]
            ]
            draw_glow_line(overlay, trail, color)
            current = trail[-1]
            previous = trail[-2] if len(trail) > 1 else (current[0] - 1, current[1])
            if phase == "search":
                pulse = 16 + round(5 * math.sin(frame * 0.22 + index))
                draw.ellipse(
                    (current[0] - pulse, current[1] - pulse,
                     current[0] + pulse, current[1] + pulse),
                    outline=(*color, 120), width=2,
                )
            draw.polygon(
                triangle(current, previous), fill=(*color, 255),
                outline=(245, 250, 255, 255),
            )
            alpha_box(
                draw,
                (current[0] + 12, current[1] - 13,
                 current[0] + 77, current[1] + 12),
                (4, 9, 13, 215),
                (*color, 210),
            )
            draw.text(
                (current[0] + 19, current[1] - 10), SHORT_NAMES[index],
                font=label_font, fill=(250, 252, 253, 255),
            )

        panel_x = 900
        alpha_box(draw, (panel_x, 82, 1257, 240), (6, 12, 17, 218), (125, 145, 155, 130))
        draw.text((panel_x + 16, 94), "任务分配", font=label_font, fill=(245, 248, 250, 255))
        for index, assignment in enumerate(mission["assignments"]):
            y = 124 + index * 27
            color = COLORS[index]
            draw.ellipse((panel_x + 16, y + 4, panel_x + 26, y + 14), fill=(*color, 255))
            draw.text(
                (panel_x + 36, y),
                f"{SHORT_NAMES[index]}  {assignment['task_id']}  {assignment['sensor']}",
                font=small_font,
                fill=(220, 229, 233, 255),
            )

        progress = frame / max(1, frame_count - 1)
        draw.rectangle((20, height - 38, width - 20, height - 18), fill=(5, 10, 14, 215))
        draw.rectangle(
            (23, height - 35, 23 + round((width - 46) * progress), height - 21),
            fill=phase_color,
        )
        draw.text(
            (24, height - 62),
            f"T+{frame / 15.0:04.1f}s  |  4机并行  |  {phase_cn}",
            font=body_font,
            fill=(238, 242, 245, 255),
        )

        image.alpha_composite(overlay)
        output = output_dir / f"frame_{frame:04d}.png"
        image.convert("RGB").save(output, optimize=False)
        if frame == 0 or (frame + 1) % 40 == 0:
            print(f"TACTICAL_RENDER frame={frame + 1}/{frame_count} phase={phase}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
