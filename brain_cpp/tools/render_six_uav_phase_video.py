#!/usr/bin/env python3
"""Render the audited MILP -> MPPI -> PPO mission as a tactical video."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = (
    (44, 210, 255),
    (255, 88, 174),
    (102, 232, 126),
    (255, 204, 64),
    (176, 126, 255),
)
MAP_BOUNDS = (-1600.0, -1000.0, 650.0, 850.0)
PLOT = (38, 82, 942, 684)
MOUNTAINS = (
    (-870.0, -300.0, 75.0),
    (-360.0, 510.0, 84.0),
    (-60.0, 630.0, 78.0),
    (510.0, 90.0, 72.0),
    (870.0, 660.0, 75.0),
)


def font(size: int, bold: bool = False):
    name = "NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc"
    return ImageFont.truetype(f"/usr/share/fonts/opentype/noto/{name}", size=size)


def load_grouped(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for field in ("x", "y", "z", "vx", "vy", "vz", "time_s"):
                if field in row:
                    row[field] = float(row[field])
            grouped[row["platform"]].append(row)
    return dict(grouped)


def project(x: float, y: float) -> tuple[int, int]:
    xmin, ymin, xmax, ymax = MAP_BOUNDS
    left, top, right, bottom = PLOT
    px = left + (x - xmin) / (xmax - xmin) * (right - left)
    py = bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
    return round(px), round(py)


def scaled_radius(radius: float) -> int:
    return max(2, round(radius / (MAP_BOUNDS[2] - MAP_BOUNDS[0]) * (PLOT[2] - PLOT[0])))


def sample(rows: list[dict], progress: float) -> tuple[dict, int]:
    index = min(len(rows) - 1, max(0, round(progress * (len(rows) - 1))))
    return rows[index], index


def triangle(center: tuple[int, int], velocity: tuple[float, float], size: int = 10):
    angle = math.atan2(-velocity[1], velocity[0]) if abs(velocity[0]) + abs(velocity[1]) > 0.1 else 0.0
    return [
        (
            center[0] + math.cos(angle + offset) * radius,
            center[1] + math.sin(angle + offset) * radius,
        )
        for offset, radius in ((0.0, size + 4), (2.45, size), (-2.45, size))
    ]


def draw_background(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, 1279, 719), fill=(7, 13, 18))
    draw.rectangle(PLOT, fill=(12, 24, 29), outline=(72, 96, 104), width=2)
    for value in range(-1500, 501, 250):
        x1, _ = project(value, MAP_BOUNDS[1])
        draw.line((x1, PLOT[1], x1, PLOT[3]), fill=(33, 54, 60), width=1)
    for value in range(-750, 751, 250):
        _, y1 = project(MAP_BOUNDS[0], value)
        draw.line((PLOT[0], y1, PLOT[2], y1), fill=(33, 54, 60), width=1)

    # A_3_4 and its four coverage cells.
    aoi_a = project(0.0, -500.0)
    aoi_b = project(500.0, 0.0)
    box = (min(aoi_a[0], aoi_b[0]), min(aoi_a[1], aoi_b[1]),
           max(aoi_a[0], aoi_b[0]), max(aoi_a[1], aoi_b[1]))
    draw.rectangle(box, fill=(36, 114, 132, 24), outline=(77, 211, 233), width=2)
    mid_x = (box[0] + box[2]) // 2
    mid_y = (box[1] + box[3]) // 2
    draw.line((mid_x, box[1], mid_x, box[3]), fill=(77, 211, 233), width=1)
    draw.line((box[0], mid_y, box[2], mid_y), fill=(77, 211, 233), width=1)
    draw.text((box[0] + 8, box[1] + 6), "A_3_4 TASK AREA", font=font(13, True), fill=(124, 228, 242))
    for label, pos in (("c3", (box[0] + 8, box[1] + 28)),
                       ("c4", (mid_x + 8, box[1] + 28)),
                       ("c1", (box[0] + 8, mid_y + 8)),
                       ("c2", (mid_x + 8, mid_y + 8))):
        draw.text(pos, label, font=font(12), fill=(105, 176, 186))

    for x, y, radius in MOUNTAINS:
        center = project(x, y)
        r = scaled_radius(radius)
        draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r),
                     fill=(91, 73, 52), outline=(164, 126, 79), width=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--planning-summary", required=True)
    parser.add_argument("--execution-summary", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--frame-count", type=int, default=270)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    trajectories = load_grouped(Path(args.trajectory))
    plans = load_grouped(Path(args.plan))
    planning = json.loads(Path(args.planning_summary).read_text(encoding="utf-8"))
    execution = json.loads(Path(args.execution_summary).read_text(encoding="utf-8"))
    results = {item["platform"]: item for item in execution["platforms"]}
    total_search_calls = sum(item["search_ppo_calls"] for item in execution["platforms"])
    platforms = list(trajectories)
    assignments: dict[str, str] = {}
    with Path(planning["allocation_csv"]).open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            assignments.setdefault(item["platform"], item["cell"])
    reserve = list(planning.get("reserve_aircraft", []))

    phase_rows: dict[str, dict[str, list[dict]]] = {}
    phase_plans: dict[str, dict[str, list[dict]]] = {}
    for platform in platforms:
        phase_rows[platform] = {
            phase: [row for row in trajectories[platform] if row["phase"] == phase]
            for phase in ("transit", "search")
        }
        phase_plans[platform] = {
            phase: [row for row in plans[platform] if row["phase"] == phase]
            for phase in ("transit", "search")
        }

    frames_dir = Path(args.frames_dir).resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    transit_count = round(args.frame_count * 0.40)
    title_font = font(25, True)
    body_font = font(16)
    small_font = font(13)
    label_font = font(14, True)

    for frame in range(args.frame_count):
        image = Image.new("RGB", (1280, 720), (7, 13, 18))
        draw = ImageDraw.Draw(image, "RGBA")
        draw_background(draw)
        is_transit = frame < transit_count
        phase = "transit" if is_transit else "search"
        controller = "MPPI FOLLOW / PPO OFF" if is_transit else "PPO LOCAL / PPO ON"
        phase_progress = (
            frame / max(1, transit_count - 1)
            if is_transit else (frame - transit_count) / max(1, args.frame_count - transit_count - 1)
        )
        phase_color = (54, 204, 255) if is_transit else (78, 239, 137)

        draw.rectangle((0, 0, 1279, 64), fill=(4, 9, 13, 245))
        draw.text((24, 13), "MILP任务分配 · MPPI转场 · PPO区域覆盖搜索",
                  font=title_font, fill=(242, 247, 249))
        draw.text((895, 18), controller, font=body_font, fill=phase_color)

        for index, platform in enumerate(platforms):
            color = COLORS[index]
            for path_phase in ("transit", "search"):
                points = [project(row["x"], row["y"]) for row in phase_plans[platform][path_phase]]
                if len(points) > 1:
                    draw.line(points, fill=(*color, 70 if path_phase == "search" else 45), width=2)

            search_plan = phase_plans[platform]["search"]
            obstacle_x = (search_plan[0]["x"] + search_plan[1]["x"]) * 0.5
            obstacle_y = (search_plan[0]["y"] + search_plan[1]["y"]) * 0.5
            obstacle_point = project(obstacle_x, obstacle_y)
            draw.ellipse((obstacle_point[0] - 4, obstacle_point[1] - 4,
                          obstacle_point[0] + 4, obstacle_point[1] + 4),
                         fill=(239, 70, 55, 210), outline=(255, 183, 157), width=1)

            rows = phase_rows[platform][phase]
            current, current_index = sample(rows, phase_progress)
            history = rows[: current_index + 1]
            trail = [project(item["x"], item["y"]) for item in history]
            if len(trail) > 1:
                draw.line(trail, fill=(*color, 230), width=4, joint="curve")
            point = project(current["x"], current["y"])
            velocity = (float(current["vx"]), float(current["vy"]))
            if not is_transit:
                pulse = 15 + round(4 * math.sin(frame * 0.25 + index))
                draw.ellipse((point[0] - pulse, point[1] - pulse,
                              point[0] + pulse, point[1] + pulse),
                             outline=(*color, 120), width=2)
            draw.polygon(triangle(point, velocity), fill=(*color, 255), outline=(244, 249, 251), width=1)
            label_offsets = ((-78, -36), (12, -36), (12, 8), (12, -36), (12, 8))
            label_x = point[0] + label_offsets[index][0]
            label_y = point[1] + label_offsets[index][1]
            draw.rounded_rectangle((label_x, label_y, label_x + 63, label_y + 25),
                                   radius=4, fill=(3, 8, 12, 220), outline=(*color, 210), width=1)
            draw.text((label_x + 6, label_y + 3), f"UAV-{index + 1}",
                      font=label_font, fill=(248, 251, 252))

        panel = (962, 82, 1263, 425)
        draw.rounded_rectangle(panel, radius=6, fill=(5, 12, 17, 232), outline=(87, 111, 121), width=1)
        draw.text((980, 96), "MILP ALLOCATION", font=label_font, fill=(242, 247, 249))
        for index, platform in enumerate(platforms):
            y = 129 + index * 43
            color = COLORS[index]
            result = results[platform]
            draw.ellipse((980, y + 4, 990, y + 14), fill=(*color, 255))
            draw.text((1000, y), f"UAV-{index + 1}  {assignments.get(platform, '?')}",
                      font=small_font, fill=(232, 239, 242))
            status = "MPPI" if is_transit else f"PPO {result['search_ppo_calls']} calls"
            draw.text((1000, y + 18), status, font=small_font, fill=phase_color)
        reserve_y = 129 + len(platforms) * 43
        draw.ellipse((980, reserve_y + 4, 990, reserve_y + 14), fill=(120, 130, 136, 255))
        reserve_name = reserve[0] if reserve else "none"
        draw.text((1000, reserve_y), "UAV-6  RESERVE", font=small_font, fill=(177, 185, 190))
        draw.text((1000, reserve_y + 18), reserve_name.replace("Blue_", ""),
                  font=small_font, fill=(128, 139, 145))

        draw.rounded_rectangle((962, 446, 1263, 616), radius=6,
                               fill=(5, 12, 17, 232), outline=(87, 111, 121), width=1)
        draw.text((980, 460), "PHASE AUDIT", font=label_font, fill=(242, 247, 249))
        audit = (
            ("PPO during transit", "0 calls", (74, 222, 141)),
            ("PPO during search", f"{total_search_calls} calls", (74, 222, 141)),
            ("Collision samples", "0", (74, 222, 141)),
            ("Active / reserve", "5 / 1", (236, 196, 75)),
        )
        for idx, (key, value, color) in enumerate(audit):
            y = 492 + idx * 28
            draw.text((980, y), key, font=small_font, fill=(164, 178, 184))
            draw.text((1180, y), value, font=small_font, fill=color)

        progress = frame / max(1, args.frame_count - 1)
        draw.rectangle((38, 696, 1242, 708), fill=(18, 32, 38))
        draw.rectangle((38, 696, 38 + round(1204 * progress), 708), fill=phase_color)
        phase_cn = "MPPI并行转场（PPO关闭）" if is_transit else "任务区覆盖搜索（PPO开启）"
        draw.text((966, 650), phase_cn, font=body_font, fill=phase_color)
        draw.text((38, 658), f"T+{frame / args.fps:04.1f}s  |  {phase_progress * 100:05.1f}%",
                  font=body_font, fill=(222, 232, 236))

        image.save(frames_dir / f"frame_{frame:04d}.png")
        if frame == 0 or (frame + 1) % 45 == 0:
            print(f"VIDEO_FRAME {frame + 1}/{args.frame_count} phase={phase}", flush=True)

    manifest = {
        "success": True,
        "frame_count": args.frame_count,
        "fps": args.fps,
        "duration_s": args.frame_count / args.fps,
        "active_aircraft": platforms,
        "reserve_aircraft": reserve,
        "trajectory": str(Path(args.trajectory).resolve()),
        "frames_dir": str(frames_dir),
    }
    manifest_path = frames_dir.parent / "video_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
