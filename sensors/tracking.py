from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


TerrainHeightFn = Callable[[float, float, float, float], float]


DEFAULT_TRACKING_CONFIG: dict[str, Any] = {
    "enabled": True,
    "report_interval": 1.0,
    "output_enabled": False,
    "sensors": {
        "eo": {"range": 380.0, "base_pd": 0.90, "measurement_noise": 3.5, "false_alarm_rate": 0.015},
        "sar": {"range": 520.0, "base_pd": 0.84, "measurement_noise": 5.5, "false_alarm_rate": 0.035},
        "arm": {"range": 680.0, "base_pd": 0.92, "measurement_noise": 4.0, "false_alarm_rate": 0.010},
    },
    "weather": {"cloud_cover": 0.20, "fog_density": 0.08, "rain_rate": 0.05, "sea_clutter": 0.18},
    "filter": {"alpha": 0.68, "beta": 0.22, "association_gate": 42.0, "track_timeout": 6.0},
}


@dataclass
class Detection:
    sensor_name: str
    measured_position: np.ndarray
    confidence: float
    truth_id: str | None
    category_scores: dict[str, float]
    false_alarm: bool = False


@dataclass
class Track:
    track_id: str
    truth_id: str | None
    position: np.ndarray
    velocity: np.ndarray
    confidence: float
    category_scores: dict[str, float]
    last_update_t: float
    hits: int = 1
    source_sensors: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        top_category = None
        top_conf = 0.0
        if self.category_scores:
            top_category, top_conf = max(self.category_scores.items(), key=lambda item: item[1])
        return {
            "track_id": self.track_id,
            "truth_id": self.truth_id,
            "position": [float(v) for v in self.position.tolist()],
            "velocity": [float(v) for v in self.velocity.tolist()],
            "confidence": float(self.confidence),
            "classification": top_category,
            "classification_confidence": float(top_conf),
            "source_sensors": sorted(self.source_sensors),
            "hits": int(self.hits),
        }


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def _is_over_water(position: np.ndarray, water_cfg: dict[str, Any], map_size: float) -> bool:
    if not bool(water_cfg.get("enabled", True)):
        return False
    x = float(position[0])
    y = float(position[1])
    river_path = water_cfg.get("river_path", [])
    river_width = float(water_cfg.get("river_width", 26.0))
    if river_path:
        river_points = [(float(px), float(py)) for px, py in river_path]
        for (x1, y1), (x2, y2) in zip(river_points, river_points[1:]):
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq <= 1e-9:
                continue
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg_len_sq))
            px = x1 + t * dx
            py = y1 + t * dy
            if math.hypot(x - px, y - py) <= river_width * 0.5:
                return True
    coast_start = map_size * 0.34
    return x >= coast_start


def _terrain_occluded(observer: np.ndarray, target: np.ndarray, terrain_height_fn: TerrainHeightFn, map_size: float, height_scale: float) -> bool:
    samples = 18
    ox, oy, oz = map(float, observer.tolist())
    tx, ty, tz = map(float, target.tolist())
    for idx in range(1, samples):
        s = idx / samples
        x = ox + (tx - ox) * s
        y = oy + (ty - oy) * s
        z = oz + (tz - oz) * s
        if float(terrain_height_fn(x, y, map_size, height_scale)) > z:
            return True
    return False


def _category_scores(sensor_name: str, target: dict[str, Any], visibility_factor: float) -> dict[str, float]:
    category = str(target.get("category", "unknown"))
    rf = bool(target.get("rf_emitter", False))
    if sensor_name == "arm":
        return {
            category: float(0.92 if rf else 0.08) * visibility_factor,
            "emitter": float(0.96 if rf else 0.04) * visibility_factor,
        }
    if sensor_name == "sar":
        base = {
            "tank": 0.82,
            "submarine": 0.58,
            "uuv": 0.26,
            "uav": 0.50,
            "helicopter": 0.62,
            "emitter": 0.48,
        }.get(category, 0.35)
        return {category: base * visibility_factor}
    base = {
        "tank": 0.88,
        "submarine": 0.44,
        "uuv": 0.10,
        "uav": 0.76,
        "helicopter": 0.82,
        "emitter": 0.40,
    }.get(category, 0.30)
    return {category: base * visibility_factor}


class SensorTrackingSystem:
    def __init__(
        self,
        config: dict[str, Any],
        terrain_height_fn: TerrainHeightFn,
        map_size: float,
        height_scale: float,
        water_cfg: dict[str, Any],
        seed: int = 42,
    ) -> None:
        self.config = json.loads(json.dumps(DEFAULT_TRACKING_CONFIG))
        self._deep_update(self.config, config or {})
        self.terrain_height_fn = terrain_height_fn
        self.map_size = float(map_size)
        self.height_scale = float(height_scale)
        self.water_cfg = dict(water_cfg or {})
        self.rng = random.Random(int(seed))
        self.track_counter = 0
        self.tracks: dict[str, Track] = {}

    def _deep_update(self, base: dict[str, Any], override: dict[str, Any]) -> None:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def report_interval(self) -> float:
        return max(0.1, float(self.config.get("report_interval", 1.0)))

    def step(
        self,
        t: float,
        observer: dict[str, Any],
        targets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        detections: list[Detection] = []
        for sensor_name, sensor_cfg in self.config.get("sensors", {}).items():
            detections.extend(self._sensor_detections(sensor_name, sensor_cfg, observer, targets))
        self._update_tracks(float(t), detections)
        report = {
            "time": float(t),
            "observer_id": str(observer.get("entity_id", "observer")),
            "detections": [self._detection_dict(det) for det in detections],
            "tracks": [track.as_dict() for track in self.tracks.values()],
        }
        return report

    def save_report(self, output_dir: Path, frame_idx: int, report: dict[str, Any]) -> None:
        if not bool(self.config.get("output_enabled", False)):
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{frame_idx:03d}_tracks.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _sensor_detections(
        self,
        sensor_name: str,
        sensor_cfg: dict[str, Any],
        observer: dict[str, Any],
        targets: list[dict[str, Any]],
    ) -> list[Detection]:
        detections: list[Detection] = []
        observer_pos = np.array(observer.get("position", [0.0, 0.0, 0.0]), dtype=float)
        observer_faction = str(observer.get("faction", "Blue"))

        for target in targets:
            if str(target.get("faction")) == observer_faction:
                continue
            target_pos = np.array(target.get("position", [0.0, 0.0, 0.0]), dtype=float)
            rng = _distance(observer_pos, target_pos)
            max_range = float(sensor_cfg.get("range", 300.0))
            if rng > max_range:
                continue

            occluded = _terrain_occluded(observer_pos, target_pos, self.terrain_height_fn, self.map_size, self.height_scale)
            if sensor_name != "sar" and occluded:
                continue

            visibility = max(0.02, 1.0 - (rng / max_range)) ** (0.85 if sensor_name == "sar" else 1.1)

            signature = float(target.get("signature", {}).get(sensor_name, 0.5))
            pd = float(sensor_cfg.get("base_pd", 0.8)) * visibility
            pd *= max(0.05, signature)
            if sensor_name == "arm" and not bool(target.get("rf_emitter", False)):
                pd *= 0.05
            pd = max(0.0, min(0.995, pd))
            if self.rng.random() > pd:
                continue

            noise = float(sensor_cfg.get("measurement_noise", 4.0))
            measured = target_pos + np.array(
                [
                    self.rng.gauss(0.0, noise),
                    self.rng.gauss(0.0, noise),
                    self.rng.gauss(0.0, noise * (0.28 if sensor_name == "sar" else 0.35)),
                ],
                dtype=float,
            )
            category_scores = _category_scores(sensor_name, target, pd)
            detections.append(
                Detection(
                    sensor_name=sensor_name,
                    measured_position=measured,
                    confidence=pd,
                    truth_id=str(target.get("entity_id")),
                    category_scores=category_scores,
                    false_alarm=False,
                )
            )

        false_alarm_rate = float(sensor_cfg.get("false_alarm_rate", 0.01))
        if self.rng.random() < false_alarm_rate:
            range_frac = self.rng.uniform(0.2, 0.95)
            angle = self.rng.uniform(-math.pi, math.pi)
            radius = range_frac * float(sensor_cfg.get("range", 300.0))
            ghost = observer_pos + np.array(
                [
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    self.rng.uniform(-6.0, 12.0),
                ],
                dtype=float,
            )
            detections.append(
                Detection(
                    sensor_name=sensor_name,
                    measured_position=ghost,
                    confidence=0.18 + 0.24 * self.rng.random(),
                    truth_id=None,
                    category_scores={"unknown": 0.35},
                    false_alarm=True,
                )
            )
        return detections

    def _detection_dict(self, det: Detection) -> dict[str, Any]:
        return {
            "sensor": det.sensor_name,
            "truth_id": det.truth_id,
            "false_alarm": bool(det.false_alarm),
            "confidence": float(det.confidence),
            "position": [float(v) for v in det.measured_position.tolist()],
            "category_scores": {k: float(v) for k, v in det.category_scores.items()},
        }

    def _update_tracks(self, t: float, detections: list[Detection]) -> None:
        filter_cfg = self.config.get("filter", {})
        gate = float(filter_cfg.get("association_gate", 42.0))
        alpha = float(filter_cfg.get("alpha", 0.68))
        beta = float(filter_cfg.get("beta", 0.22))
        timeout = float(filter_cfg.get("track_timeout", 6.0))

        active_ids = []
        for det in detections:
            best_track = None
            best_dist = float("inf")
            for track in self.tracks.values():
                dist = _distance(track.position, det.measured_position)
                if dist < gate and dist < best_dist:
                    best_track = track
                    best_dist = dist
            if best_track is None:
                track_id = f"TRK_{self.track_counter:04d}"
                self.track_counter += 1
                best_track = Track(
                    track_id=track_id,
                    truth_id=det.truth_id,
                    position=np.array(det.measured_position, dtype=float),
                    velocity=np.zeros(3, dtype=float),
                    confidence=float(det.confidence),
                    category_scores=dict(det.category_scores),
                    last_update_t=float(t),
                    source_sensors={det.sensor_name},
                )
                self.tracks[track_id] = best_track
            else:
                dt = max(1e-3, float(t) - float(best_track.last_update_t))
                prediction = best_track.position + best_track.velocity * dt
                residual = det.measured_position - prediction
                best_track.position = prediction + alpha * residual
                best_track.velocity = best_track.velocity + (beta / dt) * residual
                best_track.confidence = max(best_track.confidence * 0.82, float(det.confidence))
                best_track.category_scores.update(
                    {
                        key: max(float(det.category_scores.get(key, 0.0)), float(best_track.category_scores.get(key, 0.0)) * 0.92)
                        for key in set(best_track.category_scores) | set(det.category_scores)
                    }
                )
                best_track.truth_id = best_track.truth_id or det.truth_id
                best_track.last_update_t = float(t)
                best_track.hits += 1
                best_track.source_sensors.add(det.sensor_name)
            active_ids.append(best_track.track_id)

        stale = [
            track_id
            for track_id, track in self.tracks.items()
            if float(t) - float(track.last_update_t) > timeout
        ]
        for track_id in stale:
            self.tracks.pop(track_id, None)
