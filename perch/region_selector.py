"""Select LLM/RAG attack regions and convert them to Perch coordinates."""

from __future__ import annotations

import logging
import math
from typing import Any

from brain.domain.attack_region import AttackRegion
from brain.domain.result import AlgorithmResult
from brain.domain.task import StrikeTask, one_to_one_strike_tasks

from .models import resolve_weapon_envelope_km
from .region_recommender import AttackRegionRecommender
from .situation_understanding import SituationUnderstanding

logger = logging.getLogger(__name__)

_KM_PER_DEG_LAT = 111.0
_DISABLED_MODES = {"disabled", "off", "none", "false", "0"}


class AttackRegionSelector:
    """Run Perch's region recommender before concrete point optimisation."""

    def __init__(
        self,
        mode: str = "llm",
        top_k: int = 3,
        recommender: Any | None = None,
    ) -> None:
        self._mode = str(mode).lower()
        self._top_k = max(1, int(top_k))
        self._recommender = recommender

    def select_regions(
        self,
        context: Any,
        action_allocation: Any,
    ) -> AlgorithmResult:
        world = getattr(context, "world_state", {})
        world.pop("attack_region_errors", None)
        world["attack_region_situations"] = {}
        world["attack_region_knowledge_sources"] = {}
        mode = str(world.get("attack_region_mode", self._mode)).lower()
        if mode in _DISABLED_MODES:
            world["attack_regions"] = {}
            return AlgorithmResult.ok({})

        tasks = one_to_one_strike_tasks(action_allocation or [])
        if not tasks:
            world["attack_regions"] = {}
            return AlgorithmResult.fail(
                "No one-to-one strike assignments for attack-region selection"
            )

        targets = self._targets_by_id(world)
        regions_by_target: dict[str, list[dict[str, Any]]] = {}
        errors: list[str] = []
        for task in tasks:
            target_id = str(task.target)
            if target_id in regions_by_target:
                continue
            target = targets.get(target_id)
            if target is None:
                errors.append(f"Target {target_id} not found")
                continue
            try:
                regions = self._select_for_target(
                    world,
                    task,
                    target,
                    agents=getattr(context, "agents", []),
                )
            except Exception as exc:  # pragma: no cover - adapter boundary
                logger.exception(
                    "Perch attack-region selection failed for target=%s",
                    target_id,
                )
                errors.append(f"{target_id}: {exc}")
                continue
            if not regions:
                errors.append(f"{target_id}: no attack regions generated")
                continue
            regions_by_target[target_id] = [
                region.to_dict() for region in regions[: self._top_k]
            ]

        world["attack_regions"] = regions_by_target
        if errors:
            world.setdefault("attack_region_errors", []).extend(errors)
        if not regions_by_target:
            return AlgorithmResult.fail(
                "; ".join(errors) or "Attack-region selection failed"
            )
        return AlgorithmResult.ok(regions_by_target)

    def _select_for_target(
        self,
        world: dict[str, Any],
        task: StrikeTask,
        target: dict[str, Any],
        agents: Any = None,
    ) -> list[AttackRegion]:
        mode = str(world.get("attack_region_mode", self._mode)).lower()
        target_id = str(task.target)
        target_km = self._target_km(target, world)
        target_geo = self._target_geo(target)
        geo_reference = world.get("geo_reference")
        target_geo_known = target_geo is not None
        if target_geo is None and geo_reference:
            target_geo = self._km_to_geo_with_reference(target_km, world)
            target_geo_known = target_geo is not None
        if target_geo is None:
            target_geo = (0.0, 0.0)
            conversion_mode = "target_relative"
        elif geo_reference:
            conversion_mode = "geo_reference"
        else:
            conversion_mode = "target_relative"

        description = self._build_description(
            task=task,
            target=target,
            target_geo=target_geo,
            target_geo_known=target_geo_known,
            world=world,
            agents=agents,
        )
        world.setdefault("attack_region_situations", {})[target_id] = description
        if mode in {"demo", "rule", "rules", "fallback"}:
            return self._local_fallback_regions(world, task, target_km)

        external_error = ""
        try:
            recommender = self._recommender or AttackRegionRecommender()
            result = recommender.recommend(description)
            world.setdefault("attack_region_knowledge_sources", {})[
                target_id
            ] = list(result.get("knowledge_sources", []))
            if result.get("success") and result.get("geojson"):
                regions = self._regions_from_geojson(
                    geojson_data=result["geojson"],
                    target_id=target_id,
                    target_km=target_km,
                    target_geo=target_geo,
                    conversion_mode=conversion_mode,
                    world=world,
                    source="perch:llm",
                )
                if regions:
                    return regions
            external_error = "; ".join(result.get("errors", []))
            if not external_error:
                external_error = "LLM returned no usable attack regions"
        except Exception as exc:
            external_error = str(exc)

        if external_error:
            logger.warning(
                "Perch LLM attack-region recommender unavailable for %s: %s",
                target_id,
                external_error,
            )
            world.setdefault("attack_region_errors", []).append(
                f"{target_id}: {external_error}"
            )
        return self._local_fallback_regions(world, task, target_km)

    def _regions_from_geojson(
        self,
        geojson_data: dict[str, Any],
        target_id: str,
        target_km: tuple[float, float],
        target_geo: tuple[float, float],
        conversion_mode: str,
        world: dict[str, Any],
        source: str,
    ) -> list[AttackRegion]:
        features = list(geojson_data.get("features", []))
        features.sort(
            key=lambda feature: float(
                feature.get("properties", {}).get("score", 0.0)
            ),
            reverse=True,
        )
        regions: list[AttackRegion] = []
        for rank, feature in enumerate(features[: self._top_k]):
            geometry = feature.get("geometry", {})
            coordinates = geometry.get("coordinates", [])
            if (
                geometry.get("type") != "Polygon"
                or not coordinates
                or not coordinates[0]
            ):
                continue
            polygon_km = [
                list(self._geo_point_to_km(
                    lon=float(point[0]),
                    lat=float(point[1]),
                    target_km=target_km,
                    target_geo=target_geo,
                    conversion_mode=conversion_mode,
                    world=world,
                ))
                for point in coordinates[0]
            ]
            polygon_km = _closed_polygon(polygon_km)
            polygon_scene = [
                list(self._km_to_scene(world, point[0], point[1]))
                for point in polygon_km
            ]
            properties = feature.get("properties", {})
            regions.append(AttackRegion(
                target_id=target_id,
                score=float(properties.get("score", 0.0)),
                reasoning=str(properties.get("reasoning", "")),
                polygon_km=polygon_km,
                polygon_scene=_closed_polygon(polygon_scene),
                source=source,
                rank=rank,
                raw_feature=dict(feature),
            ))
        return regions

    def _geo_point_to_km(
        self,
        lon: float,
        lat: float,
        target_km: tuple[float, float],
        target_geo: tuple[float, float],
        conversion_mode: str,
        world: dict[str, Any],
    ) -> tuple[float, float]:
        if conversion_mode == "geo_reference":
            converted = self._geo_to_km_with_reference(lon, lat, world)
            if converted is not None:
                return converted
        dx, dy = _geo_offset_km(lon, lat, *target_geo)
        return target_km[0] + dx, target_km[1] + dy

    def _geo_to_km_with_reference(
        self,
        lon: float,
        lat: float,
        world: dict[str, Any],
    ) -> tuple[float, float] | None:
        reference = world.get("geo_reference")
        if not isinstance(reference, dict):
            return None
        origin_geo = reference.get("origin_geo") or reference.get("geo_origin")
        origin_km = reference.get("origin_km") or reference.get("km_origin")
        origin_lon = reference.get("origin_lon", reference.get("lon"))
        origin_lat = reference.get("origin_lat", reference.get("lat"))
        if origin_geo is not None:
            origin_lon, origin_lat = float(origin_geo[0]), float(origin_geo[1])
        if origin_km is None:
            origin_km = [
                reference.get("origin_x_km", reference.get("x_km", 0.0)),
                reference.get("origin_y_km", reference.get("y_km", 0.0)),
            ]
        if origin_lon is None or origin_lat is None:
            return None
        dx, dy = _geo_offset_km(
            lon,
            lat,
            float(origin_lon),
            float(origin_lat),
        )
        return float(origin_km[0]) + dx, float(origin_km[1]) + dy

    def _km_to_geo_with_reference(
        self,
        point_km: tuple[float, float],
        world: dict[str, Any],
    ) -> tuple[float, float] | None:
        reference = world.get("geo_reference")
        if not isinstance(reference, dict):
            return None
        origin_geo = reference.get("origin_geo") or reference.get("geo_origin")
        origin_km = reference.get("origin_km") or reference.get("km_origin")
        origin_lon = reference.get("origin_lon", reference.get("lon"))
        origin_lat = reference.get("origin_lat", reference.get("lat"))
        if origin_geo is not None:
            origin_lon, origin_lat = float(origin_geo[0]), float(origin_geo[1])
        if origin_km is None:
            origin_km = [
                reference.get("origin_x_km", reference.get("x_km", 0.0)),
                reference.get("origin_y_km", reference.get("y_km", 0.0)),
            ]
        if origin_lon is None or origin_lat is None:
            return None
        dx = float(point_km[0]) - float(origin_km[0])
        dy = float(point_km[1]) - float(origin_km[1])
        latitude = float(origin_lat) + dy / _KM_PER_DEG_LAT
        km_per_deg_lon = _KM_PER_DEG_LAT * math.cos(
            math.radians(float(origin_lat))
        )
        if abs(km_per_deg_lon) <= 1e-9:
            return None
        longitude = float(origin_lon) + dx / km_per_deg_lon
        return longitude, latitude

    def _local_fallback_regions(
        self,
        world: dict[str, Any],
        task: StrikeTask,
        target_km: tuple[float, float],
    ) -> list[AttackRegion]:
        munition = str(task.munition).upper()
        envelope = resolve_weapon_envelope_km(
            world, str(task.platform), munition
        )
        distance_km = float(envelope.get("optimal_range_km", 3.5))
        half_side = max(0.3, min(0.75, distance_km * 0.16))
        regions: list[AttackRegion] = []
        for rank, angle_deg in enumerate((45.0, 315.0)):
            angle = math.radians(angle_deg)
            center_x = target_km[0] + distance_km * math.sin(angle)
            center_y = target_km[1] + distance_km * math.cos(angle)
            polygon_km = _closed_polygon([
                [center_x - half_side, center_y - half_side],
                [center_x + half_side, center_y - half_side],
                [center_x + half_side, center_y + half_side],
                [center_x - half_side, center_y + half_side],
            ])
            polygon_scene = [
                list(self._km_to_scene(world, point[0], point[1]))
                for point in polygon_km
            ]
            regions.append(AttackRegion(
                target_id=str(task.target),
                score=0.65 - rank * 0.05,
                reasoning=(
                    "Perch local region near the weapon's optimal standoff "
                    "range; used because LLM selection was disabled or unavailable"
                ),
                polygon_km=polygon_km,
                polygon_scene=_closed_polygon(polygon_scene),
                source="perch:local_fallback",
                rank=rank,
            ))
        return regions[: self._top_k]

    def _build_description(
        self,
        task: StrikeTask,
        target: dict[str, Any],
        target_geo: tuple[float, float],
        target_geo_known: bool,
        world: dict[str, Any],
        agents: Any = None,
    ) -> str:
        return SituationUnderstanding().describe(
            world=world,
            task=task,
            target=target,
            target_km=self._target_km(target, world),
            target_geo=target_geo,
            target_geo_known=target_geo_known,
            agents=agents,
        )

    @staticmethod
    def _targets_by_id(world: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            str(target.get("tid", target.get("target_id", ""))): target
            for target in world.get("targets", [])
        }

    @classmethod
    def _target_km(
        cls,
        target: dict[str, Any],
        world: dict[str, Any],
    ) -> tuple[float, float]:
        for key in ("pos", "position_km"):
            if target.get(key) is not None:
                raw = target[key]
                return float(raw[0]), float(raw[1])
        for key in ("position_scene", "pos_scene"):
            if target.get(key) is not None:
                raw = target[key]
                return cls._scene_to_km(world, float(raw[0]), float(raw[1]))
        return 0.0, 0.0

    @staticmethod
    def _target_geo(target: dict[str, Any]) -> tuple[float, float] | None:
        raw = target.get("geo_pos") or target.get("geo")
        if isinstance(raw, dict):
            lon = raw.get("lon", raw.get("longitude"))
            lat = raw.get("lat", raw.get("latitude"))
            if lon is not None and lat is not None:
                return float(lon), float(lat)
        if raw is not None and not isinstance(raw, dict) and len(raw) >= 2:
            return float(raw[0]), float(raw[1])
        lon = target.get("lon", target.get("longitude"))
        lat = target.get("lat", target.get("latitude"))
        if lon is not None and lat is not None:
            return float(lon), float(lat)
        return None

    @classmethod
    def _km_to_scene(
        cls,
        world: dict[str, Any],
        x_km: float,
        y_km: float,
    ) -> tuple[float, float]:
        scale = 1000.0 / cls._meters_per_unit(world)
        half = cls._map_size_km(world) * 0.5
        return (float(x_km) - half) * scale, (float(y_km) - half) * scale

    @classmethod
    def _scene_to_km(
        cls,
        world: dict[str, Any],
        x_scene: float,
        y_scene: float,
    ) -> tuple[float, float]:
        scale = 1000.0 / cls._meters_per_unit(world)
        half = cls._map_size_km(world) * 0.5
        return float(x_scene) / scale + half, float(y_scene) / scale + half

    @staticmethod
    def _meters_per_unit(world: dict[str, Any]) -> float:
        return float(world.get("meters_per_unit", 100.0))

    @classmethod
    def _map_size_km(cls, world: dict[str, Any]) -> float:
        if "map_size_km" in world:
            return float(world["map_size_km"])
        if "map_size_units" in world:
            return (
                float(world["map_size_units"])
                * cls._meters_per_unit(world) / 1000.0
            )
        return 300.0


def _geo_offset_km(
    lon: float,
    lat: float,
    base_lon: float,
    base_lat: float,
) -> tuple[float, float]:
    km_per_deg_lon = _KM_PER_DEG_LAT * math.cos(math.radians(base_lat))
    return (
        (float(lon) - float(base_lon)) * km_per_deg_lon,
        (float(lat) - float(base_lat)) * _KM_PER_DEG_LAT,
    )


def _closed_polygon(points: list[list[float]]) -> list[list[float]]:
    if not points:
        return points
    result = [[float(point[0]), float(point[1])] for point in points]
    if result[0] != result[-1]:
        result.append(list(result[0]))
    return result
