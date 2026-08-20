"""Task data models — what the MILP allocator produces and the FSM consumes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable


class TaskType(str, Enum):
    RECON = "recon"
    STRIKE = "strike"


@dataclass
class ReconTask:
    """A single reconnaissance assignment from MILP."""

    platform: str
    """Platform ID, e.g. ``"U1"``."""

    cell: str
    """Grid cell identifier, e.g. ``"c0"`` … ``"c4"``."""

    sensor: str
    """Sensor to use: ``"EO"``, ``"SAR"``, or ``"ESM"``."""

    role: str = "area_scan"
    """Tactical role: ``"area_scan"``, ``"subarea_search"``, ``"esm_patrol"``."""

    aoi: str = ""
    """AOI label, e.g. ``"A_3_4"``."""


@dataclass
class StrikeTask:
    """A single strike assignment from MILP."""

    platform: str
    """Platform ID, e.g. ``"H1"``."""

    target: str
    """Target ID, e.g. ``"g1"``."""

    munition: str
    """Primary munition type: ``"HF"``, ``"RKT"``, or ``"GUN"``."""

    qty: int
    """Number of primary-weapon rounds to expend."""

    role: str = "lead"
    """Tactical role: ``"lead"``, ``"striker"``, ``"wing"``, or ``"*_support"``."""

    aoi: str = ""

    assigned_munitions: dict[str, int] = field(default_factory=dict)
    """All munitions assigned to this platform/target pair."""


@dataclass
class TaskSpec:
    """Aggregated task list produced by MILP and consumed by downstream stages."""

    task_type: TaskType
    """``recon`` or ``strike``."""

    platform: str
    """Assigned platform ID."""

    # Recon-specific
    cell: str | None = None
    sensor: str | None = None
    recon_role: str | None = None
    aoi: str | None = None

    # Strike-specific
    target: str | None = None
    munition: str | None = None
    qty: int = 0
    strike_role: str | None = None

    @classmethod
    def from_recon(cls, recon: ReconTask) -> TaskSpec:
        return cls(
            task_type=TaskType.RECON,
            platform=recon.platform,
            cell=recon.cell,
            sensor=recon.sensor,
            recon_role=recon.role,
            aoi=recon.aoi,
        )

    @classmethod
    def from_strike(cls, strike: StrikeTask) -> TaskSpec:
        return cls(
            task_type=TaskType.STRIKE,
            platform=strike.platform,
            target=strike.target,
            munition=strike.munition,
            qty=strike.qty,
            strike_role=strike.role,
            aoi=strike.aoi,
        )


_MUNITION_PRIORITY = {"HF": 0, "RKT": 1, "GUN": 2}


def one_to_one_strike_tasks(tasks: Iterable[StrikeTask]) -> list[StrikeTask]:
    """Build a deterministic maximum one-platform/one-target matching.

    Duplicate MILP rows for the same platform and target are merged into an
    ammunition manifest.  Each returned platform and target appears once.
    """
    raw = [
        task for task in tasks
        if getattr(task, "platform", None) and getattr(task, "target", None)
    ]
    if not raw:
        return []

    pair_rows: dict[tuple[str, str], list[StrikeTask]] = {}
    platforms: list[str] = []
    adjacency: dict[str, list[str]] = {}
    for task in raw:
        platform = str(task.platform)
        target = str(task.target)
        pair_rows.setdefault((platform, target), []).append(task)
        if platform not in adjacency:
            platforms.append(platform)
            adjacency[platform] = []
        if target not in adjacency[platform]:
            adjacency[platform].append(target)

    target_to_platform: dict[str, str] = {}

    def augment(platform: str, seen_targets: set[str]) -> bool:
        for target in adjacency.get(platform, []):
            if target in seen_targets:
                continue
            seen_targets.add(target)
            previous = target_to_platform.get(target)
            if previous is None or augment(previous, seen_targets):
                target_to_platform[target] = platform
                return True
        return False

    for platform in platforms:
        augment(platform, set())

    platform_to_target = {
        platform: target for target, platform in target_to_platform.items()
    }
    result: list[StrikeTask] = []
    for platform in platforms:
        target = platform_to_target.get(platform)
        if target is None:
            continue
        rows = pair_rows[(platform, target)]
        manifest: dict[str, int] = {}
        for row in rows:
            loads = row.assigned_munitions or {str(row.munition): int(row.qty)}
            for name, qty in loads.items():
                manifest[str(name)] = manifest.get(str(name), 0) + int(qty)

        primary = min(
            manifest,
            key=lambda name: (_MUNITION_PRIORITY.get(name, 99), name),
        )
        template = next(
            (row for row in rows if str(row.munition) == primary),
            rows[0],
        )
        result.append(replace(
            template,
            platform=platform,
            target=target,
            munition=primary,
            qty=manifest[primary],
            assigned_munitions=manifest,
        ))
    return result
