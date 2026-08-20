"""Platform / agent specification used by MILP and MPPI adapters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentSpec:
    """Describes a single platform (UAV or HELI).

    Mirrors the MILP ``PlatformInfo`` fields so the MILP adapter can
    translate without additional mapping.
    """

    pid: str
    """Unique platform identifier, e.g. ``"U1"``, ``"H1"``."""

    type: str
    """``"UAV"`` for reconnaissance, ``"HELI"`` for strike."""

    position: tuple[float, float]
    """Current (x, y) position in **kilometres** (matching the MILP convention)."""

    sensors: list[str] = field(default_factory=lambda: ["EO", "SAR", "ESM"])
    """Sensor payload, e.g. ``["EO","SAR","ESM"]`` for UAVs."""

    munitions: dict[str, int] = field(default_factory=dict)
    """Ammunition counts, e.g. ``{"HF":16,"RKT":76,"GUN":1200}`` for HELI."""

    altitude_km: float = 2.0
    """Cruise altitude in kilometres."""

    lost: bool = False
    """Whether the platform has been destroyed / disabled."""

    @property
    def pos_xy(self) -> tuple[float, float]:
        """Convenience accessor for the 2-D position tuple."""
        return self.position
