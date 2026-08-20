from __future__ import annotations

from dataclasses import dataclass, replace

from sensors.air_combat_specs import (
    AH64E_SENSORS,
    CH4_RECON_SENSORS,
    CH4_STRIKE_RECON_SENSORS,
    QUAD_RECON_SENSORS,
    QUAD_STRIKE_SENSORS,
    WZ21_SENSORS,
    SensorSpec,
)
from weapons.air_combat_specs import (
    AH64E_WEAPONS,
    CH4_STRIKE_RECON_WEAPONS,
    QUAD_STRIKE_UAV_WEAPONS,
    UNARMED_WEAPONS,
    WZ21_WEAPONS,
    WeaponSuite,
)


@dataclass(frozen=True)
class PlatformSpec:
    name: str
    role: str
    faction: str
    length_m: float
    height_m: float
    rotor_diameter_m: float
    empty_mass_kg: float
    payload_kg: float
    max_level_speed_kmh: float
    cruise_speed_kmh: float
    service_ceiling_m: float
    ferry_range_km: float
    combat_radius_km: float
    climb_rate_mps: float
    positive_g_limit: float
    negative_g_limit: float
    sensors: tuple[SensorSpec, ...]
    weapons: WeaponSuite
    color: tuple[float, float, float]
    accent_color: tuple[float, float, float]
    default_usd: str | None = None
    env_var: str | None = None

    @property
    def cruise_speed_mps(self) -> float:
        return self.cruise_speed_kmh / 3.6

    @property
    def max_level_speed_mps(self) -> float:
        return self.max_level_speed_kmh / 3.6

    @property
    def gross_mass_kg(self) -> float:
        return self.empty_mass_kg + self.payload_kg


@dataclass(frozen=True)
class GroundTargetSpec:
    name: str
    category: str
    faction: str
    is_fixed: bool
    is_radiating: bool
    priority: int
    dimensions_m: tuple[float, float, float]
    color: tuple[float, float, float]
    signature: dict[str, float]
    mobility_speed_kmh: float = 0.0

    @property
    def mobility_speed_mps(self) -> float:
        return self.mobility_speed_kmh / 3.6


WZ21_LEADER = PlatformSpec(
    name="WZ-21 Leader",
    role="leader_attack_helicopter",
    faction="Blue",
    length_m=19.4,
    height_m=4.5,
    rotor_diameter_m=16.2,
    empty_mass_kg=6500.0,
    payload_kg=4000.0,
    max_level_speed_kmh=360.0,
    cruise_speed_kmh=290.0,
    service_ceiling_m=6000.0,
    ferry_range_km=1500.0,
    combat_radius_km=560.0,
    climb_rate_mps=7.1,
    positive_g_limit=3.5,
    negative_g_limit=-0.5,
    sensors=WZ21_SENSORS,
    weapons=WZ21_WEAPONS,
    color=(0.17, 0.26, 0.20),
    accent_color=(0.08, 0.09, 0.07),
    default_usd="/home/isaac/ql/asset/WZ21/wz21.usd",
    env_var="QL_WZ21_USD",
)


WZ21_WINGMAN = replace(
    WZ21_LEADER,
    name="WZ-21 Wingman",
    role="wingman_attack_helicopter",
    color=(0.15, 0.24, 0.18),
    accent_color=(0.07, 0.08, 0.06),
)


AH64E_OPFOR = PlatformSpec(
    name="AH-64E Apache",
    role="enemy_attack_helicopter",
    faction="Red",
    length_m=15.06,
    height_m=3.87,
    rotor_diameter_m=14.63,
    empty_mass_kg=5165.0,
    # The provided 43.6 t payload is outside the AH-64 class; 4.36 t keeps the scene physically usable.
    payload_kg=4360.0,
    max_level_speed_kmh=293.0,
    cruise_speed_kmh=265.0,
    service_ceiling_m=4572.0,
    ferry_range_km=482.0,
    combat_radius_km=480.0,
    climb_rate_mps=12.7,
    positive_g_limit=3.5,
    negative_g_limit=-0.5,
    sensors=AH64E_SENSORS,
    weapons=AH64E_WEAPONS,
    color=(0.23, 0.25, 0.16),
    accent_color=(0.06, 0.06, 0.05),
    default_usd="/home/isaac/ql/asset/AH64/apache.usd",
    env_var="QL_AH64_USD",
)


CH4_RECON = PlatformSpec(
    name="CH-4 Recon UAV",
    role="recon_uav",
    faction="Blue",
    length_m=8.5,
    height_m=2.4,
    rotor_diameter_m=18.0,
    empty_mass_kg=900.0,
    payload_kg=120.0,
    max_level_speed_kmh=235.0,
    cruise_speed_kmh=180.0,
    service_ceiling_m=7000.0,
    ferry_range_km=2000.0,
    combat_radius_km=350.0,
    climb_rate_mps=5.0,
    positive_g_limit=3.0,
    negative_g_limit=-0.5,
    sensors=CH4_RECON_SENSORS,
    weapons=UNARMED_WEAPONS,
    color=(0.34, 0.39, 0.42),
    accent_color=(0.08, 0.10, 0.11),
    default_usd=None,
    env_var="QL_CH4_USD",
)


CH4_STRIKE_RECON = replace(
    CH4_RECON,
    name="CH-4 Strike-Recon UAV",
    role="strike_recon_uav",
    payload_kg=345.0,
    sensors=CH4_STRIKE_RECON_SENSORS,
    weapons=CH4_STRIKE_RECON_WEAPONS,
    color=(0.30, 0.36, 0.34),
    accent_color=(0.07, 0.09, 0.08),
)


QUAD_RECON_UAV = PlatformSpec(
    name="Quadrotor Recon UAV",
    role="quadrotor_recon_uav",
    faction="Blue",
    length_m=2.1,
    height_m=0.72,
    rotor_diameter_m=3.2,
    empty_mass_kg=42.0,
    payload_kg=8.0,
    max_level_speed_kmh=95.0,
    cruise_speed_kmh=62.0,
    service_ceiling_m=4500.0,
    ferry_range_km=90.0,
    combat_radius_km=35.0,
    climb_rate_mps=4.2,
    positive_g_limit=2.5,
    negative_g_limit=-0.2,
    sensors=QUAD_RECON_SENSORS,
    weapons=UNARMED_WEAPONS,
    color=(0.18, 0.25, 0.27),
    accent_color=(0.02, 0.55, 0.72),
    default_usd=None,
    env_var="QL_QUAD_UAV_USD",
)


QUAD_STRIKE_UAV = replace(
    QUAD_RECON_UAV,
    name="Quadrotor Strike UAV",
    role="quadrotor_strike_uav",
    length_m=2.3,
    height_m=0.78,
    rotor_diameter_m=3.4,
    empty_mass_kg=48.0,
    payload_kg=24.0,
    max_level_speed_kmh=88.0,
    cruise_speed_kmh=56.0,
    service_ceiling_m=4200.0,
    ferry_range_km=80.0,
    combat_radius_km=30.0,
    climb_rate_mps=3.8,
    positive_g_limit=2.3,
    sensors=QUAD_STRIKE_SENSORS,
    weapons=QUAD_STRIKE_UAV_WEAPONS,
    color=(0.20, 0.24, 0.20),
    accent_color=(0.96, 0.64, 0.10),
)


RADAR_SITE = GroundTargetSpec(
    name="Enemy radar emitter",
    category="radar",
    faction="Red",
    is_fixed=True,
    is_radiating=True,
    priority=95,
    dimensions_m=(18.0, 12.0, 10.0),
    color=(0.75, 0.12, 0.08),
    signature={"eo_ir": 0.72, "sar": 0.94, "elint": 1.0, "mmw_radar": 0.95},
)


COMMAND_POST = GroundTargetSpec(
    name="Enemy command post",
    category="command_post",
    faction="Red",
    is_fixed=True,
    is_radiating=False,
    priority=100,
    dimensions_m=(42.0, 30.0, 8.0),
    color=(0.40, 0.34, 0.25),
    signature={"eo_ir": 0.68, "sar": 0.88, "elint": 0.15, "mmw_radar": 0.80},
)


ARMORED_VEHICLE = GroundTargetSpec(
    name="Enemy armored vehicle",
    category="armor",
    faction="Red",
    is_fixed=False,
    is_radiating=False,
    priority=70,
    dimensions_m=(8.0, 4.2, 2.7),
    color=(0.42, 0.28, 0.18),
    signature={"eo_ir": 0.92, "sar": 0.86, "elint": 0.04, "mmw_radar": 0.90},
    mobility_speed_kmh=28.0,
)


BLUE_FORWARD_BASE = GroundTargetSpec(
    name="Blue forward base",
    category="base",
    faction="Blue",
    is_fixed=True,
    is_radiating=True,
    priority=98,
    dimensions_m=(1800.0, 1260.0, 35.0),
    color=(0.13, 0.30, 0.42),
    signature={"eo_ir": 0.78, "sar": 0.96, "elint": 0.85, "mmw_radar": 0.92},
)


RED_FORWARD_BASE = GroundTargetSpec(
    name="Red forward base",
    category="base",
    faction="Red",
    is_fixed=True,
    is_radiating=True,
    priority=98,
    dimensions_m=(1800.0, 1260.0, 35.0),
    color=(0.42, 0.18, 0.12),
    signature={"eo_ir": 0.78, "sar": 0.96, "elint": 0.85, "mmw_radar": 0.92},
)
