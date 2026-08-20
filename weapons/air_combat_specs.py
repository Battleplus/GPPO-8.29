from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GunSpec:
    name: str
    caliber_mm: float
    ammo_rounds: int
    rate_of_fire_rpm_min: int
    rate_of_fire_rpm_max: int
    armor_penetration_mm_rha: float
    effective_range_km: float
    role: str

    def scene_range(self, meters_per_unit: float) -> float:
        return self.effective_range_km * 1000.0 / float(meters_per_unit)


@dataclass(frozen=True)
class MissileSpec:
    name: str
    role: str
    max_range_km: float
    count: int
    guidance: tuple[str, ...]
    armor_penetration_mm_rha: float | None = None
    fire_and_forget: bool = False
    notes: str = ""

    def scene_range(self, meters_per_unit: float) -> float:
        return self.max_range_km * 1000.0 / float(meters_per_unit)


@dataclass(frozen=True)
class WeaponSuite:
    guns: tuple[GunSpec, ...] = ()
    missiles: tuple[MissileSpec, ...] = ()

    @property
    def missile_count(self) -> int:
        return sum(missile.count for missile in self.missiles)

    @property
    def max_standoff_range_km(self) -> float:
        ranges = [missile.max_range_km for missile in self.missiles] + [gun.effective_range_km for gun in self.guns]
        return max(ranges) if ranges else 0.0


LIGHT_AIR_TO_GROUND_GUIDANCE = ("semi_active_laser", "eo_ir")
CH4_LIGHT_AGM_RANGE_KM = 8.0


WZ21_WEAPONS = WeaponSuite(
    guns=(
        GunSpec(
            name="CS/LM12 30 mm chain gun",
            caliber_mm=30.0,
            ammo_rounds=1280,
            rate_of_fire_rpm_min=500,
            rate_of_fire_rpm_max=800,
            armor_penetration_mm_rha=100.0,
            effective_range_km=2.0,
            role="close suppression against infantry, light armor, fieldworks and low-altitude targets",
        ),
    ),
    missiles=(
        MissileSpec(
            name="HJ-21 / AKD-21",
            role="long-range anti-armor",
            max_range_km=30.0,
            count=16,
            guidance=("active_mmw_aesa", "imaging_ir", "semi_active_laser"),
            armor_penetration_mm_rha=1400.0,
            fire_and_forget=True,
            notes="Long-range beyond-line-of-sight anti-armor missile.",
        ),
        MissileSpec(
            name="PL-10E",
            role="short-range air-to-air",
            max_range_km=20.0,
            count=4,
            guidance=("imaging_ir",),
            armor_penetration_mm_rha=None,
            fire_and_forget=True,
            notes="Optional self-defense air-to-air missile load.",
        ),
    ),
)


AH64E_WEAPONS = WeaponSuite(
    guns=(
        GunSpec(
            name="M230 30 mm chain gun",
            caliber_mm=30.0,
            ammo_rounds=1200,
            rate_of_fire_rpm_min=625,
            rate_of_fire_rpm_max=625,
            armor_penetration_mm_rha=80.0,
            effective_range_km=1.5,
            role="close suppression",
        ),
    ),
    missiles=(
        MissileSpec(
            name="AGM-114 Hellfire",
            role="anti-armor",
            max_range_km=11.0,
            count=16,
            guidance=("semi_active_laser", "mmw_radar_variant"),
            armor_penetration_mm_rha=1200.0,
            fire_and_forget=True,
            notes="Shorter-range anti-armor missile family.",
        ),
        MissileSpec(
            name="FIM-92 Stinger",
            role="short-range air-to-air self-defense",
            max_range_km=5.0,
            count=4,
            guidance=("infrared",),
            armor_penetration_mm_rha=None,
            fire_and_forget=True,
            notes="Optional short-range self-defense load.",
        ),
    ),
)


CH4_STRIKE_RECON_WEAPONS = WeaponSuite(
    missiles=(
        MissileSpec(
            name="CH-4 AR-1 class air-to-ground missile",
            role="light air-to-ground anti-armor precision attack",
            max_range_km=CH4_LIGHT_AGM_RANGE_KM,
            count=4,
            guidance=LIGHT_AIR_TO_GROUND_GUIDANCE,
            armor_penetration_mm_rha=450.0,
            fire_and_forget=False,
            notes="CH-4 reference air-to-ground weapon envelope: 2-8 km.",
        ),
    ),
)


QUAD_STRIKE_UAV_WEAPONS = WeaponSuite(
    missiles=(
        replace(
            CH4_STRIKE_RECON_WEAPONS.missiles[0],
            name="Quadrotor light air-to-ground missile",
            max_range_km=6.0,
            count=2,
            guidance=("eo_ir", "semi_active_laser"),
            armor_penetration_mm_rha=260.0,
            notes="Small UAV strike store in the CH-4 reference 2-8 km air-to-ground envelope.",
        ),
    ),
)


UNARMED_WEAPONS = WeaponSuite()
