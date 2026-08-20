from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SensorSpec:
    name: str
    channel: str
    max_range_km: float
    target_kinds: tuple[str, ...]
    modes: tuple[str, ...] = ()
    band: str = ""
    designation_range_km: float | None = None
    azimuth_fov_deg: float = 360.0
    elevation_fov_deg: float = 60.0
    scan_rate_hz: float = 1.0
    dwell_time_s: float = 0.25
    boresight_yaw_offset_deg: float = 0.0
    boresight_pitch_deg: float = -6.0
    can_detect_emitters: bool = False
    recognition_accuracy: float | None = None
    notes: str = ""

    @property
    def max_range_m(self) -> float:
        return self.max_range_km * 1000.0

    def scene_range(self, meters_per_unit: float) -> float:
        return self.max_range_m / float(meters_per_unit)


GROUND_SURVEILLANCE_TARGETS = ("armor", "command_post", "radar", "vehicle")
EMITTER_TARGETS = ("radar", "command_post")


WZ21_SENSORS: tuple[SensorSpec, ...] = (
    SensorSpec(
        name="WZ-21 AESA millimeter-wave radar",
        channel="mmw_radar",
        max_range_km=100.0,
        target_kinds=("armor", "command_post", "radar", "helicopter", "uav", "ship"),
        modes=("search", "fire_control", "terrain_following", "terrain_avoidance", "uav_cooperation"),
        band="35 GHz millimeter wave",
        azimuth_fov_deg=120.0,
        elevation_fov_deg=34.0,
        scan_rate_hz=2.0,
        dwell_time_s=0.18,
        boresight_pitch_deg=-4.0,
        can_detect_emitters=False,
        recognition_accuracy=0.95,
        notes="Rotor-mast AESA radar, all-weather search and fire-control support.",
    ),
    SensorSpec(
        name="WZ-21 third-generation EO/IR turret",
        channel="eo_ir",
        max_range_km=25.0,
        designation_range_km=20.0,
        target_kinds=("armor", "command_post", "radar", "helicopter", "uav"),
        modes=("ir", "visible", "laser_rangefinding", "laser_designation"),
        azimuth_fov_deg=70.0,
        elevation_fov_deg=45.0,
        scan_rate_hz=1.5,
        dwell_time_s=0.35,
        boresight_pitch_deg=-10.0,
        recognition_accuracy=0.95,
        notes="Nose turret with IR imaging, TV camera, laser ranging and laser designation.",
    ),
)


AH64E_SENSORS: tuple[SensorSpec, ...] = (
    SensorSpec(
        name="AH-64E AN/APG-78 Longbow radar",
        channel="mmw_radar",
        max_range_km=16.0,
        target_kinds=("armor", "command_post", "radar", "helicopter", "uav"),
        modes=("search", "fire_control", "terrain_following"),
        band="35 GHz millimeter wave",
        azimuth_fov_deg=90.0,
        elevation_fov_deg=28.0,
        scan_rate_hz=1.2,
        dwell_time_s=0.20,
        boresight_pitch_deg=-4.0,
        recognition_accuracy=0.85,
        notes="Rotor-mast millimeter-wave radar used for target detection and fire control.",
    ),
    SensorSpec(
        name="AH-64E TADS/PNVS",
        channel="eo_ir",
        max_range_km=12.0,
        designation_range_km=8.0,
        target_kinds=("armor", "command_post", "radar", "helicopter", "uav"),
        modes=("ir", "night_vision", "laser_rangefinding", "laser_designation"),
        azimuth_fov_deg=60.0,
        elevation_fov_deg=40.0,
        scan_rate_hz=1.0,
        dwell_time_s=0.32,
        boresight_pitch_deg=-10.0,
        recognition_accuracy=0.85,
        notes="Target acquisition and pilot night-vision system.",
    ),
)


CH4_RECON_SENSORS: tuple[SensorSpec, ...] = (
    SensorSpec(
        name="CH-4 EO payload",
        channel="eo_ir",
        max_range_km=15.0,
        target_kinds=GROUND_SURVEILLANCE_TARGETS,
        modes=("visible", "ir", "laser_rangefinding"),
        azimuth_fov_deg=55.0,
        elevation_fov_deg=35.0,
        scan_rate_hz=0.8,
        dwell_time_s=0.45,
        boresight_pitch_deg=-28.0,
        notes="Electro-optical search payload range reference.",
    ),
    SensorSpec(
        name="CH-4 SAR payload",
        channel="sar",
        max_range_km=50.0,
        target_kinds=GROUND_SURVEILLANCE_TARGETS,
        modes=("stripmap", "spotlight", "moving_target_indication"),
        azimuth_fov_deg=90.0,
        elevation_fov_deg=42.0,
        scan_rate_hz=0.5,
        dwell_time_s=0.80,
        boresight_pitch_deg=-35.0,
        notes="Synthetic aperture radar payload range reference.",
    ),
    SensorSpec(
        name="CH-4 ELINT payload",
        channel="elint",
        max_range_km=100.0,
        target_kinds=EMITTER_TARGETS,
        modes=("radar_emitter_search", "bearing_only_geolocation"),
        azimuth_fov_deg=360.0,
        elevation_fov_deg=80.0,
        scan_rate_hz=0.25,
        dwell_time_s=1.00,
        boresight_pitch_deg=-4.0,
        can_detect_emitters=True,
        notes="Electronic reconnaissance payload for radiating radar sources.",
    ),
)


CH4_STRIKE_RECON_SENSORS: tuple[SensorSpec, ...] = CH4_RECON_SENSORS


QUAD_RECON_EO_IR = SensorSpec(
    name="Quadrotor stabilized EO/IR payload",
    channel="eo_ir",
    max_range_km=8.0,
    designation_range_km=6.0,
    target_kinds=GROUND_SURVEILLANCE_TARGETS,
    modes=("visible", "ir", "laser_rangefinding"),
    azimuth_fov_deg=65.0,
    elevation_fov_deg=50.0,
    scan_rate_hz=1.4,
    dwell_time_s=0.30,
    boresight_pitch_deg=-32.0,
    recognition_accuracy=0.82,
    notes="Low-altitude quadrotor EO/IR search payload for close reconnaissance.",
)


QUAD_RECON_SAR = SensorSpec(
    name="Quadrotor lightweight SAR payload",
    channel="sar",
    max_range_km=20.0,
    target_kinds=GROUND_SURVEILLANCE_TARGETS,
    modes=("stripmap", "spotlight"),
    azimuth_fov_deg=90.0,
    elevation_fov_deg=45.0,
    scan_rate_hz=0.5,
    dwell_time_s=0.80,
    boresight_pitch_deg=-35.0,
    recognition_accuracy=0.80,
    notes="Lightweight SAR payload for all-weather subarea reconnaissance.",
)


QUAD_RECON_SENSORS: tuple[SensorSpec, ...] = (
    QUAD_RECON_EO_IR,
    QUAD_RECON_SAR,
)


QUAD_STRIKE_SENSORS: tuple[SensorSpec, ...] = (
    replace(
        QUAD_RECON_EO_IR,
        name="Quadrotor strike EO/IR designator",
        modes=("visible", "ir", "laser_designation"),
        azimuth_fov_deg=60.0,
        elevation_fov_deg=46.0,
        scan_rate_hz=1.2,
        dwell_time_s=0.32,
        boresight_pitch_deg=-30.0,
        recognition_accuracy=0.80,
        notes="EO/IR and laser designation package for light air-to-ground stores.",
    ),
)


def max_range_km(sensors: tuple[SensorSpec, ...], channel: str | None = None) -> float:
    selected = [sensor.max_range_km for sensor in sensors if channel is None or sensor.channel == channel]
    return max(selected) if selected else 0.0
