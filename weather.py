from __future__ import annotations

from copy import deepcopy
from typing import Any


WEATHER_PRESETS: dict[str, dict[str, Any]] = {
    "clear": {
        "preset": "clear",
        "cloud_cover": 0.05,
        "fog_density": 0.02,
        "rain_rate": 0.0,
        "sea_clutter": 0.10,
        "visual_clouds": True,
        "visual_rain": False,
    },
    "cloudy": {
        "preset": "cloudy",
        "cloud_cover": 0.55,
        "fog_density": 0.06,
        "rain_rate": 0.0,
        "sea_clutter": 0.16,
        "visual_clouds": True,
        "visual_rain": False,
    },
    "foggy": {
        "preset": "foggy",
        "cloud_cover": 0.22,
        "fog_density": 0.55,
        "rain_rate": 0.0,
        "sea_clutter": 0.20,
        "visual_clouds": True,
        "visual_rain": False,
    },
    "rainy": {
        "preset": "rainy",
        "cloud_cover": 0.42,
        "fog_density": 0.14,
        "rain_rate": 0.34,
        "sea_clutter": 0.28,
        "visual_clouds": True,
        "visual_rain": True,
    },
    "storm": {
        "preset": "storm",
        "cloud_cover": 0.62,
        "fog_density": 0.24,
        "rain_rate": 0.62,
        "sea_clutter": 0.45,
        "visual_clouds": True,
        "visual_rain": True,
    },
}


def resolve_weather_config(raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
    raw_cfg = dict(raw_cfg or {})
    preset = str(raw_cfg.get("preset", "clear")).lower()
    base = deepcopy(WEATHER_PRESETS.get(preset, WEATHER_PRESETS["clear"]))
    for key, value in raw_cfg.items():
        if value is not None:
            base[key] = value
    return base
