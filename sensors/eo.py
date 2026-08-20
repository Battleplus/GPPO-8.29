from __future__ import annotations

import numpy as np
from .weather_effects import apply_eo_weather


def process(rgb: np.ndarray, weather: dict | None = None, **_kwargs) -> np.ndarray:
    arr = np.clip(rgb[:, :, :3], 0, 255).astype(np.uint8)
    return apply_eo_weather(arr, weather=weather)
