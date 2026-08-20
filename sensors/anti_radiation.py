from __future__ import annotations

import numpy as np
from .weather_effects import apply_arm_weather


def process(
    rgb: np.ndarray,
    red_threshold: int = 135,
    red_ratio: float = 1.35,
    border_thickness: int = 3,
    weather: dict | None = None,
    **_kwargs,
) -> np.ndarray:
    arr = rgb[:, :, :3].astype(np.uint8)
    red = arr[:, :, 0].astype(np.int16)
    green = arr[:, :, 1].astype(np.int16)
    blue = arr[:, :, 2].astype(np.int16)
    red_threshold = int(red_threshold)
    red_ratio = float(red_ratio)
    border_thickness = max(1, int(border_thickness))
    mask = (red > red_threshold) & (red > green * red_ratio) & (red > blue * red_ratio)

    out = np.zeros_like(arr)
    gray = ((red + green + blue) / 3).astype(np.uint8)
    out[:, :, 0] = gray // 4
    out[:, :, 1] = gray // 4
    out[:, :, 2] = gray // 4
    out[mask] = np.array([255, 32, 0], dtype=np.uint8)

    if mask.any():
        ys, xs = np.where(mask)
        pad = 3 + border_thickness
        y0, y1 = max(0, ys.min() - pad), min(out.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(out.shape[1], xs.max() + pad + 1)
        out[y0:y1, x0 : x0 + border_thickness] = [255, 240, 0]
        out[y0:y1, x1 - border_thickness : x1] = [255, 240, 0]
        out[y0 : y0 + border_thickness, x0:x1] = [255, 240, 0]
        out[y1 - border_thickness : y1, x0:x1] = [255, 240, 0]
    return apply_arm_weather(out, weather=weather)
