from __future__ import annotations

import numpy as np

try:
    from PIL import Image, ImageFilter

    HAS_PIL = True
except Exception:
    Image = None
    ImageFilter = None
    HAS_PIL = False


def _gaussian_blur_rgb(arr: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 1e-6:
        return arr
    if HAS_PIL:
        return np.array(Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=radius)))
    return arr


def apply_eo_weather(rgb: np.ndarray, weather: dict | None = None) -> np.ndarray:
    weather = dict(weather or {})
    arr = np.clip(rgb[:, :, :3], 0, 255).astype(np.float32)
    cloud = float(weather.get("cloud_cover", 0.0))
    fog = float(weather.get("fog_density", 0.0))
    rain = float(weather.get("rain_rate", 0.0))
    preset = str(weather.get("preset", "clear")).lower()

    brightness = max(0.30, 1.0 - 0.24 * cloud - 0.26 * fog - 0.20 * rain)
    contrast = max(0.22, 1.0 - 0.62 * fog - 0.28 * rain - 0.12 * cloud)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    arr = (arr - mean) * contrast + mean
    arr = arr * brightness

    haze = np.array([214.0, 220.0, 226.0], dtype=np.float32)
    arr = arr * (1.0 - 0.62 * fog) + haze * (0.62 * fog)
    blur_radius = 1.5 * cloud + 3.0 * fog + 2.0 * rain
    if preset == "storm":
        blur_radius *= 1.25
    arr = _gaussian_blur_rgb(np.clip(arr, 0, 255).astype(np.uint8), radius=blur_radius).astype(np.float32)

    # Overcast gradient: darker image top under heavy cloud/rain so the weather reads clearly.
    if cloud > 0.10 or rain > 0.10:
        h, w = arr.shape[:2]
        row = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        top_shadow = 1.0 - (0.18 * cloud + 0.16 * fog + 0.16 * rain) * (1.0 - row)
        arr = arr * np.clip(top_shadow, 0.45, 1.0)

    if rain > 0.08:
        streaks = np.zeros_like(arr)
        h, w = arr.shape[:2]
        count = max(40, int((w * h) / 2400 * rain * 3.8))
        rng = np.random.default_rng(12345)
        for _ in range(count):
            x = int(rng.integers(0, w))
            y = int(rng.integers(0, h))
            length = int(rng.integers(18, 44))
            thickness = int(rng.integers(1, 3))
            for k in range(length):
                yy = min(h - 1, y + k)
                xx = min(w - 1, max(0, x + k // 5))
                x0 = max(0, xx - thickness)
                x1 = min(w, xx + thickness + 1)
                streaks[yy, x0:x1, :] = 232.0
        arr = np.clip(arr * (1.0 - 0.28 * rain) + streaks * (0.28 * rain), 0, 255)

    # Near-lens fog bloom under heavy fog/rain for immediate visual feedback in saved sensor frames.
    if fog > 0.20 or rain > 0.25:
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx = w * 0.5
        cy = h * 0.45
        rx = max(1.0, w * 0.52)
        ry = max(1.0, h * 0.42)
        glow = 1.0 - np.clip(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2, 0.0, 1.0)
        bloom = glow[..., None] * (18.0 * fog + 14.0 * rain)
        arr = np.clip(arr + bloom, 0, 255)

    return np.clip(arr, 0, 255).astype(np.uint8)


def apply_sar_weather(rgb: np.ndarray, weather: dict | None = None) -> np.ndarray:
    weather = dict(weather or {})
    arr = np.clip(rgb[:, :, :3], 0, 255).astype(np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    cloud = float(weather.get("cloud_cover", 0.0))
    sea_clutter = float(weather.get("sea_clutter", 0.0))
    rain = float(weather.get("rain_rate", 0.0))
    fog = float(weather.get("fog_density", 0.0))
    preset = str(weather.get("preset", "clear")).lower()

    clutter = 22.0 * sea_clutter + 12.0 * rain + 4.0 * fog
    noisy = gray + np.random.normal(0.0, 5.0 + clutter, size=gray.shape).astype(np.float32)
    if sea_clutter > 0.05:
        h, w = noisy.shape
        row = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        banding = np.sin(row * 24.0) * (20.0 * sea_clutter)
        noisy = noisy + banding
    if cloud > 0.05:
        noisy *= max(0.72, 1.0 - 0.06 * cloud)
    noisy *= max(0.50, 1.0 - 0.18 * fog - 0.16 * rain)
    if preset == "storm":
        noisy *= 0.92
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return np.dstack([noisy, noisy, noisy])


def apply_arm_weather(rgb: np.ndarray, weather: dict | None = None) -> np.ndarray:
    weather = dict(weather or {})
    arr = np.clip(rgb[:, :, :3], 0, 255).astype(np.float32)
    rain = float(weather.get("rain_rate", 0.0))
    fog = float(weather.get("fog_density", 0.0))
    cloud = float(weather.get("cloud_cover", 0.0))
    preset = str(weather.get("preset", "clear")).lower()
    attenuation = max(0.78, 1.0 - 0.08 * rain - 0.05 * fog - 0.02 * cloud)
    noise = np.random.normal(0.0, 4.0 + 8.0 * rain + 4.0 * fog, size=arr.shape).astype(np.float32)
    arr = arr * attenuation + noise
    if preset in {"foggy", "rainy", "storm"}:
        arr = arr * max(0.88, 1.0 - 0.03 * fog - 0.02 * rain)
    return np.clip(arr, 0, 255).astype(np.uint8)
