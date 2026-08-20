"""天气工具函数。

天气影响传感器的有效性：
- 晴天 (SUNNY): EO 和 SAR 均可正常工作
- 雨天 (RAINY): 仅 SAR 可用，EO 失效

注意：当前全部使用 SAR 传感器，SAR 全天候工作，天气不再影响传感器有效性。
以下函数已注释保留，以备将来恢复 EO 传感器时使用。
"""

# import numpy as np
# from config import Weather


# def sample_weather(rng: np.random.Generator) -> Weather:
#     """随机采样天气状态。
#
#     以 50% 概率分别返回晴天或雨天。
#     使用 numpy Generator 以保证可复现性。
#
#     Args:
#         rng: numpy 随机数生成器
#
#     Returns:
#         Weather: 随机采样的天气枚举值
#     """
#     return Weather(int(rng.integers(0, 2)))


# def flip_weather(weather: Weather) -> Weather:
#     """翻转天气状态（晴天 ↔ 雨天）。
#
#     用于 WEATHER_INVALID 事件中，将某区域的天气
#     从晴天切换为雨天，模拟环境恶化。
#
#     Args:
#         weather: 当前天气
#
#     Returns:
#         Weather: 翻转后的天气
#     """
#     return Weather.RAINY if weather == Weather.SUNNY else Weather.SUNNY
