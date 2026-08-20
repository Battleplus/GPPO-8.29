# QL Isaac Sim 场景

该文件夹包含用于 Isaac Sim 场景的重构后 QL 入口点。

默认场景内容：

- 山地地形
- 更大的对垒基地地图：两侧平坦平原、较低的中部山脉、河道以及海岸海湾
- 工程化战场细节：连接的道路网络、停机坪、土方工事、地堡、雷达站、集装箱及工业建筑
- 原始坦克、潜艇和河道 UUV 单元
- 森林/岩石散布
- 无人机及带巡逻动画的简易直升机
- 左/右飞机队列，包含无人机和直升机
- 来自无人机和直升机的视觉导弹发射效果
- EO、SAR 和抗辐射传感器相机；默认关闭图像捕获
- 可选择的飞机运动后端：`kinematic`（运动学）和 `dynamic`（点质量）
- 包含概率、虚警、漏检、地形遮挡、天气/海杂波、滤波和分类置信度的检测?跟踪链路
- 用于传感器测试的天气预设：`clear`（晴朗）、`cloudy`（多云）、`foggy`（有雾）、`rainy`（下雨）、`storm`（暴风雨）

## 运行方式

使用 Isaac Sim Python 运行：

```powershell
/home/isaac/isaacsim/python.sh /home/isaac/ql/test_air_combat_scene.py 

```

### 运行方式
默认运行不会保存传感器图像。要启用 EO/SAR/ARM 图像输出：】
20w场景
```
/home/isaac/isaacsim/python.sh /home/isaac/ql/test_air_combat_scene.py 

```

城市场景
```
/home/isaac/isaacsim/python.sh /home/isaac/ql/changjing.py 

```

### 编队规划（混合A* + 可视化）
```
# 仅规划（输出 JSON）
/home/isaac/isaacsim/python.sh scripts/planner.py --count 4 --formation v_shape --start -800 -600 80 --goal 800 600 80

# 规划 + Isaac Sim 可视化运行
/home/isaac/isaacsim/python.sh scripts/run_formation_mission.py --count 4 --formation v_shape --start -800 -600 80 --goal 800 600 80 --speed 25
```

参数说明:
- `--count` 编队成员数 (默认 4)
- `--formation` 编队类型: v_shape / diamond / line_abreast / column / echelon_left / echelon_right
- `--start / --goal` 起终点 x y z（单位坐标, 1单位=100m）
- `--spacing` 成员间距 (默认 40 单位)
- `--speed` 巡航速度 单位/秒 (默认 25 = 2500m/s)
- `--max-altitude` 最高飞行高度 (默认 200)
- `--no-vis` 禁用路径可视化

## 项目结构说明
ql 是自包含的 Python 项目文件。入口点是 ql/run_isaacsim_replay.py；场景构建位于 ql/scenes/ 中，设置位于 ql/configs/mountain_forest_aircraft.yaml 中。

默认配置控制地形、战场细节、森林、岩石、发射源、飞机队列、相机、传感器和导弹发射时机。CLI 选项仍可作为快速覆盖使用。

当 ../assets/drone_models/iris.usd 可用时，无人机将加载该文件。坦克和 UUV 通过 ground_units.tank_asset_path 和 ground_units.uuv_asset_path 支持可选的 USD 资源；否则场景会使用详细的过程化后备模型。
20w的场景在D:\code\xiangmu\ql\scenes\air_combat_scene.py

### 运动学介绍
采用的是简化到极致的运动学 没有电机之类的 就是一个点+模型的运动 存在速度和加速度限制
接口如下：
D:\code\xiangmu\ql\scenes\physx_motion_models.py
### 传感器介绍
传感器检测和跟踪在同一个 YAML 文件的 tracking 下配置。用于传感器测试的天气驱动配置在 weather 下。跟踪器模型包括：

检测概率
虚警和漏检
地形遮挡
云/雾/雨惩罚系数
海杂波惩罚系数
alpha?beta 滤波跟踪
按传感器类型的目标分类置信度
接口如下：
D:\code\xiangmu\ql\sensors
### 天气介绍

有简单的雾天、雨天、多云天、晴天
可以地图局部天气
接口如下：
D:\code\xiangmu\ql\sensors\weather_effects.py

### 武器介绍
一个简单的导弹
接口如下：
D:\code\xiangmu\ql\weapons