"""
枚举类型定义 —— 项目中所有枚举类型集中管理。

职责:
  1. 定义目标类型、平台类型、传感器类型、弹药类型、角色类型
  2. 使用 StrEnum 实现字符串可直接比较与序列化

对外接口:
  - TargetType   — 目标类型 (RADAR / CP / AV)
  - PlatformType — 平台类型 (UAV / HELI)
  - SensorType   — 传感器类型 (EO / SAR / ESM / MMW / IR)
  - MunitionType — 弹药类型 (HF / RKT / GUN)
  - RoleType     — 角色类型 (lead / wing)

参考:
  设计方案 §4.1
"""

from enum import StrEnum


class TargetType(StrEnum):
    """
    目标类型枚举。

    值:
        RADAR: 雷达（静止，高威胁，需 ELINT/ESM 发现）
        CP:    指挥所 (Command Post)（静止，中威胁，需 SAR/EO 发现）
        AV:    装甲车 (Armored Vehicle)（机动，需 IR/EO 发现）
    """
    RADAR = "RADAR"
    CP = "CP"
    AV = "AV"


class PlatformType(StrEnum):
    """
    平台类型枚举。

    值:
        UAV:  彩虹-4 无人机（仅传感器，无打击能力）
        HELI: AH-64E 武装直升机（传感器 + 打击能力）
    """
    UAV = "UAV"
    HELI = "HELI"


class SensorType(StrEnum):
    """
    传感器类型枚举。

    值:
        EO:  光电传感器（受天气影响，作用距离 15 km）
        SAR: 合成孔径雷达（全天候，作用距离 50 km）
        ESM: 电子侦察/ELINT（仅探测雷达辐射，作用距离 100 km）
        MMW: 毫米波雷达（直升机用）
        IR:  红外传感器（直升机用，预留）
    """
    EO = "EO"
    SAR = "SAR"
    ESM = "ESM"
    MMW = "MMW"
    IR = "IR"


class MunitionType(StrEnum):
    """
    弹药类型枚举。

    值:
        HF:  AGM-114 Hellfire（反坦克导弹，成本最高，射程 8 km）
        RKT: 2.75" Hydra 火箭弹（面杀伤，成本中等，射程 8 km）
        GUN: 30mm M230 链炮（近程，成本最低，射程 2 km）
    """
    HF = "HF"
    RKT = "RKT"
    GUN = "GUN"


class RoleType(StrEnum):
    """
    协同角色枚举。

    值:
        LEAD: 长机（主攻，负责目标指示与首发）
        WING: 僚机（支援，协同攻击）
    """
    LEAD = "lead"
    WING = "wing"
