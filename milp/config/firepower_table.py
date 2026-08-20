"""
火力需求表 — 定义各目标类型所需的平台数量、弹药需求、武器成本与射程。

基于设计方案 §4.5 约束(9) 火力需求:
  - 雷达(RADAR): 需 2 架直升机, Hellfire 优先
  - 指挥所(CP):   需 2 架直升机, Hellfire + 火箭
  - 装甲车(AV):   需 1 架直升机, Hellfire 优先
"""

FIREPOWER_TABLE = {
    "RADAR": {
        "req_plat": 2,
        "req_weapon": {"HF": 2, "RKT": 0, "GUN": 0},
    },
    "CP": {
        "req_plat": 2,
        "req_weapon": {"HF": 1, "RKT": 4, "GUN": 0},
    },
    "AV": {
        "req_plat": 1,
        "req_weapon": {"HF": 1, "RKT": 2, "GUN": 50},
    },
}

WEAPON_COST = {
    "HF": 1.0,
    "RKT": 0.3,
    "GUN": 0.05,
}

WEAPON_RANGE_KM = {
    "HF": 8.0,
    "RKT": 8.0,
    "GUN": 2.0,
}


def get_firepower_requirement(target_type: str) -> dict:
    """返回目标类型对应的火力需求字典，含 req_plat 与 req_weapon。"""
    if target_type not in FIREPOWER_TABLE:
        raise KeyError(f"未知目标类型: {target_type}，已知类型: {list(FIREPOWER_TABLE.keys())}")
    return FIREPOWER_TABLE[target_type]


def get_weapon_cost(weapon_type: str) -> float:
    """返回弹药单发/单位成本。"""
    return WEAPON_COST.get(weapon_type, 1.0)


def get_weapon_range_km(weapon_type: str) -> float:
    """返回弹药最大射程 (km)。"""
    return WEAPON_RANGE_KM.get(weapon_type, 5.0)
