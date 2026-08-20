import numpy as np


def euclidean_distance_km(a: np.ndarray, b: np.ndarray) -> float:
    """计算两点之间的欧氏距离 (km)。a, b 为 shape (2,) 的 numpy 数组 [x, y]。"""
    return float(np.linalg.norm(a - b))


def haversine_distance_km(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两点之间的 Haversine 大圆距离 (km)。
    a, b 为 shape (2,) 的 numpy 数组 [lon, lat]（度）。
    当战场范围 ≤ 300 km 时，与欧氏距离差异 < 0.1%，直接使用 euclidean_distance_km 即可。
    """
    R = 6371.0
    lon1, lat1 = np.radians(a[0]), np.radians(a[1])
    lon2, lat2 = np.radians(b[0]), np.radians(b[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a_val = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(R * 2 * np.arctan2(np.sqrt(a_val), np.sqrt(1 - a_val)))


def compute_distance_matrix(positions_a: list, positions_b: list) -> np.ndarray:
    """
    计算两组点之间的欧氏距离矩阵。
    返回 shape (len(positions_a), len(positions_b)) 的矩阵。
    """
    N = len(positions_a)
    M = len(positions_b)
    D = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            D[i, j] = euclidean_distance_km(positions_a[i], positions_b[j])
    return D
