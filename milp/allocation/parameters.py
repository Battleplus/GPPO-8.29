import numpy as np
from core.snapshot import SituationSnapshot
from config.settings import GlobalSettings
from config.firepower_table import get_firepower_requirement, get_weapon_cost, get_weapon_range_km
from utils.geometry import euclidean_distance_km


def _weather_gain(sensor_name: str, weather_w: float) -> float:
    """
    天气-传感器增益 β_s(w_c)，按 5 档分段常数。
    基于设计方案 §4.5 约束(3) 天气-传感器分级表。
    """
    if sensor_name == "EO":
        if weather_w < 0.30:
            return 1.0
        elif weather_w < 0.45:
            return 0.7
        elif weather_w < 0.65:
            return 0.4
        elif weather_w < 0.80:
            return 0.1
        else:
            return 0.0
    elif sensor_name == "SAR":
        if weather_w < 0.30:
            return 0.6
        elif weather_w < 0.45:
            return 0.7
        elif weather_w < 0.65:
            return 0.9
        elif weather_w < 0.80:
            return 1.0
        else:
            return 1.0
    else:
        return 1.0


class ParameterBuilder:
    """从 SituationSnapshot 提取并计算全部 MILP 参数矩阵。"""

    def __init__(self, snapshot: SituationSnapshot, settings: GlobalSettings):
        self.snap = snapshot
        self.settings = settings

        # -- 平台分组 --
        self.uav_platforms = snapshot.get_uav_platforms()
        self.heli_platforms = snapshot.get_heli_platforms()
        self.all_platforms = [p for p in snapshot.platforms if not p.lost]

        # -- 活跃目标（存活且已确认）--
        self.active_targets = snapshot.get_active_targets()

        # -- 栅格 --
        self.cells = list(snapshot.grids)

        # -- 传感器列表（从第一架 UAV 获取；统一使用 UAV 传感器）--
        if self.uav_platforms:
            all_sensors = set()
            for p in self.uav_platforms:
                all_sensors.update(p.sensors_mounted)
            self.sensor_names = sorted(all_sensors)
        else:
            self.sensor_names = ["EO", "SAR", "ESM"]

        # -- 武器列表 --
        self.weapon_names = ["HF", "RKT", "GUN"]

        # -- 维度 --
        self.N_U = len(self.uav_platforms)
        self.N_H = len(self.heli_platforms)
        self.N_G = len(self.active_targets)
        self.N_C = len(self.cells)
        self.N_S = len(self.sensor_names)
        self.N_W = len(self.weapon_names)

        # -- 传感器参数查找表 --
        self._sensor_param_map = {}
        for sp in snapshot.sensor_params:
            self._sensor_param_map[sp.name] = {"P0": sp.P0, "R": sp.R, "weather_sensitive": sp.weather_sensitive}

        # ---- 栅格分类（巡逻区 vs 子区域） ----
        self.cell_is_patrol = np.array(
            [c.cell_id == "c0" for c in self.cells], dtype=bool
        )
        self.sub_cell_indices = [ci for ci in range(self.N_C)
                                 if not self.cell_is_patrol[ci]]

        # ---- 计算派生参数 ----
        self._extract_terrain_vectors()
        self._compute_cell_target_composition()
        self._compute_distance_matrices()
        self._compute_scan_feasibility()
        self._compute_sensor_cell_compat()
        self._compute_detection_tensors()
        self._compute_los_occlusion()
        self._extract_target_vectors()
        self._extract_cell_vectors()
        self._compile_firepower_params()
        self._compute_effective_threat()
        self._compute_scan_time_matrix()

    def _compute_distance_matrices(self):
        """计算平台-栅格距离矩阵 D_pc 和直升机-目标距离矩阵 D_hg。"""
        # 平台-栅格距离 (所有平台 × 栅格)
        self.D_pc = np.zeros((len(self.all_platforms), self.N_C))
        for i, plat in enumerate(self.all_platforms):
            for j, cell in enumerate(self.cells):
                self.D_pc[i, j] = euclidean_distance_km(plat.pos, cell.center) * self.dist_factor[j]

        # UAV-栅格距离
        self.D_uc = np.zeros((self.N_U, self.N_C))
        for i, uav in enumerate(self.uav_platforms):
            for j, cell in enumerate(self.cells):
                self.D_uc[i, j] = euclidean_distance_km(uav.pos, cell.center) * self.dist_factor[j]

        # 直升机-目标距离：从直升机位置到目标所在栅格中心（转场距离）
        self.D_hg = np.zeros((self.N_H, self.N_G))
        for j, tgt in enumerate(self.active_targets):
            cell_idx = self.target_cell_idx[j]
            cell_center = self.cells[cell_idx].center
            for i, heli in enumerate(self.heli_platforms):
                self.D_hg[i, j] = euclidean_distance_km(heli.pos, cell_center) * self.dist_factor[cell_idx]

    def _compute_scan_feasibility(self):
        """
        计算传感器-栅格扫描可行性矩阵 scan_feasible[N_S][N_C]。

        逻辑:
          swath = 2 × R_s（传感器探测直径）
          n_passes = ceil(cell.width_km / swath)
          scan_dist = n_passes × cell.height_km
          scan_time_min = scan_dist / loiter_speed × 60
          transit_time_min = 集结区到栅格中心的转场时间
          feasible = scan_time_min + transit_time_min <= total_time_max

        此矩阵与 UAV 无关——同一传感器对所有 UAV 的扫描时间相同。
        """
        loiter_speed = self.settings.uav_loiter_speed_kmh
        total_time_max = self.settings.mission_total_time_max_min

        # 计算转场距离（从集结区或各平台平均位置）
        if self.snap.staging_position is not None:
            staging_pos = self.snap.staging_position
        else:
            staging_pos = np.array([150.0, 150.0])  # 兼容旧接口：假想中心

        self.scan_feasible = np.ones((self.N_S, self.N_C), dtype=np.float64)
        for si, s_name in enumerate(self.sensor_names):
            R_s = self._sensor_param_map[s_name]["R"]
            swath = 2.0 * R_s
            for ci, cell in enumerate(self.cells):
                n_passes = max(1, int(np.ceil(cell.width_km / swath)))
                scan_dist = n_passes * cell.height_km
                scan_time_min = (scan_dist / loiter_speed) * 60.0 * self.time_factor[ci]
                if scan_time_min > total_time_max:
                    self.scan_feasible[si, ci] = 0.0

    def _compute_cell_target_composition(self):
        """将活跃目标按位置映射到栅格，记录每个栅格包含的目标类型集合
        以及每个目标所属（或最近）的栅格索引。"""
        self.cell_target_types = [set() for _ in range(self.N_C)]
        self.target_cell_idx = np.full(self.N_G, -1, dtype=int)
        for gi, tgt in enumerate(self.active_targets):
            tx, ty = tgt.pos_est[0], tgt.pos_est[1]
            for ci, cell in enumerate(self.cells):
                half_w = cell.width_km / 2.0
                half_h = cell.height_km / 2.0
                cx, cy = cell.center[0], cell.center[1]
                if (cx - half_w <= tx <= cx + half_w and
                        cy - half_h <= ty <= cy + half_h):
                    self.cell_target_types[ci].add(tgt.type)
                    self.target_cell_idx[gi] = ci
                    break
            # 目标不在任何栅格内 → 找最近栅格
            if self.target_cell_idx[gi] == -1:
                min_d = float('inf')
                for ci, cell in enumerate(self.cells):
                    d = euclidean_distance_km(tgt.pos_est, cell.center)
                    if d < min_d:
                        min_d = d
                        self.target_cell_idx[gi] = ci
                self.cell_target_types[self.target_cell_idx[gi]].add(tgt.type)

    def _compute_sensor_cell_compat(self):
        """计算传感器-栅格兼容性矩阵 sensor_cell_compat[N_S][N_C]。

        栅格内无目标 → 1.0（纯侦察，各传感器均可尝试）
        栅格内有目标 → 可探测目标类型数 / 目标类型总数
        """
        compat_map = self.settings.sensor_target_compat
        self.sensor_cell_compat = np.ones((self.N_S, self.N_C))
        for si, s_name in enumerate(self.sensor_names):
            s_compat = compat_map.get(s_name, {})
            for ci in range(self.N_C):
                types_in_cell = self.cell_target_types[ci]
                if len(types_in_cell) == 0:
                    self.sensor_cell_compat[si, ci] = 1.0
                else:
                    detectable = sum(
                        1 for t in types_in_cell
                        if s_compat.get(t, 1.0) > 0.5
                    )
                    self.sensor_cell_compat[si, ci] = detectable / len(types_in_cell)

    def _compute_detection_tensors(self):
        """
        计算探测概率张量 P_det[p][s][c] 和探测效能张量 E_det[p][s][c]。
        P_det_{p,s,c} = P0_s * β_s(w_c) * scan_feasible[s,c]
        E_det_{p,s,c} = β_s(w_c) * P_det_{p,s,c}

        in_range 由扫描可行性矩阵决定：传感器能扫完栅格即为 in_range。
        """
        self.P_det = np.zeros((self.N_U, self.N_S, self.N_C))
        self.E_det = np.zeros((self.N_U, self.N_S, self.N_C))

        for si, s_name in enumerate(self.sensor_names):
            params = self._sensor_param_map.get(s_name, {"P0": 0.8, "R": 50.0})
            P0 = params["P0"]

            for ci, cell in enumerate(self.cells):
                beta = _weather_gain(s_name, cell.weather_w)
                in_range = self.scan_feasible[si, ci]

                for ui in range(self.N_U):
                    p_det = P0 * beta * in_range * self.sensor_cell_compat[si, ci] * self.occ_factor[ci]
                    self.P_det[ui, si, ci] = p_det
                    self.E_det[ui, si, ci] = beta * p_det

        # 天气-传感器增益矩阵 (N_S × N_C)
        self.beta_sc = np.zeros((self.N_S, self.N_C))
        for si, s_name in enumerate(self.sensor_names):
            for ci, cell in enumerate(self.cells):
                self.beta_sc[si, ci] = (
                    _weather_gain(s_name, cell.weather_w) *
                    self.sensor_cell_compat[si, ci]
                )

    def _compute_los_occlusion(self):
        """提取或构造通视/遮挡矩阵。"""
        # LOS 矩阵: (N_H × N_G)，默认全 1
        if self.snap.los_matrix is not None and self.snap.los_matrix.size > 0:
            self.V_hg = np.array(self.snap.los_matrix, dtype=np.float64)
            if self.V_hg.shape[0] >= self.N_H and self.V_hg.shape[1] >= self.N_G:
                self.V_hg = self.V_hg[:self.N_H, :self.N_G]
            else:
                self.V_hg = np.ones((self.N_H, self.N_G))
        else:
            self.V_hg = np.ones((self.N_H, self.N_G))

        # 遮挡矩阵: (N_H × N_G)，默认全 1
        if self.snap.occlusion_matrix is not None and self.snap.occlusion_matrix.size > 0:
            self.eta_hg = np.array(self.snap.occlusion_matrix, dtype=np.float64)
            if self.eta_hg.shape[0] >= self.N_H and self.eta_hg.shape[1] >= self.N_G:
                self.eta_hg = self.eta_hg[:self.N_H, :self.N_G]
            else:
                self.eta_hg = np.ones((self.N_H, self.N_G))
        else:
            self.eta_hg = np.ones((self.N_H, self.N_G))

        # 武器-目标距离可行性矩阵 (N_H × N_W × N_G)
        self.weapon_range_feasible = np.ones((self.N_H, self.N_W, self.N_G), dtype=np.float64)
        for hi in range(self.N_H):
            for wi, w_name in enumerate(self.weapon_names):
                w_range = get_weapon_range_km(w_name)
                for gi in range(self.N_G):
                    if self.D_hg[hi, gi] > w_range:
                        self.weapon_range_feasible[hi, wi, gi] = 0.0

    def _extract_target_vectors(self):
        """提取目标价值、威胁向量。"""
        self.target_values = np.zeros(self.N_G)
        self.target_threats = np.zeros(self.N_G)
        self.target_types = []
        self.target_confirmed = np.zeros(self.N_G)
        self.target_alive = np.zeros(self.N_G)
        for gi, tgt in enumerate(self.active_targets):
            self.target_values[gi] = tgt.value
            self.target_threats[gi] = tgt.threat
            self.target_types.append(tgt.type)
            self.target_confirmed[gi] = 1.0 if tgt.confirmed else 0.0
            self.target_alive[gi] = 1.0 if tgt.alive else 0.0

    def _extract_cell_vectors(self):
        """提取栅格先验概率、天气系数向量。"""
        self.cell_priors = np.zeros(self.N_C)
        self.cell_weather = np.zeros(self.N_C)
        self.cell_ids = []
        for ci, cell in enumerate(self.cells):
            self.cell_priors[ci] = cell.target_prior
            self.cell_weather[ci] = cell.weather_w
            self.cell_ids.append(cell.cell_id)

        # 标记哪些栅格在 commander_AOI 内
        self.cell_in_aoi = np.ones(self.N_C, dtype=np.float64)
        if self.snap.commander_AOI:
            self.cell_in_aoi = np.zeros(self.N_C, dtype=np.float64)
            for ci, cell in enumerate(self.cells):
                self.cell_in_aoi[ci] = 1.0

    def _compile_firepower_params(self):
        """编译火力需求参数。"""
        self.req_plat = np.zeros(self.N_G)
        self.req_weapon = np.zeros((self.N_G, self.N_W))
        self.weapon_costs = np.array([get_weapon_cost(w) for w in self.weapon_names])

        for gi, tgt in enumerate(self.active_targets):
            fp = get_firepower_requirement(tgt.type)
            self.req_plat[gi] = fp["req_plat"]
            for wi, w_name in enumerate(self.weapon_names):
                self.req_weapon[gi, wi] = fp["req_weapon"].get(w_name, 0)

        # 弹药余量 (N_H × N_W)
        self.ammo_available = np.zeros((self.N_H, self.N_W))
        for hi, heli in enumerate(self.heli_platforms):
            for wi, w_name in enumerate(self.weapon_names):
                self.ammo_available[hi, wi] = heli.munitions.get(w_name, 0)

    def _extract_terrain_vectors(self):
        """从每个栅格的 terrain_level 查表得到四个地形系数向量。"""
        s = self.settings
        self.terrain_level = np.zeros(self.N_C, dtype=int)
        self.occ_factor = np.ones(self.N_C)
        self.time_factor = np.ones(self.N_C)
        self.dist_factor = np.ones(self.N_C)
        self.shield_factor = np.zeros(self.N_C)

        for ci, cell in enumerate(self.cells):
            tl = cell.terrain_level
            self.terrain_level[ci] = tl
            self.occ_factor[ci] = s.terrain_occ[tl]
            self.time_factor[ci] = s.terrain_time[tl]
            self.dist_factor[ci] = s.terrain_dist[tl]
            self.shield_factor[ci] = s.terrain_shield[tl]

    def _compute_effective_threat(self):
        """计算每个目标的有效威胁（加入地形掩护修正）。"""
        k_s = self.settings.k_shield
        self.effective_threat = np.zeros(self.N_G)
        for gi in range(self.N_G):
            ci = self.target_cell_idx[gi]
            shield = self.shield_factor[ci]
            self.effective_threat[gi] = self.target_threats[gi] * (1.0 - k_s * shield)

    def _compute_scan_time_matrix(self):
        """计算每架 UAV 扫描每个栅格的实际用时(分钟)，已含地形修正。"""
        loiter_speed = self.settings.uav_loiter_speed_kmh
        self.T_scan_uc = np.zeros((self.N_U, self.N_C))
        for ci, cell in enumerate(self.cells):
            best_time = float('inf')
            for si, s_name in enumerate(self.sensor_names):
                if self.scan_feasible[si, ci] < 0.5:
                    continue
                R_s = self._sensor_param_map[s_name]["R"]
                swath = 2.0 * R_s
                n_passes = max(1, int(np.ceil(cell.width_km / swath)))
                scan_dist = n_passes * cell.height_km
                t = (scan_dist / loiter_speed) * 60.0 * self.time_factor[ci]
                if t < best_time:
                    best_time = t
            if best_time == float('inf'):
                best_time = 999.0
            for ui in range(self.N_U):
                self.T_scan_uc[ui, ci] = best_time
