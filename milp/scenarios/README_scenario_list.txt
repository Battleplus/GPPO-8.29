MILP 侦察/打击任务分配 测试场景清单
共 64 个场景文件

TC_AOI_01_near_staging.json                          TC_AOI_01: staging_position 距离 AOI 很近
TC_AOI_02_far_staging.json                           TC_AOI_02: staging_position 距离 AOI 很远
TC_AOI_03_target_near_staging.json                   TC_AOI_03: 目标靠近集结点
TC_AOI_04_target_far_staging.json                    TC_AOI_04: 目标远离集结点
TC_AOI_05_aoi_1_1.json                               TC_AOI_05: AOI 修改为 A_1_1
TC_AOI_06_aoi_5_5.json                               TC_AOI_06: AOI 修改为 A_5_5
TC_AOI_07_aoi_target_mismatch.json                   TC_AOI_07: AOI 为 A_2_2 但目标分布偏向其他区域
TC_BASE_01_default.json                              TC_BASE_01: 默认场景, AOI A_3_4, 5 UAV + 2 HELI + 4 目标
TC_BASE_02_uav_no_target.json                        TC_BASE_02: 有 UAV 与 HELI 但无目标
TC_BASE_03_targets_insufficient_heli.json            TC_BASE_03: 目标较多但直升机数量不足 (1 HELI vs 5 目标)
TC_BASE_04_plenty_platform_few_targets.json          TC_BASE_04: 平台充足但目标很少 (8 UAV + 4 HELI vs 1 目标)
TC_HELI_01_zero_heli.json                            TC_HELI_01: HELI 数量为 0 (无打击平台)
TC_HELI_02_one_heli.json                             TC_HELI_02: HELI 数量为 1
TC_HELI_03_two_heli.json                             TC_HELI_03: HELI 数量为 2 (基准对照)
TC_HELI_04_four_heli.json                            TC_HELI_04: HELI 数量为 4 (打击平台过剩)
TC_HELI_05_low_hf.json                               TC_HELI_05: HF 弹药不足 (HF=1, RKT/GUN 正常)
TC_HELI_06_low_rkt.json                              TC_HELI_06: RKT 弹药不足 (RKT=2, HF/GUN 正常)
TC_HELI_07_low_gun.json                              TC_HELI_07: GUN 弹药不足 (GUN=10, HF/RKT 正常)
TC_HELI_08_no_ammo.json                              TC_HELI_08: 所有弹药为 0 (HELI 无可用弹药)
TC_HELI_09_ammo_enough_many_targets.json             TC_HELI_09: 弹药充足但目标过多 (2 HELI vs 8 目标)
TC_SENSOR_01_uav_only_eo.json                        TC_SENSOR_01: UAV 仅装备 EO
TC_SENSOR_02_uav_only_sar.json                       TC_SENSOR_02: UAV 仅装备 SAR
TC_SENSOR_03_uav_only_esm.json                       TC_SENSOR_03: UAV 仅装备 ESM
TC_SENSOR_04_uav_no_esm.json                         TC_SENSOR_04: UAV 缺少 ESM (仅 EO + SAR)
TC_SENSOR_05_uav_no_sar.json                         TC_SENSOR_05: UAV 缺少 SAR (仅 EO + ESM)
TC_SENSOR_06_heli_no_eoir.json                       TC_SENSOR_06: HELI 缺少 EOIR (仅 MMW)
TC_SENSOR_07_heli_no_mmw.json                        TC_SENSOR_07: HELI 缺少 MMW (仅 EOIR)
TC_STABILITY_01_small_pos_change.json                TC_STABILITY_01: 接近默认场景, 目标位置轻微变化
TC_STABILITY_02_small_weather_change.json            TC_STABILITY_02: 接近默认场景, 天气值轻微变化且不跨阈值
TC_STABILITY_03_small_value_change.json              TC_STABILITY_03: 接近默认场景, 目标价值轻微变化
TC_STABILITY_04_tie_break_value.json                 TC_STABILITY_04: 多个目标价值完全相同, 测试 tie-break
TC_STABILITY_05_very_close_targets.json              TC_STABILITY_05: 多个目标距离非常接近, 测试分配稳定性
TC_STABILITY_06_av_opposite_velocity.json            TC_STABILITY_06: 两个 AV 目标速度方向相反
TC_STABILITY_07_av_small_cov.json                    TC_STABILITY_07: AV 目标位置协方差较小
TC_STABILITY_08_av_large_cov.json                    TC_STABILITY_08: AV 目标位置协方差较大
TC_STRESS_01_few_platform_many_target_bad_weather.json TC_STRESS_01: 组合压力 (平台少 + 目标多 + 天气恶劣 + 弹药紧张)
TC_STRESS_02_extreme_aoi_far_staging_mixed.json      TC_STRESS_02: 组合压力 (极端 AOI A_5_5 + 远集结点 + 混合阈值天气 + 多类型目标)
TC_STRESS_03_min_resources_max_targets.json          TC_STRESS_03: 组合压力 (最少资源 + 最多目标, 无弹药 + 无 UAV 传感器)
TC_TARGET_01_no_target.json                          TC_TARGET_01: 无目标
TC_TARGET_02_one_radar.json                          TC_TARGET_02: 只有 1 个 RADAR
TC_TARGET_03_one_cp.json                             TC_TARGET_03: 只有 1 个 CP
TC_TARGET_04_one_av.json                             TC_TARGET_04: 只有 1 个 AV
TC_TARGET_05_multi_radar.json                        TC_TARGET_05: 多个 RADAR
TC_TARGET_06_multi_cp.json                           TC_TARGET_06: 多个 CP
TC_TARGET_07_multi_av.json                           TC_TARGET_07: 多个 AV
TC_TARGET_08_mixed.json                              TC_TARGET_08: RADAR + CP + AV 混合
TC_TARGET_09_equal_value.json                        TC_TARGET_09: 所有目标价值相同 (value 全部 0.8)
TC_TARGET_10_equal_threat.json                       TC_TARGET_10: 所有目标威胁相同 (threat 全部 0.7)
TC_TARGET_11_value_threat_contrast.json              TC_TARGET_11: 高价值低威胁与低价值高威胁目标同时存在
TC_TARGET_12_close_positions.json                    TC_TARGET_12: 目标位置非常接近 (聚集在小范围内)
TC_TARGET_13_dispersed.json                          TC_TARGET_13: 目标分散在不同区域 (四角分布)
TC_UAV_01_zero_uav.json                              TC_UAV_01: UAV 数量为 0
TC_UAV_02_one_uav.json                               TC_UAV_02: UAV 数量为 1
TC_UAV_03_two_uav.json                               TC_UAV_03: UAV 数量为 2
TC_UAV_04_four_uav.json                              TC_UAV_04: UAV 数量为 4
TC_UAV_05_eight_uav.json                             TC_UAV_05: UAV 数量为 8 (侦察资源过剩)
TC_UAV_06_enough_uav_no_sensor.json                  TC_UAV_06: UAV 数量充足但无可用传感器 (sensors 为空)
TC_WEATHER_01_all_good.json                          TC_WEATHER_01: 全部子区天气良好 (低天气值)
TC_WEATHER_02_all_bad.json                           TC_WEATHER_02: 全部子区天气恶劣 (高天气值)
TC_WEATHER_03_threshold_030.json                     TC_WEATHER_03: 天气值围绕 0.30 阈值 (0.29 / 0.30)
TC_WEATHER_04_threshold_045.json                     TC_WEATHER_04: 天气值围绕 0.45 阈值 (0.44 / 0.45)
TC_WEATHER_05_threshold_065.json                     TC_WEATHER_05: 天气值围绕 0.65 阈值 (0.64 / 0.65)
TC_WEATHER_06_threshold_080.json                     TC_WEATHER_06: 天气值围绕 0.80 阈值 (0.79 / 0.80)
TC_WEATHER_07_mixed.json                             TC_WEATHER_07: 混合天气, 五个子区分别跨越各档阈值
