"""
Streamlit 可视化入口 —— UAV–HELI 协同侦察打击任务分配前端。

启动方式:
    streamlit run frontend_app.py

功能:
  - 单 AOI 模式：侧边栏选择场景、求解器 → 运行 → 时间步动画
  - 多 AOI 模式：AOI 排序 + 逐个执行 → 进度追踪 + 历史表格
"""

from __future__ import annotations
import sys
import os
import traceback
import copy

# 确保项目根在 sys.path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd

# ── 页面配置（必须是第一个 Streamlit 命令） ──────────
st.set_page_config(
    page_title="UAV–HELI 任务分配可视化",
    page_icon="🚁",
    layout="wide",
)

from task_interface import TaskAllocator
from visualization.scenario import load_default_snapshot, load_scenario_by_name, list_scenario_names
from visualization.state_builder import build_visualization_state, build_multi_aoi_visualization_state
from visualization.battlefield_map import render_figure


# ── 多 AOI 场景列表 ─────────────────────────────────────
def _list_multi_aoi_scenarios() -> list:
    """列出 scenarios/ 目录下所有 multi_aoi_ 前缀的 JSON 文件。"""
    import glob
    scenarios_root = os.path.join(_project_root, "scenarios")
    pattern = os.path.join(scenarios_root, "multi_aoi_*.json")
    names = []
    for fpath in sorted(glob.glob(pattern)):
        base = os.path.splitext(os.path.basename(fpath))[0]
        names.append(base)
    return names or ["multi_aoi_example"]


def _load_multi_aoi_input(scenario_name: str) -> dict:
    """按名称加载多 AOI 场景 JSON。"""
    from multi_aoi_interface import load_multi_aoi_request_from_json
    filepath = os.path.join(_project_root, "scenarios", f"{scenario_name}.json")
    return load_multi_aoi_request_from_json(filepath)


# ── 标题区 ────────────────────────────────────────────
st.title("UAV–HELI 协同侦察打击 · 任务分配可视化")
st.caption("平台从集结区出发 → 经任务分配 → 移动到指定栅格 / 目标")


# ── 侧边栏 ────────────────────────────────────────────
st.sidebar.header("运行模式")

mode = st.sidebar.radio(
    "模式",
    options=["单 AOI（原版）", "多 AOI"],
    index=0,
    help="单 AOI：传统单个区域任务分配；多 AOI：多区域排序 + 依次执行",
)

is_multi_aoi = (mode == "多 AOI")

# ── 场景选择 ──────────────────────────────────────────
if is_multi_aoi:
    st.sidebar.header("多 AOI 场景选择")
    scenario_names = _list_multi_aoi_scenarios()
    selected_scenario = st.sidebar.selectbox(
        "场景",
        options=scenario_names,
        index=0,
        help="从 scenarios/ 目录加载 multi_aoi_*.json",
    )
else:
    st.sidebar.header("场景选择")
    scenario_names = list_scenario_names()
    if not scenario_names:
        scenario_names = ["default"]
    selected_scenario = st.sidebar.selectbox(
        "场景",
        options=scenario_names,
        index=0,
        help="从 scenarios/ 目录加载 JSON 场景文件",
    )

# ── 求解器设置 ────────────────────────────────────────
st.sidebar.header("求解器设置")

solver_choice = st.sidebar.selectbox(
    "求解器",
    options=["cbc", "highs", "ortools", "gurobi"],
    index=0,
    help="CBC 为默认开源求解器；HiGHS/OR-Tools 为备选；Gurobi 需商业授权。",
)

time_limit = st.sidebar.slider(
    "时间上限 (秒)",
    min_value=1.0, max_value=60.0, value=5.0, step=1.0,
    help="单次 MILP 求解时间上限。",
)

# ── 会话状态初始化 ────────────────────────────────────
if "viz" not in st.session_state:
    st.session_state.viz = None
if "loaded_scenario" not in st.session_state:
    st.session_state.loaded_scenario = None
if "mode" not in st.session_state:
    st.session_state.mode = "single"
if "multi_aoi_input" not in st.session_state:
    st.session_state.multi_aoi_input = None
if "multi_aoi_result" not in st.session_state:
    st.session_state.multi_aoi_result = None
if "multi_aoi_history" not in st.session_state:
    st.session_state.multi_aoi_history = []
if "multi_aoi_allocator" not in st.session_state:
    st.session_state.multi_aoi_allocator = None

# 模式切换时清空旧状态
if st.session_state.mode != ("multi" if is_multi_aoi else "single"):
    st.session_state.mode = "multi" if is_multi_aoi else "single"
    st.session_state.viz = None
    st.session_state.loaded_scenario = None
    st.session_state.multi_aoi_input = None
    st.session_state.multi_aoi_result = None
    st.session_state.multi_aoi_history = []

# ── 单 AOI 模式：场景加载 ──────────────────────────────
if not is_multi_aoi:
    if st.session_state.loaded_scenario != selected_scenario:
        with st.spinner(f"加载场景: {selected_scenario}..."):
            try:
                st.session_state.snapshot = load_scenario_by_name(selected_scenario)
                st.session_state.loaded_scenario = selected_scenario
                st.session_state.viz = None
            except FileNotFoundError:
                st.session_state.snapshot = load_default_snapshot(with_targets=True)
                st.session_state.loaded_scenario = selected_scenario
                st.session_state.viz = None
    elif "snapshot" not in st.session_state:
        with st.spinner("加载默认场景..."):
            st.session_state.snapshot = load_default_snapshot(with_targets=True)
        st.session_state.loaded_scenario = selected_scenario

# ── 运行按钮 ──────────────────────────────────────────
if is_multi_aoi:
    run_label = "运行任务分配" if st.session_state.multi_aoi_input is None else "运行任务分配（当前 AOI）"
else:
    run_label = "运行任务分配"

run_button = st.sidebar.button(run_label, type="primary", use_container_width=True)

# ── 多 AOI：侦察反馈按钮（同一 AOI 内侦察→打击循环） ──
if is_multi_aoi:
    recon_feedback_disabled = (
        st.session_state.multi_aoi_result is None
        or st.session_state.multi_aoi_result.get("status") == "ALL_AOI_FINISHED"
    )
    recon_feedback_button = st.sidebar.button(
        "提交侦察反馈 → 重分配当前 AOI",
        use_container_width=True,
        disabled=recon_feedback_disabled,
        help="模拟 UAV 完成侦察，将当前 AOI 内目标标记为已探测，然后重新分配（含打击任务）",
    )
else:
    recon_feedback_button = False

# ── 多 AOI：下一 AOI 按钮 ──────────────────────────────
if is_multi_aoi:
    next_aoi_disabled = (
        st.session_state.multi_aoi_result is None
        or st.session_state.multi_aoi_result.get("status") == "ALL_AOI_FINISHED"
    )
    next_aoi_button = st.sidebar.button(
        "推进到下一 AOI ▶",
        use_container_width=True,
        disabled=next_aoi_disabled,
    )
    # 重置按钮
    reset_button = st.sidebar.button("重置多 AOI 流程", use_container_width=True,
                                     disabled=st.session_state.multi_aoi_input is None)
else:
    next_aoi_button = False
    reset_button = False

# ── 执行求解（单 AOI） ────────────────────────────────
if run_button and not is_multi_aoi:
    with st.spinner(f"正在用 {solver_choice.upper()} 求解（时限 {time_limit:.0f}s）..."):
        try:
            allocator = TaskAllocator(
                solver=solver_choice,
                time_limit_s=float(time_limit),
                verbose=0,
            )
            snap = st.session_state.snapshot
            plan = allocator.solve(snap)
            st.session_state.viz = build_visualization_state(snap, plan)
            st.success(f"求解完成！状态: {plan.status}")
        except Exception as exc:
            st.error(f"求解失败: {exc}")
            st.code(traceback.format_exc())

# ── 执行求解（多 AOI） ────────────────────────────────
if run_button and is_multi_aoi:
    with st.spinner(f"正在用 {solver_choice.upper()} 求解多 AOI（时限 {time_limit:.0f}s）..."):
        try:
            from multi_aoi_interface import MultiAOITaskAllocator

            # 首次运行或求解器变更时创建新分配器
            if (st.session_state.multi_aoi_allocator is None
                    or st.session_state.loaded_scenario != selected_scenario):
                st.session_state.multi_aoi_allocator = MultiAOITaskAllocator(
                    solver=solver_choice,
                    time_limit_s=float(time_limit),
                    verbose=0,
                )
                st.session_state.multi_aoi_input = _load_multi_aoi_input(selected_scenario)
                st.session_state.multi_aoi_history = []
                st.session_state.loaded_scenario = selected_scenario

            allocator = st.session_state.multi_aoi_allocator
            input_data = st.session_state.multi_aoi_input

            # 如果已有上一次结果，带回状态
            if st.session_state.multi_aoi_result is not None:
                prev = st.session_state.multi_aoi_result
                input_data["aoi_route_state"] = prev["aoi_route_state"]
                # 不自动带入 execution_feedback——用户点"下一 AOI"时手动设置

            result = allocator.run(input_data)
            st.session_state.multi_aoi_result = result

            # 构造可视化
            st.session_state.viz = build_multi_aoi_visualization_state(input_data, result)

            solved_aoi = result["aoi_route_state"].get("current_aoi", "?")
            st.success(f"多 AOI 求解完成！状态: {result['status']}  |  当前 AOI: {solved_aoi}")
        except Exception as exc:
            st.error(f"求解失败: {exc}")
            st.code(traceback.format_exc())

# ── 多 AOI：侦察反馈（同一 AOI 内重新分配） ──────────
if recon_feedback_button and is_multi_aoi:
    result = st.session_state.multi_aoi_result
    input_data = st.session_state.multi_aoi_input

    if result is not None and result["status"] != "ALL_AOI_FINISHED":
        # 找出当前 AOI 内的所有目标 id
        current_aoi = result["aoi_route_state"].get("current_aoi", "")
        aoi_info = next((a for a in input_data.get("aois", []) if a.get("id") == current_aoi), None)
        detected = []
        if aoi_info:
            row, col = int(aoi_info.get("row", 0)), int(aoi_info.get("col", 0))
            for t in input_data.get("targets", []):
                tx, ty = t["pos"]
                if (col - 1) * 50 <= tx <= col * 50 and (row - 1) * 50 <= ty <= row * 50:
                    detected.append(t["tid"])

        # 设置反馈（aoi_status=RUNNING 不推进 AOI）
        input_data["aoi_route_state"] = result["aoi_route_state"]
        input_data["execution_feedback"] = {
            "aoi_id": current_aoi,
            "aoi_status": "RUNNING",
            "detected_targets": detected,
            "coverage_rate": 0.90,
        }

        with st.spinner(f"侦察完成，重新分配 {current_aoi}（已探测 {len(detected)} 个目标）..."):
            try:
                result2 = st.session_state.multi_aoi_allocator.run(input_data)
                st.session_state.multi_aoi_result = result2
                st.session_state.viz = build_multi_aoi_visualization_state(input_data, result2)
                n_strikes = sum(1 for t in result2.get("current_aoi_plan", {}).get("tasks", [])
                               if t.get("task_type") == "strike")
                st.success(f"重分配完成！已探测目标: {', '.join(detected) or '(无)'}  |  打击任务: {n_strikes}")
            except Exception as exc:
                st.error(f"重分配失败: {exc}")
                st.code(traceback.format_exc())

# ── 多 AOI：推进到下一 AOI ────────────────────────────
if next_aoi_button and is_multi_aoi:
    result = st.session_state.multi_aoi_result
    input_data = st.session_state.multi_aoi_input

    if result is not None and result["status"] != "ALL_AOI_FINISHED":
        # 记录当前 AOI 完成状态到历史
        current_state = result["aoi_route_state"]
        completed_aoi = current_state.get("current_aoi", "?")
        plan = result.get("current_aoi_plan", {})
        st.session_state.multi_aoi_history.append({
            "aoi": completed_aoi,
            "solve_status": plan.get("solve_status", "FINISHED"),
            "objective": plan.get("objective", 0),
            "tasks_count": len(plan.get("tasks", [])),
            "solve_time_ms": plan.get("solve_time_ms", 0),
        })

        # 设置 feedback 并重新运行
        input_data["aoi_route_state"] = current_state
        input_data["execution_feedback"] = {
            "aoi_id": completed_aoi,
            "aoi_status": "FINISHED",
        }
        input_data["cycle_id"] = input_data.get("cycle_id", 0) + 1

        with st.spinner(f"推进到下一 AOI（{current_state.get('next_aoi', '?')}）..."):
            try:
                result2 = st.session_state.multi_aoi_allocator.run(input_data)
                st.session_state.multi_aoi_result = result2
                st.session_state.viz = build_multi_aoi_visualization_state(input_data, result2)

                if result2["status"] == "ALL_AOI_FINISHED":
                    st.success("所有 AOI 执行完成！")
                    # 记录最后一个 AOI 完成
                    last_state = result2.get("aoi_route_state", {})
                    last_plan = result2.get("current_aoi_plan")
                    if last_plan is None:
                        # ALL_AOI_FINISHED 时 current_aoi_plan 为 null，从 history 推算
                        last_aoi = last_state.get("aoi_sequence", [])[-1] if last_state.get("aoi_sequence") else "?"
                        st.session_state.multi_aoi_history.append({
                            "aoi": last_aoi,
                            "solve_status": "FINISHED",
                            "objective": 0,
                            "tasks_count": 0,
                            "solve_time_ms": 0,
                        })
                    st.rerun()
                else:
                    current = result2["aoi_route_state"].get("current_aoi", "?")
                    st.success(f"已推进到: {current}")
            except Exception as exc:
                st.error(f"推进失败: {exc}")
                st.code(traceback.format_exc())

# ── 多 AOI：重置 ──────────────────────────────────────
if reset_button and is_multi_aoi:
    st.session_state.multi_aoi_input = None
    st.session_state.multi_aoi_result = None
    st.session_state.multi_aoi_history = []
    st.session_state.viz = None
    st.rerun()


# ── 多 AOI 进度面板 ────────────────────────────────────
if is_multi_aoi and st.session_state.multi_aoi_result is not None:
    result = st.session_state.multi_aoi_result
    route = result.get("aoi_route_state", {})
    seq = route.get("aoi_sequence", [])
    cur = route.get("current_aoi_index", 0)
    total = len(seq)
    route_status = route.get("route_status", "RUNNING")

    st.markdown("### AOI 执行进度")

    # 进度条
    progress_val = min(cur / max(total, 1), 1.0) if route_status == "ALL_AOI_FINISHED" else min((cur + 1) / max(total, 1), 1.0)
    if route_status == "ALL_AOI_FINISHED":
        progress_val = 1.0

    cols = st.columns([3, 1, 1])
    cols[0].progress(progress_val)
    cols[1].metric("已完成", str(min(cur + (0 if route_status == "ALL_AOI_FINISHED" else 0), total)) if route_status != "ALL_AOI_FINISHED" else str(total))
    cols[2].metric("总计", str(total))

    # AOI 状态卡片
    card_cols = st.columns(min(len(seq), 4))
    for i, aoi_id in enumerate(seq):
        with card_cols[i % len(card_cols)]:
            if route_status == "ALL_AOI_FINISHED":
                emoji, label = "✅", "已完成"
            elif i < cur:
                emoji, label = "✅", "已完成"
            elif i == cur:
                emoji, label = "📍", "执行中"
            else:
                emoji, label = "⏳", "待执行"
            st.markdown(f"**{emoji} #{i+1}** {aoi_id}  \n<small>{label}</small>", unsafe_allow_html=True)

    st.markdown("---")


# ── 求解状态摘要 ──────────────────────────────────────
viz = st.session_state.viz

if viz is not None:
    meta = viz.get("meta", {})
    if is_multi_aoi:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("整体状态", meta.get("route_status", meta.get("status", "N/A")))
        col2.metric("当前 AOI", meta.get("current_aoi", "N/A"))
        col3.metric("求解状态", meta.get("status", "N/A"))
        col4.metric("目标函数值", f"{meta.get('objective', 0):.3f}")
        col5.metric("求解耗时", f"{meta.get('solve_time_ms', 0):.1f} ms")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("求解状态", meta.get("status", "N/A"))
        col2.metric("目标函数值", f"{meta.get('objective', 0):.3f}")
        col3.metric("求解耗时", f"{meta.get('solve_time_ms', 0):.1f} ms")
        col4.metric("求解器", meta.get("solver_used", "N/A"))


# ── 主区：时间步 + 地图 ───────────────────────────────
multi_aoi_data = None
if is_multi_aoi and viz is not None:
    multi_aoi_data = {"aois": viz.get("aois", []), "aoi_route": viz.get("aoi_route", {}),
                      "meta": viz.get("meta", {})}

if viz is None:
    # 未运行分配前：显示 t0 静态预览
    if is_multi_aoi:
        st.info("点击左侧「运行任务分配」按钮，对多 AOI 场景求解。")
        # 加载场景预览
        try:
            preview_input = _load_multi_aoi_input(selected_scenario)
            preview_viz = build_multi_aoi_visualization_state(
                preview_input,
                {"status": "UNSOLVED", "aoi_route_state": {}, "current_aoi_plan": None},
            )
            multi_aoi_data = {"aois": preview_viz.get("aois", []),
                              "aoi_route": preview_viz.get("aoi_route", {}),
                              "meta": preview_viz.get("meta", {})}
            frames = preview_viz.get("animation_frames", [])
            st.plotly_chart(
                render_figure(preview_viz, frame_idx=0, multi_aoi_data=multi_aoi_data),
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"预览加载失败: {exc}")
    else:
        st.info("点击左侧「运行任务分配」按钮，对默认场景求解。当前显示 t0 集结区预览。")
        snapshot = st.session_state.snapshot
        preview_viz = build_visualization_state(snapshot, {
            "meta": {"status": "UNSOLVED"},
        })
        st.plotly_chart(
            render_figure(preview_viz, frame_idx=0),
            use_container_width=True,
        )
else:
    frames = viz.get("animation_frames", [])
    n_frames = len(frames)

    if n_frames > 0:
        # 时间步 slider
        frame_labels = []
        for f_ in frames:
            phase = f_["phase"]
            if phase == "t2":
                tau = f_.get("tau", 0)
                frame_labels.append(f"t2@{tau:.1f}")
            else:
                frame_labels.append(phase)

        frame_idx = st.select_slider(
            "时间步",
            options=list(range(n_frames)),
            format_func=lambda i: frame_labels[i] if i < len(frame_labels) else str(i),
            value=0,
        )

        # Plotly 地图
        fig = render_figure(viz, frame_idx=frame_idx, multi_aoi_data=multi_aoi_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("无动画帧数据。")

    # ── 下方表格 ───────────────────────────────────────
    st.markdown("---")

    if is_multi_aoi:
        tab1, tab2, tab3 = st.tabs(["侦察分配", "打击分配", "AOI 执行历史"])
    else:
        tab1, tab2 = st.tabs(["侦察分配", "打击分配"])
        tab3 = None

    with tab1:
        recon = viz.get("recon_assignments", [])
        if recon:
            df_recon = pd.DataFrame(recon)
            col_map = {c: c for c in df_recon.columns}
            st.dataframe(df_recon, use_container_width=True, hide_index=True)
        else:
            st.info("无侦察分配")

    with tab2:
        strike = viz.get("strike_assignments", [])
        if strike:
            df_strike = pd.DataFrame(strike)
            st.dataframe(df_strike, use_container_width=True, hide_index=True)
        else:
            st.info("无打击分配")

    if tab3 is not None:
        with tab3:
            history = st.session_state.multi_aoi_history
            if history:
                df_hist = pd.DataFrame(history)
                df_hist.columns = ["AOI", "求解状态", "目标函数", "任务数", "耗时(ms)"]
                st.dataframe(df_hist, use_container_width=True, hide_index=True)
            else:
                st.info("尚无已完成的 AOI（点「推进到下一 AOI」后记录）")


# ── 页脚 ──────────────────────────────────────────────
st.markdown("---")
mode_label = "多 AOI" if is_multi_aoi else "单 AOI"
st.caption(
    f"模式: {mode_label} | 场景: {selected_scenario} | 求解器: {solver_choice.upper()}"
)
