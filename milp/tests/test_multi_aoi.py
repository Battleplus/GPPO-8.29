"""
多 AOI 排序与控制逻辑单元测试。

运行方式:
    cd 代码-v2
    python -m pytest tests/test_multi_aoi.py -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aoi.aoi_state import AoiInfo, AOIRouteState, ExecutionFeedback
from aoi.aoi_router import AOIRouter


# ── 辅助数据 ─────────────────────────────────────────────

def make_aois():
    return [
        AoiInfo(id="A_5_6", row=5, col=6, priority=1.0,
                target_prior=0.4, target_value=0.9, target_threat=0.85),
        AoiInfo(id="A_3_4", row=3, col=4, priority=0.8,
                target_prior=0.3, target_value=0.75, target_threat=0.6),
        AoiInfo(id="A_1_5", row=1, col=5, priority=0.7,
                target_prior=0.5, target_value=0.6, target_threat=0.5),
    ]


# ── AOIRouteState 测试 ────────────────────────────────────

class TestAOIRouteState:
    def test_initial_state(self):
        state = AOIRouteState(
            aoi_sequence=["A_5_6", "A_3_4", "A_1_5"],
            current_aoi_index=0,
            route_status="RUNNING",
        )
        assert state.current_aoi == "A_5_6"
        assert state.next_aoi == "A_3_4"
        assert not state.is_finished()

    def test_advance(self):
        state = AOIRouteState(
            aoi_sequence=["A_5_6", "A_3_4"],
            current_aoi_index=0,
            route_status="RUNNING",
        )
        state.advance()
        assert state.current_aoi == "A_3_4"
        assert state.next_aoi is None

    def test_all_finished(self):
        state = AOIRouteState(
            aoi_sequence=["A_5_6"],
            current_aoi_index=0,
            route_status="RUNNING",
        )
        state.advance()
        assert state.is_finished()
        assert state.route_status == "ALL_FINISHED"
        assert state.current_aoi is None

    def test_serialization_roundtrip(self):
        state = AOIRouteState(
            aoi_sequence=["A_5_6", "A_3_4", "A_1_5"],
            current_aoi_index=1,
            route_status="RUNNING",
        )
        d = state.to_dict()
        restored = AOIRouteState.from_dict(d)
        assert restored.aoi_sequence == state.aoi_sequence
        assert restored.current_aoi_index == 1
        assert restored.current_aoi == "A_3_4"


# ── AOIRouter 测试 ────────────────────────────────────────

class TestAOIRouter:
    def test_single_aoi(self):
        router = AOIRouter()
        aois = [AoiInfo(id="A_3_4", row=3, col=4)]
        state = router.sort(aois)
        assert state.aoi_sequence == ["A_3_4"]
        assert state.current_aoi_index == 0

    def test_sort_returns_valid_sequence(self):
        router = AOIRouter()
        aois = make_aois()
        state = router.sort(aois, start_pos=np.array([142.0, -38.0]))
        assert len(state.aoi_sequence) == 3
        assert set(state.aoi_sequence) == {"A_5_6", "A_3_4", "A_1_5"}
        assert state.route_status == "RUNNING"

    def test_sort_empty_raises(self):
        router = AOIRouter()
        with pytest.raises(ValueError):
            router.sort([])

    def test_high_priority_first(self):
        """优先级最高的 AOI 应最优先（在出发点较近时）。"""
        router = AOIRouter()
        aois = [
            AoiInfo(id="A_1_1", row=1, col=1, priority=1.0,
                    target_prior=0.9, target_value=1.0, target_threat=1.0),
            AoiInfo(id="A_6_6", row=6, col=6, priority=0.1,
                    target_prior=0.1, target_value=0.1, target_threat=0.1),
        ]
        # 出发点靠近 A_1_1
        state = router.sort(aois, start_pos=np.array([0.0, 0.0]))
        assert state.aoi_sequence[0] == "A_1_1"


# ── ExecutionFeedback 测试 ────────────────────────────────

class TestExecutionFeedback:
    def test_from_dict_minimal(self):
        d = {"aoi_id": "A_5_6", "aoi_status": "FINISHED"}
        fb = ExecutionFeedback.from_dict(d)
        assert fb.aoi_id == "A_5_6"
        assert fb.aoi_status == "FINISHED"
        assert fb.coverage_rate == 0.0

    def test_from_dict_full(self):
        d = {
            "aoi_id": "A_5_6",
            "aoi_status": "FINISHED",
            "coverage_rate": 0.85,
            "detected_targets": ["g1"],
            "destroyed_targets": ["g1"],
            "elapsed_time": 12.5,
        }
        fb = ExecutionFeedback.from_dict(d)
        assert fb.coverage_rate == 0.85
        assert "g1" in fb.detected_targets
