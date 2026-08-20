import json
import math
from pathlib import Path

from perch.cpp_bridge import OUTPUT_FIELDS, _write_positions, select_positions


def _request():
    return {
        "mission_id": "PY_CPP_BRIDGE_TEST",
        "world_state": {
            "meters_per_unit": 100.0,
            "map_size_km": 300.0,
            "targets": [{
                "tid": "g1",
                "type": "RADAR",
                "pos": [150.0, 150.0],
                "value": 1.0,
                "threat": 0.9,
                "confirmed": True,
                "alive": True,
            }],
        },
        "agents": [{
            "pid": "H1",
            "type": "HELI",
            "position": [140.0, 150.0],
            "altitude_km": 0.3,
            "sensors": ["EOIR"],
            "munitions": {"HF": 4},
        }],
        "tasks": [{
            "platform": "H1",
            "target": "g1",
            "munition": "HF",
            "qty": 1,
            "assigned_munitions": {"HF": 1},
        }],
        "options": {
            "attack_region_mode": "demo",
            "terrain_mode": "flat",
            "preference": "balanced",
            "top_k": 1,
            "use_pymoo": False,
        },
    }


def test_cpp_bridge_runs_situation_region_and_frea(tmp_path: Path):
    positions, audit = select_positions(_request())

    assert len(positions) == 1
    metadata = positions[0].metadata
    assert metadata["platform_id"] == "H1"
    assert metadata["target_id"] == "g1"
    assert metadata["g_violation"] == 0.0
    assert metadata["attack_region_source"] == "perch:local_fallback"
    assert "[任务]" in audit["situations"]["g1"]

    output = tmp_path / "positions.tsv"
    _write_positions(output, positions, audit)
    rows = output.read_text(encoding="utf-8").splitlines()
    assert rows[0].split("\t") == list(OUTPUT_FIELDS)
    assert len(rows[1].split("\t")) == len(OUTPUT_FIELDS)
    assert "\\n" in rows[1]
    values = dict(zip(OUTPUT_FIELDS, rows[1].split("\t"), strict=True))
    assert values["coordinate_frame"] == "mission_km"
    assert 2.0 <= math.hypot(
        float(values["x"]) - 150.0,
        float(values["y"]) - 150.0,
    ) <= 8.0


def test_cpp_bridge_audit_is_json_serializable():
    _, audit = select_positions(_request())

    encoded = json.dumps(audit, ensure_ascii=False)

    assert "attack_regions" in encoded
    assert "g1" in encoded
