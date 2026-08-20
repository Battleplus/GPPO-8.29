"""Helpers for loading external mission inputs at startup."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_mission_world(input_path: str | Path | None) -> dict[str, Any]:
    """Load a JSON mission input into a clean ``world_state`` dict."""
    if input_path is None:
        return {}
    path = Path(input_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    data = strip_metadata_fields(raw)
    if isinstance(data, dict) and isinstance(data.get("world_state"), dict):
        return dict(data["world_state"])
    if not isinstance(data, dict):
        raise ValueError(f"mission input must be a JSON object: {path}")
    return dict(data)


def strip_metadata_fields(value: Any) -> Any:
    """Remove template-only keys such as ``_说明`` before runtime use."""
    if isinstance(value, dict):
        return {
            key: strip_metadata_fields(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [strip_metadata_fields(item) for item in value]
    return value


def apply_task_area_overrides(
    world: dict[str, Any],
    *,
    aoi: str | dict[str, Any] | None = None,
    aois: str | list[Any] | None = None,
) -> dict[str, Any]:
    """Apply startup AOI overrides to a mutable ``world_state`` dict."""
    if aois:
        parsed_aois = parse_aoi_list(aois)
        world["aois"] = parsed_aois
        world["aoi"] = {
            "row": parsed_aois[0]["row"],
            "col": parsed_aois[0]["col"],
        }
        world["commander_AOI"] = [item["id"] for item in parsed_aois]
    elif aoi:
        parsed = parse_aoi(aoi)
        world["aoi"] = {"row": parsed["row"], "col": parsed["col"]}
        world["commander_AOI"] = [parsed["id"]]
    return world


def parse_aoi_list(raw: str | list[Any] | tuple[Any, ...]) -> list[dict[str, Any]]:
    """Parse one or more AOIs from strings/dicts/lists."""
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        if "," in raw and not _looks_like_row_col(raw):
            items: list[Any] = [item.strip() for item in raw.split(",") if item.strip()]
        else:
            items = [raw]
    else:
        items = list(raw)
    return [parse_aoi(item) for item in items]


def parse_aoi(raw: Any) -> dict[str, Any]:
    """Normalize AOI input into ``{"id": "A_r_c", "row": r, "col": c}``."""
    extra: dict[str, Any] = {}
    if isinstance(raw, dict):
        extra = {
            key: value
            for key, value in raw.items()
            if key not in {"row", "col", "id", "aoi"}
        }
        if raw.get("row") is not None and raw.get("col") is not None:
            row, col = int(raw["row"]), int(raw["col"])
            aoi_id = str(raw.get("id") or raw.get("aoi") or f"A_{row}_{col}")
            return {**extra, "id": aoi_id, "row": row, "col": col}
        raw = raw.get("id") or raw.get("aoi")

    if isinstance(raw, str):
        value = raw.strip()
        match = re.match(r"^A[_-]?(\d+)[_-](\d+)$", value, re.IGNORECASE)
        if match:
            row, col = int(match.group(1)), int(match.group(2))
            return {**extra, "id": f"A_{row}_{col}", "row": row, "col": col}
        match = re.match(r"^(\d+)\s*[,:\s]\s*(\d+)$", value)
        if match:
            row, col = int(match.group(1)), int(match.group(2))
            return {**extra, "id": f"A_{row}_{col}", "row": row, "col": col}

    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        row, col = int(raw[0]), int(raw[1])
        return {**extra, "id": f"A_{row}_{col}", "row": row, "col": col}

    raise ValueError(
        "AOI must be like {'row':3,'col':4}, 'A_3_4', or '3,4'"
    )


def _looks_like_row_col(raw: str) -> bool:
    return bool(re.match(r"^\d+\s*,\s*\d+$", raw.strip()))
