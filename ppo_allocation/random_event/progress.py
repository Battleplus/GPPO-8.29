"""Observability-only exact progress heartbeat utilities.

The writer deliberately uses no random state and is best-effort: a heartbeat
failure must never change training state or interrupt an algorithm run.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_progress(path: str | Path, payload: dict[str, Any]) -> bool:
    """Atomically publish a progress payload; return False on any I/O error."""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, target)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        return True
    except (OSError, TypeError, ValueError):
        return False


def read_progress(path: str | Path) -> dict[str, Any] | None:
    """Read one complete heartbeat, returning None when unavailable."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


__all__ = ["read_progress", "write_progress"]
