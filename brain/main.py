#!/usr/bin/env python3
"""Compatibility launcher for the C++ Brain demo entry.

The mission demo has been moved to ``brain/main.cpp``.  This file is kept only
so old commands such as ``python brain/main.py`` still work during the C++
migration.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    source = here / "main.cpp"
    build_dir = here.parent / "build"
    binary = build_dir / "brain_demo"

    if not source.exists():
        raise FileNotFoundError(f"C++ entry not found: {source}")

    needs_build = (
        not binary.exists()
        or source.stat().st_mtime > binary.stat().st_mtime
    )
    if needs_build:
        build_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "g++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            str(source),
            "-o",
            str(binary),
        ]
        subprocess.check_call(cmd)

    os.execv(str(binary), [str(binary), *sys.argv[1:]])


if __name__ == "__main__":
    main()
