"""Compatibility import for the PPO reallocation service.

The implementation lives in ``ppo_allocation/reallocation_service.py``.  This
module keeps older callers such as ``from reallocation_service import ...``
working from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PPO_DIR = Path(__file__).resolve().parent / "ppo_allocation"
if str(_PPO_DIR) not in sys.path:
    sys.path.insert(0, str(_PPO_DIR))

from ppo_allocation.reallocation_service import *  # noqa: F401,F403,E402
