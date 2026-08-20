from __future__ import annotations

import sys
from pathlib import Path


QL_ROOT = Path(__file__).resolve().parent
if str(QL_ROOT) not in sys.path:
    sys.path.insert(0, str(QL_ROOT))

import run_isaacsim_replay


def main() -> None:
    run_isaacsim_replay.main()


if __name__ == "__main__":
    main()

