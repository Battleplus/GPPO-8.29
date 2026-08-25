from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


class BuildP0GateBootstrapTests(unittest.TestCase):
    def test_script_module_loads_from_repo_root_without_path_override(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        script = repo / "scripts" / "build_p0_gate.py"
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        probe = (
            "import importlib.util, pathlib, sys; "
            "p=pathlib.Path(sys.argv[1]); "
            "s=importlib.util.spec_from_file_location('gate_bootstrap_probe', p); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert m.ROOT == p.resolve().parents[1]; "
            "assert (m.ROOT / 'ppo_allocation').is_dir()"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe, str(script)],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)
        self.assertNotIn("ImportError", result.stderr)
        self.assertNotIn("No module named", result.stderr)
