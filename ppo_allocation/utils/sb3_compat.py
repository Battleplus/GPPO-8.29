"""Small import bootstrap for reproducible Stable-Baselines3 inference.

The bundled checkpoint was serialized with NumPy 2, while some deployment
machines still use NumPy 1.26.  Conversely, this repository may be executed in
an Anaconda environment containing optional binary packages built for NumPy 1.
This function removes those unrelated ABI hazards without changing the model.
"""

from __future__ import annotations

import importlib
import sys
import types


def prepare_sb3_import() -> None:
    """Prepare harmless compatibility aliases before importing SB3."""

    # Tell TensorBoard to use its bundled TensorFlow stub.  SB3 inference does
    # not use TensorFlow, and importing a host TensorFlow solely for logging can
    # otherwise introduce an unrelated NumPy ABI failure.
    sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

    # Pandas probes these optional accelerators.  They are not required by SB3
    # and may come from an incompatible system-site installation.
    for optional_binary in ("pyarrow", "numexpr", "bottleneck"):
        sys.modules.setdefault(optional_binary, None)

    # NumPy 2 renamed the private ``numpy.core`` package to ``numpy._core``.
    # The checkpoint records that private module path.  On NumPy 1.x, provide
    # read-only aliases to the corresponding modules so cloudpickle can restore
    # arrays; no numerical behavior is changed.
    try:
        importlib.import_module("numpy._core")
    except ImportError:
        legacy_core = importlib.import_module("numpy.core")
        sys.modules.setdefault("numpy._core", legacy_core)
    # NumPy 1.26 contains a compatibility ``numpy._core`` package but not all
    # private children recorded by a NumPy 2 pickle, so check each child rather
    # than treating the parent import as sufficient.
    for child in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        try:
            importlib.import_module(f"numpy._core.{child}")
            continue
        except ImportError:
            pass
        try:
            module = importlib.import_module(f"numpy.core.{child}")
        except ImportError:
            continue
        sys.modules.setdefault(f"numpy._core.{child}", module)


__all__ = ["prepare_sb3_import"]
