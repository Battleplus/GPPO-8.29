"""Adapter that keeps the trained 165-D MLP-MaskablePPO operational.

The checkpoint and its network are not rewritten.  The adapter supplies the
legacy flat observation/multi-discrete mask, then maps the first pending
Region's selected UAV back to the new edge-action index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from config import ActionCode, NUM_REGIONS, NUM_UAVS, TaskType
except ImportError:
    from ..config import ActionCode, NUM_REGIONS, NUM_UAVS, TaskType

from .graph import HeteroGraphState

try:
    from utils.sb3_compat import prepare_sb3_import
except ImportError:
    from ..utils.sb3_compat import prepare_sb3_import


def legacy_pending_mask(env) -> np.ndarray:
    mask = np.zeros((NUM_REGIONS, NUM_UAVS + 2), dtype=bool)
    pending = set(int(r) for r in getattr(env, "pending_regions", ()))
    for rid in range(NUM_REGIONS):
        if rid not in pending:
            mask[rid, int(ActionCode.KEEP)] = True
            continue
        legal = []
        for uid, uav in env.uavs.items():
            if uav.alive and not uav.sensor_failed and uav.task != TaskType.TRACK:
                legal.append(uid)
                mask[rid, uid + 1] = True
        if not legal:
            mask[rid, int(ActionCode.NO_UAV)] = True
    return mask.reshape(-1)


class LegacyMLPPPOPolicy:
    name = "MLP-PPO"

    def __init__(self, model_or_path: Any, deterministic: bool = True):
        if isinstance(model_or_path, (str, Path)):
            prepare_sb3_import()
            from sb3_contrib import MaskablePPO

            self.model = MaskablePPO.load(str(model_or_path))
        else:
            self.model = model_or_path
        self.deterministic = bool(deterministic)

    def select_action(self, env, graph: HeteroGraphState, deterministic: bool = True) -> int:
        if not getattr(env, "pending_regions", None):
            return graph.noop_action
        observation = env.legacy_observation()
        mask = legacy_pending_mask(env)
        raw_action, _ = self.model.predict(
            observation,
            action_masks=mask,
            deterministic=self.deterministic if deterministic else False,
        )
        raw_action = np.asarray(raw_action, dtype=np.int64).reshape(NUM_REGIONS)
        for rid in sorted(env.pending_regions):
            code = int(raw_action[rid])
            if 1 <= code <= NUM_UAVS:
                uid = code - 1
                edge_action = uid * NUM_REGIONS + rid
                if bool(graph.action_mask[edge_action]):
                    return int(edge_action)
        legal_edges = np.flatnonzero(graph.action_mask[:-1].cpu().numpy())
        return int(legal_edges[0]) if legal_edges.size else graph.noop_action
