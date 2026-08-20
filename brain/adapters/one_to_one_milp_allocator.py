"""One-to-one policy wrapper around the existing MILP task allocator."""

from __future__ import annotations

from ..domain.result import AlgorithmResult
from ..domain.task import one_to_one_strike_tasks
from ..integration import sync_context_from_air_combat_scene
from .milp_task_allocator import MILPTaskAllocator as _BaseMILPTaskAllocator


class MILPTaskAllocator(_BaseMILPTaskAllocator):
    """Synchronise optional Isaac state and enforce one-to-one strikes."""

    @staticmethod
    def _sync_isaac_if_present(context) -> None:
        scene = getattr(context, "world_state", {}).get("isaac_scene")
        if scene is not None:
            sync_context_from_air_combat_scene(context, scene)

    def allocate_recon(self, context) -> AlgorithmResult:
        self._sync_isaac_if_present(context)
        return super().allocate_recon(context)

    def allocate_action(
        self,
        context,
        target_ids=None,
        *,
        include_engaged: bool = False,
    ) -> AlgorithmResult:
        self._sync_isaac_if_present(context)
        result = super().allocate_action(
            context,
            target_ids=target_ids,
            include_engaged=include_engaged,
        )
        if not result.success:
            return result
        assignments = one_to_one_strike_tasks(result.data or [])
        if not assignments:
            return AlgorithmResult.fail(
                "MILP produced no feasible one-to-one strike assignments"
            )
        return AlgorithmResult.ok(assignments)
