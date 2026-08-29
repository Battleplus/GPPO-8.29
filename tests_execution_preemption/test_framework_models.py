from __future__ import annotations

import math
import unittest

import torch

from execution_preemption.adapter import build_flat_observation, build_hetero_observation
from execution_preemption.controller import PreemptionController
from execution_preemption.framework import flat_to_torch, hetero_to_torch
from execution_preemption.graph import FEATURE_DIMENSIONS, NODE_TYPES, RELATIONS, build_execution_graph
from execution_preemption.models import (
    EventPriority,
    RuntimeEvent,
    RuntimeEventType,
    TaskRuntime,
    UAVRuntime,
)
from execution_preemption.policy_models import ExecutionGPPOAdaptive, ExecutionPPOMLP
from execution_preemption.runtime import ExecutionRuntime


class FrameworkTensorAndModelTests(unittest.TestCase):
    @staticmethod
    def observations():
        runtime = ExecutionRuntime()
        runtime.add_uav(UAVRuntime("FAILED", energy_ratio=0.9))
        runtime.add_uav(UAVRuntime("REPLACEMENT", energy_ratio=0.8))
        runtime.add_task(TaskRuntime(
            "RUN", "SEARCH", 80, 20.0, progress=0.4, remaining_work=0.6
        ))
        runtime.assign_task("RUN", "FAILED", at=0.0)
        event = RuntimeEvent(
            "E0", RuntimeEventType.EXECUTION_FAILURE, EventPriority.P0,
            occurred_at=1.0, received_at=1.0, task_id="RUN", uav_id="FAILED",
        )
        pending = runtime.begin_event_batch_transaction(
            [event], PreemptionController(), now=1.0
        )
        staged = pending.staged_runtime_copy()
        graph = build_execution_graph(
            staged,
            now=1.0,
            events=[event],
            allocation_request=pending.allocation_request,
        )
        flat = build_flat_observation(
            graph, request=pending.allocation_request, decision=pending.decision
        )
        hetero = build_hetero_observation(
            graph, request=pending.allocation_request, decision=pending.decision
        )
        return flat, hetero

    def test_tensor_conversion_preserves_graph_and_shared_mask(self) -> None:
        flat, hetero = self.observations()
        flat_tensor = flat_to_torch(flat)
        hetero_tensor = hetero_to_torch(hetero)
        self.assertEqual(flat_tensor.vector.dtype, torch.float32)
        self.assertEqual(flat_tensor.action_mask.dtype, torch.bool)
        self.assertEqual(tuple(flat_tensor.vector.shape), (37976,))
        self.assertEqual(tuple(flat_tensor.action_mask.shape), (3073,))
        self.assertTrue(torch.equal(flat_tensor.action_mask, hetero_tensor.action_mask))
        self.assertEqual(flat_tensor.graph_sha256, hetero_tensor.graph_sha256)
        self.assertEqual(flat_tensor.graph_version, hetero_tensor.graph_version)
        for node_type in NODE_TYPES:
            self.assertEqual(
                hetero_tensor.nodes[node_type].shape[1], FEATURE_DIMENSIONS[node_type]
            )
        for relation in RELATIONS:
            self.assertEqual(hetero_tensor.edge_index[relation].shape[0], 2)

    def test_ppo_forward_backward_and_action_are_finite_and_legal(self) -> None:
        flat, _ = self.observations()
        observation = flat_to_torch(flat)
        torch.manual_seed(1234)
        model = ExecutionPPOMLP(hidden_dim=16)
        logits, value, diagnostics = model(observation)
        self.assertEqual(tuple(logits.shape), (3073,))
        self.assertTrue(torch.isfinite(logits[observation.action_mask]).all())
        self.assertTrue(torch.isfinite(value))
        loss = -logits[observation.action_mask].mean() + value.square()
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))
        action, log_prob, scalar_value, clean = model.act(observation)
        self.assertTrue(bool(observation.action_mask[action]))
        self.assertTrue(math.isfinite(log_prob))
        self.assertTrue(math.isfinite(scalar_value))
        self.assertTrue(math.isfinite(clean["pre_mask_invalid_probability"]))
        self.assertTrue(torch.isfinite(diagnostics["pre_mask_invalid_probability"]))

    def test_gppo_forward_backward_and_action_are_finite_and_legal(self) -> None:
        _, hetero = self.observations()
        observation = hetero_to_torch(hetero)
        torch.manual_seed(1234)
        model = ExecutionGPPOAdaptive(hidden_dim=16, layers=1)
        logits, value, diagnostics = model(observation)
        self.assertEqual(tuple(logits.shape), (3073,))
        self.assertTrue(torch.isfinite(logits[observation.action_mask]).all())
        self.assertTrue(torch.isfinite(value))
        loss = -logits[observation.action_mask].mean() + value.square()
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))
        action, log_prob, scalar_value, clean = model.act(observation)
        self.assertTrue(bool(observation.action_mask[action]))
        self.assertTrue(math.isfinite(log_prob))
        self.assertTrue(math.isfinite(scalar_value))
        self.assertTrue(math.isfinite(clean["pre_mask_invalid_probability"]))
        self.assertTrue(diagnostics["gates"])

    def test_forward_is_deterministic_in_eval_mode(self) -> None:
        flat, hetero = self.observations()
        flat_tensor = flat_to_torch(flat)
        hetero_tensor = hetero_to_torch(hetero)
        torch.manual_seed(77)
        ppo = ExecutionPPOMLP(hidden_dim=8).eval()
        first = ppo(flat_tensor)[0].detach().clone()
        second = ppo(flat_tensor)[0].detach().clone()
        self.assertTrue(torch.equal(first, second))
        torch.manual_seed(88)
        gppo = ExecutionGPPOAdaptive(hidden_dim=8, layers=1).eval()
        first = gppo(hetero_tensor)[0].detach().clone()
        second = gppo(hetero_tensor)[0].detach().clone()
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
