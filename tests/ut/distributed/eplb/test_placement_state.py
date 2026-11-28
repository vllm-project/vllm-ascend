# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed.eplb import eplb_state as module
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbLayerState, AscendEplbState


class TestPlacementState(unittest.TestCase):
    def test_publish_marks_every_registered_model_rebalanced(self):
        state = AscendEplbState.__new__(AscendEplbState)
        model = SimpleNamespace(num_physical_experts=4, num_expert_groups=1)
        target = SimpleNamespace(rebalanced=False, model=model)
        draft = SimpleNamespace(rebalanced=False, model=model)
        state.model_states = {"target": target, "draft": draft}
        state.rearrange_event = MagicMock()
        state.get_rank_node_ids = MagicMock(return_value=[0])
        stats = SimpleNamespace(values=torch.zeros(1))

        device_group = SimpleNamespace(size=lambda: 1)
        with (
            patch.object(module, "get_eplb_group", return_value=SimpleNamespace(device_group=device_group)),
            patch.object(module._eplb_state, "EplbStats", MagicMock()),
        ):
            state.publish_async_load_stats({"target": stats, "draft": stats})

        # The draft shares the target lifecycle, so one publication marks every
        # registered model rebalanced and the async worker rebalances both.
        self.assertTrue(target.rebalanced)
        self.assertTrue(draft.rebalanced)
        state.rearrange_event.record.assert_called_once_with()

    def test_static_layer_skips_allgather_and_token_recording(self):
        layer = AscendEplbLayerState()
        layer.load_recording_enabled = False
        gather = MagicMock()
        with patch.object(module.torch.ops.vllm, "ascend_eplb_record_expert_tokens", create=True) as record:
            layer.record_load(
                None,
                0,
                topk_ids=torch.empty((0, 1), dtype=torch.long),
                num_source_tokens=0,
                all_gather_mask=gather,
            )
        gather.assert_not_called()
        record.assert_not_called()

    def test_static_state_toggles_layer_recording_only_when_needed(self):
        state = AscendEplbState.__new__(AscendEplbState)
        state._static_recording_enabled = False
        layers = [
            SimpleNamespace(load_recording_enabled=False),
            SimpleNamespace(load_recording_enabled=False),
        ]
        model = SimpleNamespace(moe_layers=[SimpleNamespace(eplb_state=layer) for layer in layers])
        state.model_states = {"target": SimpleNamespace(model=model)}

        state.set_static_recording_enabled(True)
        self.assertTrue(all(layer.load_recording_enabled for layer in layers))

        state.set_static_recording_enabled(False)
        self.assertFalse(any(layer.load_recording_enabled for layer in layers))

    def test_static_state_skips_profile_rearrangement_and_background_start(self):
        state = AscendEplbState.__new__(AscendEplbState)
        state.static_mode = True
        state._suspended = False
        state.model_states = {}
        state._static_recording_enabled = False
        state.should_record_tensor = None
        state.prepare_movement_resources = MagicMock()
        with (
            patch.object(module._eplb_state.EplbState, "step") as step,
            patch.object(module._eplb_state.EplbState, "rearrange") as rearrange,
            patch.object(module._eplb_state.EplbState, "start_async_loop") as start,
        ):
            state.step(is_profile=True)
            state.rearrange(is_profile=True)
            state.start_async_loop()
        step.assert_not_called()
        rearrange.assert_not_called()
        start.assert_not_called()
        state.prepare_movement_resources.assert_not_called()

    def test_static_load_stats_are_rank_local_without_all_reduce(self):
        state = AscendEplbState.__new__(AscendEplbState)
        state._static_observation_step = 7
        state.model_states = {
            "target": SimpleNamespace(
                model_name="glm",
                expert_load_pass=torch.tensor(
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                    ]
                ),
            )
        }
        group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
        with (
            patch.object(module, "get_ep_group", return_value=SimpleNamespace(device_group=group)),
            patch.object(module._eplb_state.EplbState, "_sync_load_pass") as sync,
            patch.object(module, "gpu_sync_allowed", return_value=nullcontext()),
            self.assertLogs("vllm", level="INFO") as logs,
        ):
            state._log_static_load_stats()
        sync.assert_not_called()
        self.assertIn("scope=local", "\n".join(logs.output))

    def test_routing_refresh_preserves_tensor_identity(self):
        layer = AscendEplbLayerState()
        layer.logical_to_physical_map = torch.tensor([[0, 2], [1, -1]])
        layer.logical_replica_count = torch.tensor([2, 1])
        with (
            patch.object(module, "get_ep_group", return_value=SimpleNamespace(rank_in_group=0)),
            patch.object(
                module, "get_current_hardware_profile", return_value=SimpleNamespace(moe_comm_policy=object())
            ),
        ):
            layer.refresh_expert_replica_routing_table()
            routing = layer.expert_replica_routing_table
            layer.logical_to_physical_map.copy_(torch.tensor([[1, 2], [0, -1]]))
            layer.refresh_expert_replica_routing_table()
        self.assertIs(layer.expert_replica_routing_table, routing)
        self.assertEqual(set(routing[:, 0].tolist()), {1, 2})
        self.assertEqual(set(routing[:, 1].tolist()), {0})
