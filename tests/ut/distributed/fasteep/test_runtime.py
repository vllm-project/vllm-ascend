# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from contextlib import nullcontext
from dataclasses import make_dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.distributed.fasteep import runtime as runtime_module
from vllm_ascend.distributed.fasteep.runtime import FastEEPRuntime, ascend_parallel_state, parallel_state


def test_v2_activation_updates_configuration_without_assigning_readonly_size(monkeypatch):
    config = SimpleNamespace(use_v2_model_runner=True, parallel_config=SimpleNamespace(data_parallel_size=4))

    class Runner:
        @property
        def dp_size(self):
            return config.parallel_config.data_parallel_size

    runtime = object.__new__(FastEEPRuntime)
    runtime.worker = SimpleNamespace(vllm_config=config, model_runner=Runner())
    runtime.rank = 0
    runtime.layers = []
    monkeypatch.setattr(parallel_state, "_DP", None)
    monkeypatch.setattr(parallel_state, "_EP", None)
    runtime._activate(3, object(), [])
    assert runtime.worker.model_runner.dp_size == 3


def test_parent_groups_release_through_their_shared_handle_owner(monkeypatch):
    monkeypatch.setattr(ascend_parallel_state, "model_parallel_initialized", lambda: False)
    handle = object()
    dp = SimpleNamespace(device_group=handle, destroy_hccl=Mock())
    ep = SimpleNamespace(device_group=handle, destroy_hccl=Mock())
    runtime = object.__new__(FastEEPRuntime)
    runtime.parent_dp, runtime.parent_ep = dp, ep
    runtime._release_initial_device_groups()
    dp.destroy_hccl.assert_called_once()
    ep.destroy_hccl.assert_called_once()


def test_initial_mc2_handle_is_released_with_replaced_dp_and_ep(monkeypatch):
    mc2 = SimpleNamespace(device_group=object(), destroy_hccl=Mock())
    monkeypatch.setattr(ascend_parallel_state, "model_parallel_initialized", lambda: True)
    monkeypatch.setattr(ascend_parallel_state, "get_mc2_group", lambda: mc2)
    runtime = object.__new__(FastEEPRuntime)
    runtime.parent_dp = SimpleNamespace(device_group=None)
    runtime.parent_ep = SimpleNamespace(device_group=None)
    runtime._release_initial_device_groups()
    mc2.destroy_hccl.assert_called_once()


@pytest.mark.parametrize("communication", ["allgather", "alltoall"])
def test_activation_reuses_routing_for_equal_placements(monkeypatch, communication):
    parallel = make_dataclass("Parallel", ["dp_size", "ep_size", "ep_rank"])
    config = SimpleNamespace(
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(data_parallel_size=3),
        additional_config={"fasteep": {"communication": communication}},
    )
    placements = [
        [[0, 1, -1, -1], [-1, -1, 0, 1]],
        [[0, 1, -1, -1], [-1, -1, 0, 1]],
        [[1, 0, -1, -1], [-1, -1, 1, 0]],
    ]
    layers = [
        SimpleNamespace(
            global_expert_map=torch.tensor(placement, dtype=torch.int32),
            log2phy=torch.zeros(4, dtype=torch.int32),
            moe_config=SimpleNamespace(
                num_local_experts=2, num_logical_experts=4, moe_parallel_config=parallel(3, 3, 1)
            ),
        )
        for placement in placements
    ]
    runtime = object.__new__(FastEEPRuntime)
    runtime.rank, runtime.layers = 1, layers
    runtime.worker = SimpleNamespace(vllm_config=config)
    original_generate = runtime_module.generate_log2phy_map
    generate = Mock(wraps=original_generate)
    monkeypatch.setattr(runtime_module, "generate_log2phy_map", generate)
    monkeypatch.setattr(runtime_module, "set_current_vllm_config", lambda config: nullcontext())
    constructor = Mock(return_value=SimpleNamespace(token_dispatcher=object()))
    monkeypatch.setattr(runtime_module, "AllGatherCommImpl", constructor)
    monkeypatch.setattr(runtime_module, "AlltoAllCommImpl", constructor)
    monkeypatch.setattr(runtime_module, "_MoECommMethods", {})
    monkeypatch.setattr(parallel_state, "_DP", None)
    monkeypatch.setattr(parallel_state, "_EP", None)
    group = object()
    runtime._activate(2, group, placements)
    assert generate.call_count == 2
    for layer, placement in zip(layers, placements):
        reference = original_generate(torch.tensor(placement), 1 if communication == "alltoall" else 0, tp_size=1)
        torch.testing.assert_close(layer.log2phy, reference)
        assert layer.moe_config.ep_group is group
    assert layers[0].global_expert_map.data_ptr() != layers[1].global_expert_map.data_ptr()
    runtime._activate(2, group, placements)
    assert generate.call_count == 4
