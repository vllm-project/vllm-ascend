# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace
from unittest.mock import Mock

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
