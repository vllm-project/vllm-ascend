# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.worker.v2 import model_runner
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


@pytest.mark.parametrize("synchronized", [None, False, True])
def test_active_mapping_does_not_reload_global_adapters(synchronized):
    runner = object.__new__(NPUModelRunner)
    coordinator = None if synchronized is None else SimpleNamespace(step_active=synchronized, refresh_slot_ids=Mock())
    runner._moe_lora_coordinator = coordinator
    requests = {"adapter-a"}
    with patch.object(GPUModelRunner, "_set_active_loras") as activate:
        runner._set_active_loras((11,), (11, 0), requests)
    assert activate.call_args.args[:2] == ((11,), (11, 0))
    assert activate.call_args.args[2] == (set() if synchronized else requests)
    if coordinator is not None:
        coordinator.refresh_slot_ids.assert_called_once()


@pytest.mark.parametrize("dummy_run", [False, True])
def test_real_and_idle_dp_forward_synchronize_once(dummy_run):
    runner = object.__new__(NPUModelRunner)
    runner._moe_lora_coordinator = SimpleNamespace(synchronize=Mock())
    runner.lora_state = SimpleNamespace(get_activate_loras=Mock(return_value={"adapter-a"}))
    runner.pcp_manager = None
    runner._target_uses_flashmla = False
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"request-a": 1})
    with (
        patch.object(GPUModelRunner, "gather_batch_req_state", return_value=(None, None)),
        patch.object(model_runner.expert_offload, "prepare_batch"),
    ):
        runner.gather_batch_req_state(scheduler_output, dummy_run)
    runner._moe_lora_coordinator.synchronize.assert_called_once_with(set() if dummy_run else {"adapter-a"})


def test_model_loading_delegates_to_upstream_before_coordinator_binding():
    original_model = torch.nn.Linear(2, 2)
    wrapped_model = torch.nn.Sequential(original_model)
    runner = object.__new__(NPUModelRunner)
    runner.lora_manager = object()
    config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=2))
    device = torch.device("cpu")
    with (
        patch.object(GPUModelRunner, "load_lora_model", return_value=wrapped_model) as load,
        patch.object(model_runner.MoELoRACoordinator, "from_model") as bind,
    ):
        assert runner.load_lora_model(original_model, config, device) is wrapped_model
    load.assert_called_once_with(original_model, config, device)
    bind.assert_called_once_with(wrapped_model, runner.lora_manager, dp_size=2)
    assert runner._moe_lora_coordinator is bind.return_value
