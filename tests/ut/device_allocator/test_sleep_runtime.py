# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.device_allocator.sleep_runtime import AscendSleepRuntime


@pytest.fixture
def runtime():
    worker = MagicMock()
    worker._sleep_saved_buffers = {}
    worker._sleep_saved_parameters = {}
    worker._pp_send_work = []
    instance = AscendSleepRuntime(worker)
    # Keep the worker alive; production owns it independently of this adapter.
    yield instance, worker


def test_worker_ownership_is_weak():
    class Worker:
        pass

    worker = Worker()
    runtime = AscendSleepRuntime(worker)
    del worker
    with pytest.raises(RuntimeError, match="released"):
        _ = runtime.worker


@pytest.mark.parametrize("level, expected", [(1, ("weights",)), (2, ())])
def test_allocator_offload_policy(runtime, level, expected):
    instance, _ = runtime
    with patch("vllm_ascend.device_allocator.sleep_runtime.CaMemAllocator") as allocator:
        instance.sleep_allocator(level)
        allocator.get_instance().sleep.assert_called_once_with(offload_tags=expected)


def test_backup_and_restore_buffers(runtime):
    instance, worker = runtime
    model = torch.nn.Module()
    model.register_buffer("running_state", torch.tensor([3.0, 4.0]))
    worker.model_runner.model = model
    instance.backup(2)
    model.running_state.fill_(0)
    instance.restore()
    assert model.running_state.tolist() == [3.0, 4.0]
    assert worker._sleep_saved_buffers == {}
    worker._save_sleep_parameters.assert_called_once_with(model)


def test_level_one_does_not_copy_model(runtime):
    instance, worker = runtime
    instance.backup(1)
    worker.model_runner.model.named_buffers.assert_not_called()
    worker._save_sleep_parameters.assert_not_called()


def test_synchronize_drains_pending_sends(runtime):
    instance, worker = runtime
    send = MagicMock()
    worker._pp_send_work = [send]
    instance.synchronize()
    send.wait.assert_called_once()
    worker.synchronize_device.assert_called_once()
    assert worker._pp_send_work == []


@pytest.mark.parametrize("enabled", [True, False])
def test_cleanup_and_rebuild_use_same_existing_manager(runtime, enabled):
    instance, worker = runtime
    with patch("vllm_ascend.device_allocator.sleep_runtime.get_ascend_config") as config:
        config.return_value.rl_config = SimpleNamespace(enabled=enabled, sleep_mode_extra_cleanup=True)
        instance.cleanup()
        instance.rebuild(["weights"])
    assert worker.sleep_wakeup_manager.sleep.call_count == int(enabled)
    assert worker.sleep_wakeup_manager.wakeup.call_count == int(enabled)
    if enabled:
        worker.sleep_wakeup_manager.wakeup.assert_called_once_with(["weights"])


def test_rank_failure_uses_cpu_group(runtime):
    instance, _ = runtime
    group = object()
    with (
        patch(
            "vllm_ascend.device_allocator.sleep_runtime.get_world_group", return_value=SimpleNamespace(cpu_group=group)
        ),
        patch.object(torch.distributed, "is_available", return_value=True),
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(torch.distributed, "all_gather_object") as gather,
    ):

        def results(output, local, *, group):
            output[:] = [local, ("remap", False)]

        gather.side_effect = results
        with pytest.raises(RuntimeError, match="one or more ranks"):
            instance.agree("remap", None)
        assert gather.call_args.kwargs["group"] is group


def test_single_rank_local_error_is_not_swallowed(runtime):
    instance, _ = runtime
    with (
        patch.object(torch.distributed, "is_initialized", return_value=False),
        pytest.raises(ValueError, match="injected"),
    ):
        instance.agree("restore", ValueError("injected"))
