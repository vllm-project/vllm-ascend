# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.worker.worker import NPUWorker


class _AllocationScope(AbstractContextManager):
    def __init__(self) -> None:
        self.active = False

    def __enter__(self):
        assert not self.active
        self.active = True
        return self

    def __exit__(self, *args: Any) -> None:
        assert self.active
        self.active = False


def test_kv_wake_does_not_run_model_runner_recovery() -> None:
    model = torch.nn.Module()
    model.register_buffer("_k_scale", torch.tensor(0.5))
    model.register_buffer("_v_scale", torch.tensor(0.25))

    class Runner:
        def __init__(self) -> None:
            self.model = model
            self.recovery_calls = 0

        def post_kv_cache_wake_up(self) -> None:
            self.recovery_calls += 1
            self.model.get_buffer("_k_scale").fill_(1.0)
            self.model.get_buffer("_v_scale").fill_(1.0)

    runner = Runner()
    with patch.object(NPUWorker, "__init__", lambda *_args, **_kwargs: None):
        worker = NPUWorker()
    worker.model_runner = runner
    worker._extra_cleanup_pending = False
    worker.sleep_wakeup_manager = MagicMock()
    worker._sleep_mode_backend = MagicMock()

    with (
        # wake_up ends with a device barrier; this UT runs on a CPU runner, so the
        # barrier must be mocked or torch.npu.synchronize() would try to init NPU.
        patch("vllm_ascend.worker.worker.NPUWorker.synchronize_device") as mock_synchronize,
        patch("vllm_ascend.worker.worker.CaMemAllocator") as mock_allocator_class,
        patch("vllm_ascend.worker.worker.get_ascend_config") as mock_get_config,
    ):
        mock_get_config.return_value = SimpleNamespace(
            weight_nz_mode=0,
            rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False),
        )
        mock_allocator_class.get_instance.return_value = MagicMock()
        NPUWorker.wake_up(worker, tags=["kv_cache"])

    assert runner.recovery_calls == 0
    assert model.get_buffer("_k_scale").item() == 0.5
    assert model.get_buffer("_v_scale").item() == 0.25
    mock_synchronize.assert_called_once_with()


def _make_sleep_worker(model: torch.nn.Module) -> NPUWorker:
    with patch.object(NPUWorker, "__init__", lambda *_args, **_kwargs: None):
        worker = NPUWorker()
    worker.model_config = SimpleNamespace(
        sleep_preserve_parameter_names=[],
        sleep_mode_backend="camem",
    )
    from vllm_ascend.device_allocator.sleep_mode_backend import CaMemBackend

    worker.model_runner = SimpleNamespace(
        model=model,
        get_draft_model=lambda: None,
    )
    worker._extra_cleanup_pending = False
    worker._sleep_mode_backend = MagicMock(spec=CaMemBackend)
    worker.sleep_wakeup_manager = MagicMock()
    return worker


@patch("vllm_ascend.worker.worker.NPUWorker.synchronize_device")
@patch("vllm_ascend.worker.worker.torch.npu.mem_get_info", side_effect=[(100, 200), (150, 200)])
@patch("vllm_ascend.worker.worker.CaMemAllocator")
@patch("vllm_ascend.worker.worker.get_ascend_config")
def test_level1_sleep_does_not_cpu_backup_hadamard(
    mock_get_config, mock_allocator_class, _mock_mem_get_info, mock_synchronize_device
) -> None:
    mock_get_config.return_value = SimpleNamespace(
        rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False)
    )
    mock_allocator_class.get_instance.return_value = MagicMock()

    hadamard = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    model = torch.nn.Module()
    model.register_buffer("_dsa_hadamard", hadamard.clone())
    worker = _make_sleep_worker(model)

    NPUWorker.sleep(worker, level=1)

    assert torch.equal(model.get_buffer("_dsa_hadamard"), hadamard)
    # Both sleep barriers fire on the cleanup-disabled path too.
    assert mock_synchronize_device.call_count == 2


@patch("vllm_ascend.worker.worker.NPUWorker.synchronize_device")
@patch("vllm_ascend.worker.worker.torch.npu.mem_get_info", side_effect=[(100, 200), (150, 200)])
@patch("vllm_ascend.worker.worker.CaMemAllocator")
@patch("vllm_ascend.worker.worker.get_ascend_config")
def test_level2_sleep_does_not_python_backup_named_buffers(
    mock_get_config, mock_allocator_class, _mock_mem_get_info, mock_synchronize_device
) -> None:
    mock_get_config.return_value = SimpleNamespace(
        rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False)
    )
    mock_allocator_class.get_instance.return_value = MagicMock()

    hadamard = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    model = torch.nn.Module()
    model.register_buffer("_dsa_hadamard", hadamard.clone())
    worker = _make_sleep_worker(model)

    NPUWorker.sleep(worker, level=2)

    assert not hasattr(worker, "_sleep_saved_buffers")
    assert torch.equal(model.get_buffer("_dsa_hadamard"), hadamard)
    worker.sleep_mode_backend.suspend.assert_called_once_with(set())
    assert mock_synchronize_device.call_count == 2
