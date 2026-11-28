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

        def restore_kv_cache_static_state(self) -> None:
            pass

    runner = Runner()
    with patch.object(NPUWorker, "__init__", lambda *_args, **_kwargs: None):
        worker = NPUWorker()
    worker.model_runner = runner
    worker._sleep_mode_backend = MagicMock()
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


def test_kv_wake_restores_c8_mxfp_v_scales_before_reregister() -> None:
    from vllm_ascend.attention.attention_c8_mxfp import (
        AscendC8MXFPAttentionBackendImpl,
        mxfp_paged_cache_views,
    )
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    heads, block, dim, num_blocks = 2, 512, 64, 2
    pitch = heads * block * (dim + dim + (dim + dim) // 32) + 256
    raw = torch.zeros((num_blocks * pitch,), dtype=torch.int8)
    views = mxfp_paged_cache_views(raw, num_blocks, heads, dim, dim, block, page_stride_bytes=pitch)
    layer = SimpleNamespace(
        v_cache_scale=torch.arange(heads * dim, dtype=torch.int32).remainder(251).to(torch.uint8),
        impl=object.__new__(AscendC8MXFPAttentionBackendImpl),
        kv_cache=views,
    )
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.kv_cache_config = SimpleNamespace(num_blocks=num_blocks)
    runner.compilation_config = SimpleNamespace(static_forward_context={"attn": layer})
    runner._fill_c8_mxfp_v_scale_caches()
    expected_v_scale = views[3].clone()

    events: list[str] = []
    real_restore = runner.restore_kv_cache_static_state

    def tracked_restore() -> None:
        events.append("restore")
        real_restore()

    runner.restore_kv_cache_static_state = tracked_restore

    with patch.object(NPUWorker, "__init__", lambda *_args, **_kwargs: None):
        worker = NPUWorker()
    worker.model_runner = runner
    worker.sleep_wakeup_manager = MagicMock()
    worker._sleep_mode_backend = MagicMock()
    worker._sleep_mode_backend.resume.side_effect = lambda _tags: events.append("resume")

    raw.fill_(73)
    assert not torch.equal(views[3], expected_v_scale)

    with (
        patch("vllm_ascend.worker.worker.NPUWorker.synchronize_device", side_effect=lambda: events.append("sync")),
        patch("vllm_ascend.worker.worker.global_te") as mock_global_te,
        patch("vllm_ascend.worker.worker.get_ascend_config") as mock_get_config,
    ):
        mock_global_te.reregister_buffer.side_effect = lambda: events.append("reregister")
        mock_get_config.return_value = SimpleNamespace(
            weight_nz_mode=0,
            rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False),
        )
        NPUWorker.wake_up(worker, tags=["weights"])
        assert events == ["resume", "sync"]
        assert not torch.equal(views[3], expected_v_scale)

        events.clear()
        NPUWorker.wake_up(worker, tags=["kv_cache"])

    assert torch.equal(views[3], expected_v_scale)
    assert events == ["resume", "restore", "reregister", "sync"]
