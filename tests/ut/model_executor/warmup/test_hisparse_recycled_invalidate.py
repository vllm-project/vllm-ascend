# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts for recycled-tag warmup and its upstream fallbacks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.hisparse import recycled_invalidate as recycled
from vllm_ascend.utils import AscendDeviceType


def _tensor(device, *, address, shape, dtype):
    return SimpleNamespace(
        device=device,
        dtype=dtype,
        ndim=len(shape),
        shape=shape,
        numel=lambda: shape[0] * (shape[1] if len(shape) > 1 else 1),
        data_ptr=lambda: address,
        element_size=lambda: 4,
        is_contiguous=lambda: True,
        storage_offset=lambda: 0,
    )


@pytest.fixture
def binding(monkeypatch):
    device = SimpleNamespace(type="npu")
    stream = object()
    tables = [_tensor(device, address=0x100000 + i * 0x100000, shape=(3, 10240), dtype=torch.int32) for i in range(2)]
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(kernel_config=SimpleNamespace(enable_jit_warmup=True)),
        backend=SimpleNamespace(is_current_stream_capturing=Mock(return_value=False)),
        kernel_block_size=128,
        leader_runtimes=[SimpleNamespace(index_group=SimpleNamespace(device_global_indices=t)) for t in tables],
        _pending_invalid_block_ids=[9, 1, 9],
    )
    sentinel = _tensor(device, address=0x2000000, shape=(1,), dtype=torch.int64)
    upload = Mock(return_value=sentinel)
    launch = Mock()
    monkeypatch.setattr(recycled, "get_ascend_device_type", lambda: AscendDeviceType.A5)
    monkeypatch.setattr(recycled, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(recycled.torch.accelerator, "current_stream", lambda: stream)
    monkeypatch.setattr(recycled, "async_tensor_h2d", upload)
    monkeypatch.setattr(recycled.RecycledGroupInvalidator, "_launch", launch)
    return SimpleNamespace(
        worker=worker, tables=tables, device=device, stream=stream, sentinel=sentinel, upload=upload, launch=launch
    )


def test_warmup_uses_negative_state_without_invalidating_live_rows(binding):
    owner = recycled.RecycledGroupInvalidator.bind(binding.worker)
    assert owner is not None
    assert binding.upload.call_args_list[0].args == (
        [t.data_ptr() for t in binding.tables],
        binding.device,
        torch.int64,
    )
    assert binding.upload.call_args_list[1].args == ([-1], binding.device, torch.int64)
    binding.launch.assert_called_once_with([0] * 8, binding.sentinel)
    assert binding.worker._pending_invalid_block_ids == [9, 1, 9]


@pytest.mark.parametrize(
    "reason", ["disabled", "other-device", "capture", "zero-block", "no-groups", "wrong-dtype", "overlap"]
)
def test_ineligible_warmup_keeps_upstream_path(binding, monkeypatch, reason):
    worker = binding.worker
    if reason == "disabled":
        worker.vllm_config.kernel_config.enable_jit_warmup = False
    elif reason == "other-device":
        monkeypatch.setattr(recycled, "get_ascend_device_type", lambda: AscendDeviceType.A2)
    elif reason == "capture":
        worker.backend.is_current_stream_capturing.return_value = True
    elif reason == "zero-block":
        worker.kernel_block_size = 0
    elif reason == "no-groups":
        worker.leader_runtimes = []
    elif reason == "wrong-dtype":
        binding.tables[0].dtype = torch.int64
    else:
        binding.tables[1].data_ptr = binding.tables[0].data_ptr
    assert recycled.RecycledGroupInvalidator.bind(worker) is None
    binding.upload.assert_not_called()
    binding.launch.assert_not_called()


@pytest.mark.parametrize("count", [1, 8, 9, 17])
def test_recycled_bucket_boundaries_deduplicate_and_pad(binding, count):
    owner = recycled.RecycledGroupInvalidator.bind(binding.worker)
    binding.launch.reset_mock()
    blocks = list(reversed(range(count))) + [0]
    binding.worker._pending_invalid_block_ids = blocks
    states = _tensor(binding.device, address=0x3000000, shape=(2,), dtype=torch.int64)
    assert owner.try_call(binding.worker, blocks, states)
    chunks = [call.args[0] for call in binding.launch.call_args_list]
    assert len(chunks) == (count + 7) // 8
    assert all(len(chunk) == 8 for chunk in chunks)
    assert sorted({block for chunk in chunks for block in chunk}) == list(range(count))
    assert all(call.args[1] is states for call in binding.launch.call_args_list)
    tail = list(range((count - 1) // 8 * 8, count))
    assert chunks[-1] == tail + [count - 1] * (8 - len(tail))


@pytest.mark.parametrize(
    "reason", ["capture", "stream", "owner", "foreign-list", "empty", "negative", "overflow", "noninteger"]
)
def test_runtime_changes_fall_back_without_launch(binding, monkeypatch, reason):
    owner = recycled.RecycledGroupInvalidator.bind(binding.worker)
    binding.launch.reset_mock()
    states = _tensor(binding.device, address=0x3000000, shape=(2,), dtype=torch.int64)
    blocks = binding.worker._pending_invalid_block_ids
    if reason == "capture":
        binding.worker.backend.is_current_stream_capturing.return_value = True
    elif reason == "stream":
        monkeypatch.setattr(recycled.torch.accelerator, "current_stream", lambda: object())
    elif reason == "owner":
        binding.worker.leader_runtimes[0].index_group.device_global_indices = SimpleNamespace()
    elif reason == "foreign-list":
        blocks = list(blocks)
    elif reason == "empty":
        blocks.clear()
    elif reason == "negative":
        blocks[:] = [-1]
    elif reason == "overflow":
        blocks[:] = [2**31 // 128]
    else:
        blocks[:] = [1.5]
    assert not owner.try_call(binding.worker, blocks, states)
    binding.launch.assert_not_called()
