# SPDX-License-Identifier: Apache-2.0
"""MRV2 wrapper selection and fallback regression cases."""

import importlib
import os

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu import buffer_utils

pytest.importorskip("torch_npu")
patch_uva = importlib.import_module("vllm_ascend.patch.worker.patch_v2.patch_uva")

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")


def test_fallback_copies_modified_prefix_and_sparse_rows(monkeypatch):
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    buffer = patch_uva.UvaBufferWrapper((4, 2), torch.int32)

    buffer.cpu[:2] = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    torch.testing.assert_close(buffer.uva().cpu(), buffer.cpu._tensor)

    buffer.cpu[3] = torch.tensor([5, 6], dtype=torch.int32)
    torch.testing.assert_close(buffer.uva().cpu(), buffer.cpu._tensor)


def test_unmapped_storage_uses_fallback(monkeypatch):
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: True)
    monkeypatch.setattr(patch_uva, "can_get_npu_view_from_cpu_tensor", lambda _: False)
    buffer = patch_uva.UvaBufferWrapper((2, 2), torch.int32)

    assert not buffer._use_real_uva
    buffer.np[0] = [7, 8]
    torch.testing.assert_close(buffer.uva().cpu(), buffer.cpu._tensor)


@pytest.mark.parametrize("max_concurrency", [2, 3])
@pytest.mark.parametrize("input_type", ["list", "numpy", "tensor"])
def test_pool_fallback_growth_shrink_and_round_robin(monkeypatch, max_concurrency, input_type):
    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: True)
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    pool = buffer_utils.UvaBufferPool((2, 2), torch.int32, max_concurrency=max_concurrency)

    for step, length in enumerate((2, 5, 1, 6, 3)):
        expected = torch.arange(length * 2, dtype=torch.int32).reshape(length, 2) + step
        values = expected.tolist() if input_type == "list" else expected.numpy() if input_type == "numpy" else expected
        before = list(pool._uva_bufs)
        slot = (pool._curr + 1) % max_concurrency

        result = pool.copy_to_uva(values)
        assert pool._curr == slot
        assert result.shape == expected.shape
        torch.testing.assert_close(result.cpu(), expected)
        for other_slot in range(max_concurrency):
            if other_slot != slot:
                assert pool._uva_bufs[other_slot] is before[other_slot]
        if length <= before[slot].cpu.shape[0]:
            assert pool._uva_bufs[slot] is before[slot]
        else:
            assert pool._uva_bufs[slot].cpu.shape[0] == 1 << (length - 1).bit_length()


@pytest.mark.skipif(
    "pinned_mem_register:True" not in os.getenv("PYTORCH_NPU_ALLOC_CONF", ""),
    reason="requires mapped pinned CPU memory",
)
def test_real_path_uses_npu_typed_view(monkeypatch):
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: True)
    buffer = patch_uva.UvaBufferWrapper((4, 2), torch.int32)

    assert buffer._use_real_uva
    assert buffer.cpu.device.type == "cpu"
    assert buffer.uva().device.type == "npu"
    assert buffer.uva().shape == buffer.cpu.shape


@pytest.mark.skipif(
    "pinned_mem_register:True" not in os.getenv("PYTORCH_NPU_ALLOC_CONF", ""),
    reason="requires mapped pinned CPU memory",
)
def test_pool_real_path_returns_mapped_view(monkeypatch):
    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: True)
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: True)
    pool = buffer_utils.UvaBufferPool((2, 2), torch.int32, max_concurrency=2)

    result = pool.copy_to_uva(np.array([[1, 2], [3, 4]], dtype=np.int32))
    assert result.device.type == "npu"
    assert result.data_ptr() == pool._uva_bufs[pool._curr].cpu.data_ptr()
