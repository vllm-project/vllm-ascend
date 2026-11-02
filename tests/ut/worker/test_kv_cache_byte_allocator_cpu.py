# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real Tensor storage tests for native and external KV byte allocators."""

import runpy
from pathlib import Path

import pytest
import torch


@pytest.fixture
def allocate_bytes():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/kv_cache_allocator.py"
    return runpy.run_path(str(source))["allocate_zeroed_byte_tensor"]


@pytest.mark.parametrize("alignment", [1, 64, 2 * 1024 * 1024])
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
def test_external_allocator_retains_storage_alignment_and_zeroing(allocate_bytes, alignment, dtype):
    allocations = []

    def external(size):
        raw = torch.full((size,), 17, dtype=dtype)
        allocations.append(raw)
        return raw

    result = allocate_bytes(256, torch.device("cpu"), alignment=alignment, allocate=external)
    assert len(allocations) == 1
    backing = allocations[0]
    assert backing.numel() == (256 + alignment if alignment > 1 else 256)
    assert result.dtype == torch.int8 and result.numel() == 256
    assert result.data_ptr() % alignment == 0
    assert result.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    assert torch.count_nonzero(backing) == 0
    offset = result.data_ptr() - backing.data_ptr()
    result[0] = 77
    assert backing.view(torch.uint8)[offset] == 77
    allocations.clear()
    del backing
    assert result[0] == 77


@pytest.mark.parametrize("alignment", [1, 64, 2 * 1024 * 1024])
def test_default_allocation_retains_zeroed_bytes(allocate_bytes, alignment):
    value = allocate_bytes(256, torch.device("cpu"), alignment=alignment)
    assert value.shape == (256,)
    assert value.dtype == torch.int8
    assert value.data_ptr() % alignment == 0
    assert torch.count_nonzero(value) == 0


@pytest.mark.parametrize("kind", ["dtype", "size", "stride", "device"])
def test_invalid_callback_output_is_rejected(allocate_bytes, kind):
    def external(size):
        if kind == "dtype":
            return torch.empty(size, dtype=torch.float32)
        if kind == "size":
            return torch.empty(size - 1, dtype=torch.int8)
        if kind == "stride":
            return torch.empty(size * 2, dtype=torch.int8)[::2]
        return torch.empty(size, dtype=torch.int8, device="meta")

    with pytest.raises(ValueError):
        allocate_bytes(128, torch.device("cpu"), allocate=external)


@pytest.mark.parametrize("size,alignment", [(-1, 1), (1, 0), (1, -1)])
def test_invalid_geometry_is_rejected(allocate_bytes, size, alignment):
    with pytest.raises(ValueError):
        allocate_bytes(size, torch.device("cpu"), alignment=alignment)


def test_non_tensor_callback_is_rejected(allocate_bytes):
    with pytest.raises(TypeError):
        allocate_bytes(128, torch.device("cpu"), allocate=lambda size: None)
