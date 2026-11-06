# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the production Mega page-copy launcher.

Real kernel arithmetic, NULL guards and graph replay are covered by the NPU
lifecycle test. Here the sealed launcher must submit raw metadata without
allocating, casting or constructing device pointer vectors.
"""

from contextlib import nullcontext
from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tests.ut.ops import test_kimi_mega_kda as contracts


class _Value(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(_Value)


class _Pointer:
    def __init__(self, data, offset=0):
        self.data, self.offset = data, offset

    def __add__(self, offset):
        return _Pointer(self.data, self.offset + offset)


class _TL:
    """Memory-operation interpreter for the actual production kernel body."""

    int64 = np.int64
    program = 0

    def program_id(self, axis):
        assert axis == 0
        return np.asarray(self.program).view(_Value)

    @staticmethod
    def arange(start, stop):
        return np.arange(start, stop).view(_Value)

    @staticmethod
    def cdiv(a, b):
        return (a + b - 1) // b

    @staticmethod
    def where(condition, a, b):
        return np.asarray(np.where(condition, a, b)).view(_Value)

    @staticmethod
    def load(pointer, mask=True, other=0):
        offsets = np.asarray(pointer.offset, dtype=np.int64)
        mask = np.broadcast_to(mask, offsets.shape)
        value = np.full(offsets.shape, other, dtype=pointer.data.dtype)
        value[mask] = pointer.data[offsets[mask]]
        return value.view(_Value)

    @staticmethod
    def store(pointer, value, mask=True):
        offsets = np.asarray(pointer.offset, dtype=np.int64)
        mask = np.broadcast_to(mask, offsets.shape)
        pointer.data[offsets[mask]] = np.broadcast_to(value, offsets.shape)[mask]


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("null_slot", [0, -1])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("block_size", [8, 16])
def test_production_copy_kernel_preserves_raw_pages_and_masks_bounds(index_dtype, null_slot, batch, block_size):
    tl = _TL()
    ns = {"tl": tl, "triton": SimpleNamespace(jit=lambda **_: lambda fn: fn)}
    contracts._definitions("vllm_ascend/ops/triton/mega_kda_cache.py", {"_mega_kda_cache_copy_kernel"}, ns)
    kernel = ns["_mega_kda_cache_copy_kernel"]
    rows, spec_len = 30, 8
    conv_payload, state_payload, guard = 24, 32, 64
    conv_bytes, state_bytes = conv_payload * 2, state_payload * 4
    page_bytes = conv_bytes + state_bytes + guard
    backing = np.full(guard + rows * page_bytes + guard, 37, dtype=np.uint8)
    # uint16 preserves every BF16 bit pattern without NumPy BF16 conversion.
    conv = np.ndarray((rows, conv_payload), np.uint16, backing, guard, (page_bytes, 2))
    state = np.ndarray((rows, state_payload), np.float32, backing, guard + conv_bytes, (page_bytes, 4))
    conv[:] = np.arange(rows * conv_payload).reshape(conv.shape)
    state[:] = np.arange(rows * state_payload).reshape(state.shape)
    indices = np.arange(batch * spec_len + 1, dtype=index_dtype)[1:].reshape(batch, spec_len)
    indices[0, :3] = 0, -1, rows
    conv_indices, state_indices = indices[:, 0], indices.reshape(-1)
    packed_conv = np.full((batch, conv_payload), 65535, dtype=np.uint16)
    packed_state = np.full((batch * spec_len, state_payload), np.nan, dtype=np.float32)
    before = backing.copy()
    args = (
        _Pointer(backing[guard:].view(np.uint16)),
        _Pointer(backing[guard + conv_bytes :].view(np.float32)),
        _Pointer(packed_conv.reshape(-1)),
        _Pointer(packed_state.reshape(-1)),
        _Pointer(state_indices),
        _Pointer(state_indices),
        rows,
        rows,
        page_bytes // 2,
        page_bytes // 4,
        conv_payload,
        state_payload,
        batch,
        batch * spec_len,
        spec_len,
        1,
    )
    grid = batch * tl.cdiv(conv_payload, block_size) + batch * spec_len * tl.cdiv(state_payload, block_size)
    for to_cache in (False, True):
        if to_cache:
            packed_conv.fill(1234)
            packed_state.fill(9.5)
        for tl.program in range(grid):
            kernel(*args, to_cache, null_slot, block_size)
        if not to_cache:
            np.testing.assert_array_equal(backing, before)
            for source, packed, selected in ((conv, packed_conv, conv_indices), (state, packed_state, state_indices)):
                for row, index in enumerate(selected):
                    np.testing.assert_array_equal(
                        packed[row], source[index] if 0 <= index < rows and index != null_slot else 0
                    )
        else:
            expected = before.copy()
            for packed, selected, offset in ((packed_conv, conv_indices, 0), (packed_state, state_indices, conv_bytes)):
                for row, index in enumerate(selected):
                    if 0 <= index < rows and index != null_slot:
                        start = guard + int(index) * page_bytes + offset
                        raw = packed[row].view(np.uint8)
                        expected[start : start + raw.size] = raw
            np.testing.assert_array_equal(backing, expected)


@pytest.fixture
def runtime():
    ns = contracts.runtime.__wrapped__()
    ns.update(
        MappingProxyType=MappingProxyType,
        cache_signature=ns["_mega_cache_layout"],
        _mega_kda_cache_copy_kernel=SimpleNamespace(pre_run_hooks=[]),
        _COPY_ALIGNMENT_BYTES=16,
        DEFAULT_KDA_BLOCK_SIZE=8,
    )
    contracts._definitions("vllm_ascend/ops/triton/mega_kda_cache.py", {"MegaKDAPagedCopyPlan"}, ns)
    return ns


def _plan(ns, dtype):
    # Both payloads have gaps, offsets and independent types.
    conv = torch.empty_strided((40, 3, 8), (64, 8, 1), dtype=torch.bfloat16)
    state = torch.empty_strided((40, 2, 4, 4), (80, 16, 4, 1))
    plan = ns["MegaKDAPagedCopyPlan"]()
    plan._sealed = True
    plan._layouts = tuple(ns["cache_signature"](t) for t in (conv, state))
    plan._limits = 4, 32
    plan._scalars = 40, 40, 64, 80, 24, 32
    plan._tiles = 3, 4
    kernel = MagicMock()
    plan._compiled = {(dtype, False): kernel, (dtype, True): kernel}
    return plan, (conv, state), kernel


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("to_cache", [False, True])
def test_sealed_copy_passes_strided_and_unaligned_indices_without_device_preparation(
    runtime, monkeypatch, dtype, to_cache
):
    plan, cache, kernel = _plan(runtime, dtype)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(device=lambda _: nullcontext()), raising=False)
    # Offset one also checks that no hidden 16-byte index normalization occurs.
    table = torch.arange(33, dtype=dtype)[1:].reshape(4, 8)
    conv_indices, state_indices = table[:, 0], table.reshape(-1)
    packed = torch.empty((4, 3, 8), dtype=torch.bfloat16), torch.empty((32, 2, 4, 4))
    with (
        patch.object(torch, "where", side_effect=AssertionError("per-layer Select")),
        patch.object(torch, "ones", side_effect=AssertionError("per-layer Ones")),
        patch.object(torch, "empty", side_effect=AssertionError("per-layer allocation")),
    ):
        plan.copy(cache, *packed, conv_indices, state_indices, to_cache=to_cache)
    kernel.__getitem__.assert_called_once_with((4 * 3 + 32 * 4, 1, 1))
    args = kernel.__getitem__.return_value.call_args.args
    assert len(args) == 19
    for actual, expected in zip(args[:6], (*cache, *packed, conv_indices, state_indices)):
        assert actual is expected
    assert args[6:16] == (*plan._scalars, 4, 32, 8, 1)
    assert args[16:] == (to_cache, 0, 8)


@pytest.mark.parametrize("bad", ["capacity", "layout", "dtype", "shape", "indices", "unsealed"])
def test_sealed_copy_rejects_invalid_contract_before_launch(runtime, monkeypatch, bad):
    plan, cache, kernel = _plan(runtime, torch.int32)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(device=lambda _: nullcontext()), raising=False)
    indices = torch.arange(32, dtype=torch.int32).reshape(4, 8)
    conv_indices, state_indices = indices[:, 0], indices.flatten()
    conv = torch.empty((4, 3, 8), dtype=torch.bfloat16)
    state = torch.empty((32, 2, 4, 4))
    if bad == "capacity":
        plan._limits = 3, 32
    elif bad == "layout":
        cache = cache[0][:30], cache[1]
    elif bad == "dtype":
        conv = conv.float()
    elif bad == "shape":
        state = state[:31]
    elif bad == "indices":
        conv_indices = conv_indices.long()
    else:
        plan._sealed = False
    with pytest.raises((ValueError, RuntimeError)):
        plan.copy(cache, conv, state, conv_indices, state_indices, to_cache=False)
    kernel.__getitem__.assert_not_called()
