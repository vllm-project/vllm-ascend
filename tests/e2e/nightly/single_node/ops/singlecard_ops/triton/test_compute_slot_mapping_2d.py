# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.compute_slot_mapping_2d import (
    _compute_slot_mapping_2d_kernel,
    compute_slot_mapping_2d,
)

BLOCK = 1024  # wrapper tile size; the 1025-token cases exercise the masked tail.


def _eager_reference(active_slots, storage_block_size, ratio):
    """Eager PyTorch chain that compute_slot_mapping_2d fuses (see dsa_v41)."""
    if ratio > 1:
        valid_comp = (active_slots >= 0) & ((active_slots + 1) % ratio == 0)
        slots = torch.where(valid_comp, active_slots // ratio, -1)
    else:
        slots = active_slots
    valid = slots >= 0
    physical = slots.clamp_min(0)
    num_tokens = active_slots.shape[0]
    slot2d = torch.full((num_tokens, 2), -1, dtype=torch.int32, device=active_slots.device)
    slot2d[:, 0] = torch.where(
        valid,
        torch.div(physical, storage_block_size, rounding_mode="floor"),
        -1,
    ).to(torch.int32)
    slot2d[:, 1] = torch.where(valid, physical.remainder(storage_block_size), -1).to(torch.int32)
    flat = torch.where(valid, slots, -1).to(torch.int64)
    return slot2d, flat


def _make_outputs(num_tokens):
    slot2d = torch.full((num_tokens, 2), -1, dtype=torch.int32, device="npu")
    flat = torch.full((num_tokens,), -1, dtype=torch.int64, device="npu")
    return slot2d, flat


def _make_slots(num_tokens, storage_block_size, ratio):
    """Mix valid slots, padded (-1) slots, storage-block / group boundaries and
    large physical slots (>= 2**24)."""
    base = torch.arange(num_tokens, dtype=torch.int64, device="npu")
    period = max(storage_block_size, ratio)
    slots = (base * 7 + 3) % (period * 4)
    slots[::5] = -1  # padded / negative slots
    slots[1::7] = storage_block_size - 1  # last slot of a storage block
    slots[2::11] = storage_block_size  # first slot of the next block
    if ratio > 1:
        slots[3::13] = ratio - 1  # incomplete compression group -> -1
        slots[4::17] = 2 * ratio - 1  # complete compression group
    # Physical slots beyond 2**24 are not exactly representable in fp32; the
    # kernel must still map them exactly (CANN 9.2.0 lowers int32 division
    # through fp32 and rounds such dividends).
    slots[6::29] = 2**24 + 2 + (base[6::29] % (period * 2))
    return slots


def _assert_matches(actual_slot2d, actual_flat, slots, storage_block_size, ratio, num_tokens):
    slot2d_ref, flat_ref = _eager_reference(slots, storage_block_size, ratio)
    torch.testing.assert_close(actual_slot2d[:num_tokens], slot2d_ref, rtol=0, atol=0)
    torch.testing.assert_close(actual_flat[:num_tokens], flat_ref, rtol=0, atol=0)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="Requires Ascend A2/A3",
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("num_tokens", [0, 1, 1023, 1024, 1025])
@pytest.mark.parametrize("storage_block_size", [128, 256])
def test_compute_slot_mapping_2d_matches_eager(dtype, ratio, num_tokens, storage_block_size):
    slots = _make_slots(num_tokens, storage_block_size, ratio).to(dtype)
    slot2d, flat = _make_outputs(num_tokens)

    compute_slot_mapping_2d(
        active_slots=slots,
        storage_block_size=storage_block_size,
        ratio=ratio,
        slot2d=slot2d,
        flat=flat,
    )
    _assert_matches(slot2d, flat, slots, storage_block_size, ratio, num_tokens)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="Requires Ascend A2/A3",
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("ratio", [1, 2])
def test_compute_slot_mapping_2d_large_slots(dtype, ratio):
    """Slots beyond 2**24 must map exactly: CANN 9.2.0 lowers int32 division
    through fp32, which rounded 222217721 to 222217728 and produced an
    off-by-one page index (1736076 instead of 1736075) before the
    shift-based implementation."""
    storage_block_size = 128
    slots = torch.tensor(
        [
            222217721,  # fp32-rounds to 222217728; page must stay 1736075
            222217720,  # even: incomplete compression group when ratio == 2
            2**24,  # first fp32-inexact int32 value
            2**24 + 127,  # storage-block boundary at scale
            2**28 + 121,
            2**31 - 130,  # near int32 max, even
            2**31 - 129,  # near int32 max, odd
            -1,
            storage_block_size - 1,
            storage_block_size,
        ],
        dtype=dtype,
        device="npu",
    )
    num_tokens = slots.shape[0]
    slot2d, flat = _make_outputs(num_tokens)
    compute_slot_mapping_2d(
        active_slots=slots,
        storage_block_size=storage_block_size,
        ratio=ratio,
        slot2d=slot2d,
        flat=flat,
    )
    _assert_matches(slot2d, flat, slots, storage_block_size, ratio, num_tokens)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="Requires Ascend A2/A3",
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("ratio", [1, 2])
def test_compute_slot_mapping_2d_overwrites_stale(dtype, ratio):
    """Repeated writes into the same buffers: invalid slots must overwrite -1."""
    storage_block_size = 128
    num_tokens = 16
    slot2d, flat = _make_outputs(num_tokens)

    # First write: all valid slots (odd for ratio=2 so every group is complete).
    valid_slots = torch.arange(num_tokens, dtype=dtype, device="npu") * storage_block_size + 1
    compute_slot_mapping_2d(
        active_slots=valid_slots,
        storage_block_size=storage_block_size,
        ratio=ratio,
        slot2d=slot2d,
        flat=flat,
    )
    slot2d_ref, flat_ref = _eager_reference(valid_slots, storage_block_size, ratio)
    torch.testing.assert_close(slot2d, slot2d_ref, rtol=0, atol=0)
    torch.testing.assert_close(flat, flat_ref, rtol=0, atol=0)

    # Second write into the same buffers, now with half the slots padded.
    padded_slots = valid_slots.clone()
    padded_slots[::2] = -1
    compute_slot_mapping_2d(
        active_slots=padded_slots,
        storage_block_size=storage_block_size,
        ratio=ratio,
        slot2d=slot2d,
        flat=flat,
    )
    _assert_matches(slot2d, flat, padded_slots, storage_block_size, ratio, num_tokens)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="Requires Ascend A2/A3",
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("ratio", [1, 2])
def test_compute_slot_mapping_2d_masked_tail(dtype, ratio):
    """The BLOCK tail must not write past num_tokens."""
    storage_block_size = 128
    num_tokens = 1025
    extra = 8
    slots = _make_slots(num_tokens, storage_block_size, ratio).to(dtype)
    slot2d = torch.full((num_tokens + extra, 2), 777, dtype=torch.int32, device="npu")
    flat = torch.full((num_tokens + extra,), 777, dtype=torch.int64, device="npu")

    compute_slot_mapping_2d(
        active_slots=slots,
        storage_block_size=storage_block_size,
        ratio=ratio,
        slot2d=slot2d,
        flat=flat,
    )
    _assert_matches(slot2d, flat, slots, storage_block_size, ratio, num_tokens)
    assert (slot2d[num_tokens:] == 777).all()
    assert (flat[num_tokens:] == 777).all()


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="Requires Ascend A2/A3",
)
def test_compute_slot_mapping_2d_dynamic_tokens_reuse_compilation(monkeypatch):
    """Changing token counts must reuse the compiled kernel (num_tokens is not specialized)."""
    compile_events = []
    original_compile = _compute_slot_mapping_2d_kernel._do_compile

    def record_compile(*args, **kwargs):
        compile_events.append(args[0])
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(_compute_slot_mapping_2d_kernel, "_do_compile", record_compile)

    def run(num_tokens, ratio):
        slots = _make_slots(num_tokens, 128, ratio).to(torch.int32)
        slot2d, flat = _make_outputs(num_tokens)
        compute_slot_mapping_2d(
            active_slots=slots,
            storage_block_size=128,
            ratio=ratio,
            slot2d=slot2d,
            flat=flat,
        )

    run(1023, 1)
    ratio1_baseline = len(compile_events)
    for num_tokens in (0, 1, 1023, 1024, 1025):
        run(num_tokens, 1)
    assert len(compile_events) == ratio1_baseline

    run(1025, 2)
    ratio2_baseline = len(compile_events)
    for num_tokens in (0, 1, 1023, 1024, 1025):
        run(num_tokens, 2)
    assert len(compile_events) == ratio2_baseline
