# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise an MRV2 lookback window across consecutive scheduling steps."""

import pytest
import torch
import torch_npu
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.engram_lookback import _gather_lookback_kernel


def check_lookback_window(depth, width, row_padding, first_slots, initial_counts):
    counts = list(initial_counts)
    capacity = len(counts)
    stride = width + row_padding
    storage = torch.arange(capacity * stride, dtype=torch.int32, device="cpu").view(capacity, stride)
    history = storage[:, :width]
    device_history = storage.to("npu")[:, :width]

    # Guard both ends while keeping the persistent window aligned across launches.
    guard_size = 32
    sentinel = -9876
    backing = torch.full((guard_size + capacity * depth + guard_size,), sentinel, dtype=torch.int32, device="npu")
    window = backing[guard_size:-guard_size].view(capacity, depth)
    slots_by_id = list(range(capacity))
    last_slot = capacity - 1
    batch_slots = [first_slots, [last_slot - 1], slots_by_id[::-1], slots_by_id[1:]]
    for step, slots in enumerate(batch_slots):
        if step == 2:
            # The last slot returns as a fresh request with the previous occupant's tokens still in storage.
            counts[last_slot] = 0
        elif step == 3:
            counts[last_slot] = 1
            history[last_slot, 0] = 0
            device_history[last_slot, :1].copy_(history[last_slot, :1])

        mapping = torch.tensor(slots, dtype=torch.int64, device="npu")
        computed = torch.tensor(counts, dtype=torch.int32, device="npu")
        _gather_lookback_kernel[(capacity,)](
            window,
            mapping,
            computed,
            device_history,
            device_history.stride(0),
            len(slots),
            DEPTH=depth,
            BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        torch_npu.npu.synchronize()

        reference = torch.full((capacity, depth), -1, dtype=torch.int32, device="cpu")
        for row, slot in enumerate(slots):
            end = counts[slot]
            recent = history[slot, max(0, end - depth) : end].flip(0)
            reference[row, : recent.numel()] = recent
        expected = torch.full((backing.numel(),), sentinel, dtype=torch.int32, device="cpu")
        expected[guard_size:-guard_size] = reference.flatten()
        torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("depth", "width", "row_padding"),
    [(1, 4, 0), (2, 5, 2), (3, 6, 2), (4, 7, 0), (5, 8, 2), (5, 3, 0)],
)
def test_lookback_window_across_batches(depth, width, row_padding):
    counts = [0, 1, min(depth - 1, width), min(depth, width), min(depth + 1, width), width]
    check_lookback_window(depth, width, row_padding, first_slots=[2, 3, 4, 5, 0, 1], initial_counts=counts)


def test_reported_full_batch_with_empty_slot_zero():
    # The first step uses the slot mapping, device counts and history stride recorded in #21392.
    check_lookback_window(
        depth=3,
        width=128 * 1024,
        row_padding=0,
        first_slots=[7, 6, 4, 3, 0, 5, 2, 1],
        initial_counts=[0, 0, 0, 0, 0, 0, 17, 21],
    )
