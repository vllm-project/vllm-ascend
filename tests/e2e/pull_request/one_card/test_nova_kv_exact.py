# SPDX-License-Identifier: Apache-2.0
"""Check tiled exact attention against the independent window reconstruction."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.nova_kv.cache import ExactWindowPool, merge_attention_segments
from vllm_ascend.ops.triton.nova_kv.exact import (
    exact_partial,
    exact_partial_grouped,
    exact_physical,
    merge_partial_into,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("sink,recent", [(0, 31), (7, 64), (64, 256)])
@pytest.mark.parametrize("exact_kernel", [exact_partial, exact_partial_grouped], ids=["per_group", "grouped"])
def test_exact_tiles_replay_masks_partial_rings_and_invalid_slots(batch, sink, recent, exact_kernel):
    torch.manual_seed(8347)
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    window = ExactWindowPool(4, 2, 256, sink, recent, device, torch.bfloat16)
    # Stale physical rows must be masked before the recent ring is full.
    window.key.normal_(0, 0.2)
    window.value.normal_(0, 0.2)
    query = torch.randn((batch, 2, 6, 256), device=device) * 0.2
    cold = torch.randn((2, batch, 6, 256), device=device).transpose(0, 1)
    cold_lse = torch.empty(query.shape[:-1], device=device)
    lengths = torch.ones(batch, dtype=torch.int32, device=device)
    slots = torch.full((batch,), -1, dtype=torch.int32, device=device)
    partial = exact_kernel(window, query, slots, lengths, 1 / 16, 64)
    out = torch.empty_like(query)

    def run():
        exact_kernel(window, query, slots, lengths, 1 / 16, 64, partial)
        merge_partial_into(cold, cold_lse, partial, out)

    cold_lse.fill_(-float("inf"))
    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    boundaries = sorted(
        {
            0,
            1,
            max(0, sink - 1),
            sink,
            sink + 1,
            window.ring - 1,
            window.ring,
            window.capacity - 1,
            window.capacity,
            window.capacity + 1,
            30000,
            90000,
        }
    )
    for length in boundaries:
        query.normal_(0, 0.2)
        lengths.fill_(length)
        slots.copy_(torch.tensor([2, 0, 3, 1][:batch], device=device, dtype=torch.int32))
        if batch == 4:
            lengths[1] = max(0, length - 1)
        cold_lse.fill_(1 if length > window.capacity else -float("inf"))
        for invalid in (False, True):
            if invalid:
                slots[0] = -1
                cold_lse[0].fill_(-float("inf"))
            expected, expected_lse = merge_attention_segments(
                cold, cold_lse, *window.attend(query, slots, lengths, 1 / 16)
            )
            serial, serial_lse, _ = exact_physical(window, query, slots, lengths, 1 / 16, cold, cold_lse)
            shared, shared_lse, _ = window.attend_and_merge(query, slots, lengths, 1 / 16, cold, cold_lse)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(serial, expected, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(serial_lse, expected_lse, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(shared, expected, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(shared_lse, expected_lse, atol=1e-5, rtol=1e-5)
