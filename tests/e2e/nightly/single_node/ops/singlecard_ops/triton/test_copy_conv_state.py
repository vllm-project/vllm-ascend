# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.kda.conv_state import copy_conv_state


@pytest.mark.parametrize("dim_first", [False, True])
def test_conv_state_copy_masks_invalid_slots_and_preserves_shared_pages(dim_first):
    slots, width, dim = 4, 6, 384
    # Leave a gap after each physical state and preserve the alternate layout.
    backing = torch.arange(slots * width * dim * 2, dtype=torch.float32, device="npu")
    strides = (width * dim * 2, 1, width) if dim_first else (width * dim * 2, dim, 1)
    cache = backing.as_strided((slots, width, dim), strides)
    saved = backing.clone()
    indices = torch.tensor([2, -1, slots, 1, 3], dtype=torch.int32, device="npu")
    starts = torch.tensor([0, 1, 2, 3, 3, 4], dtype=torch.int32, device="npu")
    packed = torch.empty(6, width, dim, device="npu")
    packed_indices = torch.empty_like(indices)
    copy_conv_state(cache, packed, indices, starts, packed_indices, write_back=False)
    torch.testing.assert_close(packed_indices.cpu(), torch.tensor([0, 5, 5, 5, 4], dtype=torch.int32))
    torch.testing.assert_close(packed[0], cache[2])
    torch.testing.assert_close(packed[4], cache[3])
    assert torch.count_nonzero(packed[1:4]).item() == 0
    packed.fill_(17)
    copy_conv_state(cache, packed, indices, starts, packed_indices, write_back=True)
    expected = saved.as_strided(cache.shape, strides)
    expected[2].fill_(17)
    expected[3].fill_(17)
    torch.testing.assert_close(backing, saved)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("use_graph", [False, True])
def test_conv_state_copy_padding_and_graph_replay(dtype, dim_first, use_graph):
    requests, state_len, dim = 8, 3, 1536
    slots = requests + 3
    # Integer-valued payloads and updates are exact in both tested dtypes.
    storage = (torch.arange(slots * state_len * dim * 2, dtype=torch.int32, device="cpu") % 97 - 48).to(dtype)
    strides = (state_len * dim * 2, 1, state_len) if dim_first else (state_len * dim * 2, dim, 1)
    initial_backing = storage.npu()
    backing = initial_backing.clone()
    cache = backing.as_strided((slots, state_len, dim), strides)
    reference_cache = storage.as_strided(cache.shape, strides)
    null_slot = requests
    packed = torch.full((requests + 1, state_len, dim), -23, dtype=dtype, device="npu")
    packed_indices = torch.full((requests,), -17, dtype=torch.int32, device="npu")
    indices = torch.arange(2, requests + 2, dtype=torch.int32, device="npu")
    starts = torch.arange(requests + 1, dtype=torch.int32, device="npu")

    def copy_states():
        copy_conv_state(cache, packed, indices, starts, packed_indices, write_back=False)
        # Update packed rows directly, including inactive ones, to test the
        # copy operator's writeback mask without a convolution dependency.
        packed[:requests].add_(17)
        copy_conv_state(cache, packed, indices, starts, packed_indices, write_back=True)

    graph = None
    if use_graph:
        copy_states()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            copy_states()

    live_counts = (requests, requests - 1, requests - 2, requests // 2 + 1, 0, requests)
    for step, live in enumerate(live_counts):
        # Mutate both captured metadata buffers and reactivate an empty batch.
        indices_cpu = torch.arange(2, requests + 2, dtype=torch.int32, device="cpu").roll(step)
        indices_cpu[live:] = 0
        starts_cpu = torch.arange(requests + 1, dtype=torch.int32, device="cpu").clamp(max=live)
        indices.copy_(indices_cpu)
        starts.copy_(starts_cpu)
        backing.copy_(initial_backing)
        packed.fill_(-23)
        packed_indices.fill_(-17)

        expected_storage = storage.clone()
        expected_cache = expected_storage.as_strided(cache.shape, strides)
        expected_packed = torch.full(packed.shape, -23, dtype=dtype, device="cpu")
        expected_packed[:requests].zero_()
        expected_indices = torch.full((requests,), null_slot, dtype=torch.int32, device="cpu")
        for request in range(live):
            slot = int(indices_cpu[request])
            expected_packed[request].copy_(reference_cache[slot])
            expected_indices[request] = request
            expected_cache[slot].copy_(expected_packed[request] + 17)
        expected_packed[:requests].add_(17)

        if graph is None:
            copy_states()
        else:
            graph.replay()
        torch.testing.assert_close(packed_indices.cpu(), expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(packed.cpu(), expected_packed, rtol=0, atol=0)
        # Include the reserved null slot, unused cache pages and physical gaps.
        torch.testing.assert_close(backing.cpu(), expected_storage, rtol=0, atol=0)
