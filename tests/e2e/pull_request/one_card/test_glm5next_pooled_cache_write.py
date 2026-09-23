# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.paged_cache import write_pooled_cache


@pytest.mark.parametrize("page_padding", [0, 327680 - 16 * 128])
@pytest.mark.parametrize("use_graph", [False, True])
def test_pooled_cache_native_scatter_preserves_shared_storage(page_padding, use_graph):
    blocks, block_size, head_dim = 3, 16, 128
    page_stride = block_size * head_dim + page_padding
    backing = torch.full((blocks * page_stride + 2 * head_dim,), 17, dtype=torch.bfloat16, device="npu")
    cache = backing.as_strided((blocks, block_size, 1, head_dim), (page_stride, head_dim, head_dim, 1), head_dim)
    slots = torch.empty(6, dtype=torch.int64, device="npu")
    values = torch.empty(6, head_dim, dtype=cache.dtype, device="npu")
    slots.fill_(-1)
    values.fill_(float("nan"))

    def run():
        write_pooled_cache(cache, slots, values)

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph() if use_graph else None
    if graph is not None:
        with torch.npu.graph(graph):
            run()

    # Change device-side indices and values without changing captured shapes.
    for selected in ([-1, 0, 31, 47, 48, -2], [-1, 1, 16, 32, 49, -1], [-1] * 6):
        backing.fill_(17)
        slots.copy_(torch.tensor(selected, dtype=torch.int64, device="npu"))
        source = torch.full((6, head_dim), float("nan"), dtype=cache.dtype)
        expected = torch.full(backing.shape, 17, dtype=cache.dtype)
        for row, slot in enumerate(selected):
            if 0 <= slot < blocks * block_size:
                source[row].fill_(row + 1)
                offset = head_dim + (slot // block_size) * page_stride + (slot % block_size) * head_dim
                expected[offset : offset + head_dim] = row + 1
        values.copy_(source)
        if graph is None:
            run()
        else:
            graph.replay()
        torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
