# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.small_indexer_row_store import try_store_small_indexer_rows


@pytest.mark.skipif(
    not get_current_hardware_profile().supports(HardwareCapability.SMALL_INDEXER_ROW_STORE),
    reason="Small indexer row store requires a supported hardware profile",
)
@pytest.mark.parametrize("rows,width", [(1, 128), (2, 128), (32, 144), (32, 2048)])
@pytest.mark.parametrize("dtype,index_dtype", [(torch.bfloat16, torch.int32), (torch.float16, torch.int64)])
def test_changes_only_addressed_rows_across_replays(rows, width, dtype, index_dtype):
    capacity = 384
    cache = torch.full((3, 128, 1, width), -1, dtype=dtype, device="npu")
    expected = cache.cpu().view(capacity, width)
    updates = torch.empty((rows, width + 16), dtype=dtype, device="npu")[:, :width]
    slots = torch.full((rows,), -1, dtype=index_dtype, device="npu")
    assert try_store_small_indexer_rows(cache, updates, slots)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        assert try_store_small_indexer_rows(cache, updates, slots)
    for step, first in enumerate(([0, 127, 128, capacity - 1, -1], [capacity - 1, -1, capacity, 0, 128])):
        indices = (first + list(range(10, 10 + rows)))[:rows]
        values = ((torch.arange(rows * width).view(rows, width) + step * 17) % 249).to(dtype)
        updates.copy_(values)
        slots.copy_(torch.tensor(indices, dtype=index_dtype))
        graph.replay()
        for i, slot in enumerate(indices):
            if 0 <= slot < capacity:
                expected[slot] = values[i]
        assert torch.equal(cache.cpu().view_as(expected), expected)
