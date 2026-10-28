# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

from vllm_ascend.worker.utils import copy_kv_cache_blocks_inplace


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn])
@pytest.mark.parametrize("reverse_views", [False, True])
@pytest.mark.parametrize("cyclic", [False, True])
def test_copy_overlaid_kv_views_on_npu(dtype, reverse_views, cyclic):
    num_blocks = 4
    page_bytes = 64
    payload_bytes = 32
    storage_offset = 16
    # Construct aliases on the device; moving views separately would allocate
    # distinct buffers and hide the bug. Compare raw bytes, including padding.
    before = torch.arange(storage_offset + num_blocks * page_bytes + 16, dtype=torch.int32).to(torch.uint8)
    raw = before.to("npu")
    state = torch.as_strided(raw, (num_blocks, 8), (page_bytes, 1), storage_offset)
    element_size = torch.empty((), dtype=dtype).element_size()
    attention = torch.as_strided(
        raw.view(dtype),
        (num_blocks, payload_bytes // element_size),
        (page_bytes // element_size, 1),
        storage_offset // element_size,
    )
    assert state.data_ptr() == attention.data_ptr()
    views = [state, attention]
    if reverse_views:
        views.reverse()
    block_pairs = [(0, 2), (2, 0), (0, 3)] if cyclic else [(0, 2)]
    expected = before.clone()
    for src, dst in block_pairs:
        src_start = storage_offset + src * page_bytes
        dst_start = storage_offset + dst * page_bytes
        expected[dst_start : dst_start + payload_bytes] = before[src_start : src_start + payload_bytes]

    copy_kv_cache_blocks_inplace(
        views,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=src, dst_block_id=dst) for src, dst in block_pairs],
    )

    torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)
