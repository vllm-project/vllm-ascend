# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.device.device_config import AscendDeviceType, get_ascend_device_type
from vllm_ascend.ops.triton.mla_cache_store import scatter_component
from vllm_ascend.worker.v2.attn_utils import _reshape_combined_mla_kv_cache


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Ling MLA writer uses the A3 page layout")
def test_block_24_writes_both_components_without_changing_kda_page_5():
    block_size, ratio, num_pages = 128, 4, 7
    nope_width, rope_width = 512, 64
    dtype = torch.bfloat16
    page_bytes = ratio * block_size * (nope_width + rope_width) * dtype.itemsize
    raw = torch.full((num_pages * page_bytes,), 37, dtype=torch.uint8, device="npu")
    shape = (num_pages * ratio, block_size, 1)
    nope, rope = _reshape_combined_mla_kv_cache(
        raw, (*shape, nope_width), (*shape, rope_width), dtype, page_bytes, ratio
    )
    slots = torch.tensor([3072, 3073, 3074, 3075, 3076, -1], device=raw.device)
    expected = raw.cpu()
    expected_nope, expected_rope = _reshape_combined_mla_kv_cache(
        expected, (*shape, nope_width), (*shape, rope_width), dtype, page_bytes, ratio
    )
    for cache, ref, value in ((nope, expected_nope, 1), (rope, expected_rope, 2)):
        rows = torch.full((slots.numel(), 1, cache.shape[-1]), value, dtype=dtype, device=raw.device)
        scatter_component(rows, slots, cache)
        ref[24, :5].fill_(value)
    # Compare the complete shared pool, including the old 5,120-byte collision.
    torch.testing.assert_close(raw.cpu(), expected)
