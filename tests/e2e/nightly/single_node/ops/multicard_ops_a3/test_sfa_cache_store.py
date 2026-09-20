# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.attention.cache_store import build_block_cache_groups, try_store_kv_blocks
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("dtype,width", [(torch.int8, 656), (torch.bfloat16, 128), (torch.float16, 128)])
@pytest.mark.parametrize("tokens", [2048, 8192])
@torch.inference_mode()
def test_block_cache_store_matches_scatter_with_distinct_cache_slots(dtype, width, tokens):
    torch_npu.npu.set_device(0)
    assert enable_custom_op()
    block_size, blocks = 128, 128
    logical = torch.arange(tokens, device="npu", dtype=torch.int32) + 37
    slots = (blocks - 2 - logical // block_size) * block_size + logical % block_size
    slots[-3:] = -1
    key = (torch.arange(tokens * width, device="npu").reshape(tokens, width) % 251 - 125).to(dtype)
    cache = torch.full((blocks, block_size, 1, width), -7, device="npu", dtype=dtype)
    reference = cache.clone()
    metadata = SimpleNamespace(block_size=block_size, **build_block_cache_groups(slots, block_size))
    for delta in (0, 1):
        key.add_(delta)
        torch_npu.npu_scatter_nd_update_(reference.view(-1, width), slots.view(-1, 1), key)
        assert try_store_kv_blocks(key, cache, metadata)
        torch.npu.synchronize()
        torch.testing.assert_close(cache, reference, rtol=0, atol=0)
