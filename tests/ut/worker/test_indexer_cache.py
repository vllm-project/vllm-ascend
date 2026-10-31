# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    allocate_shared_indexer_cache,
)


@pytest.mark.parametrize("c8_first", [False, True])
@pytest.mark.parametrize("replication", [1, 2])
def test_mixed_views_share_one_allocation_and_keep_transfer_geometry(c8_first, replication):
    bf16 = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sfa_dcp_replicated_indexer_size=replication,
    )
    c8 = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
        sfa_dcp_replicated_indexer_size=replication,
    )
    specs = {"c8": c8, "bf16": bf16} if c8_first else {"bf16": bf16, "c8": c8}
    allocations = []

    def allocate(size):
        allocations.append(size)
        return torch.zeros(size, dtype=torch.int8)

    raw = allocate_shared_indexer_cache(specs, 3, allocate)
    assert allocations == [3 * bf16.page_size_bytes]
    bf_view = raw["bf16"][0].view(torch.bfloat16).view(3, replication * 128, 128)
    data = raw["c8"][0].view(3, replication * 128, 128)
    scales = raw["c8"][1].view(torch.float16).view(3, replication * 128, 1)
    assert bf_view.data_ptr() == data.data_ptr()
    assert scales.untyped_storage().data_ptr() == data.untyped_storage().data_ptr()
    assert scales.data_ptr() - data.data_ptr() == data.numel()
    # These are the per-view byte lengths and strides used by KV transfer.
    for view, block_bytes in ((bf_view, replication * 32768), (data, replication * 16384), (scales, replication * 256)):
        assert view[0].numel() * view.element_size() == block_bytes
        assert view.stride(0) * view.element_size() == block_bytes
    data.fill_(7)
    scales.fill_(0.5)
    assert torch.all(data == 7) and torch.all(scales == 0.5)
    bf_view.fill_(1.25)
    assert torch.all(bf_view == 1.25)
    data.fill_(-3)
    scales.fill_(2)
    assert torch.all(data == -3) and torch.all(scales == 2)
