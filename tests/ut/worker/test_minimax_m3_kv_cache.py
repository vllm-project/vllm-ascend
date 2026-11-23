# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 indexer pages must follow the kernel block table."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.models.minimax_m3.msa_m3 import AscendMiniMaxM3IndexerBackend
from vllm_ascend.worker.v2 import attn_utils


@pytest.mark.parametrize("scheduler_block_size,expected_pages", [(128, 2), (256, 4), (512, 8)])
@pytest.mark.parametrize("scale_dim", [0, 1])
def test_indexer_views_split_scheduler_blocks_into_kernel_pages(
    monkeypatch, scheduler_block_size, expected_pages, scale_dim
):
    name = "model.layers.3.self_attn.indexer.k_cache"
    spec = AscendSFAIndexerCacheSpec(
        block_size=scheduler_block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        scale_dim=scale_dim,
        scale_dtype=torch.float32,
    )
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)],
    )
    group = SimpleNamespace(
        kv_cache_spec=spec, kv_cache_group_id=0, layer_names=[name], backend=AscendMiniMaxM3IndexerBackend
    )
    key = torch.arange(2 * scheduler_block_size * 128).to(torch.bfloat16)
    raw = [key.view(torch.uint8)]
    if scale_dim:
        scales = torch.arange(2 * scheduler_block_size, dtype=torch.float32)
        raw.append(scales.view(torch.uint8))
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: SimpleNamespace())

    cache = attn_utils._reshape_kv_cache_v2([group], {name: tuple(raw)}, "auto", [128], {}, kv_cache_config=config)[
        name
    ]

    assert cache[0].shape == (expected_pages, 128, 128)
    assert cache[0].data_ptr() == key.data_ptr()
    torch.testing.assert_close(cache[0].flatten(), key)
    if scale_dim:
        assert cache[1].shape == (expected_pages, 128, 1)
        assert cache[1].data_ptr() == scales.data_ptr()
        torch.testing.assert_close(cache[1].flatten(), scales)
