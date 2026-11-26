# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.config import KVTransferConfig
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import LayerwiseCachePlanner
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
)


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("indexer_c8", [False, True])
def test_layerwise_plan_shares_each_physical_slot(reuse, indexer_c8):
    main_names = [f"model.layers.{i}.self_attn.attn" for i in range(4)] + ["model.mtp.0.self_attn.attn"]
    indexer_names = [f"model.layers.{i}.self_attn.indexer.k_cache" for i in (1, 2, 3)]
    main_spec = AscendMLAAttentionSpec(block_size=2, num_kv_heads=1, head_size=12, dtype=torch.bfloat16)
    indexer_spec = AscendSFAIndexerCacheSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8 if indexer_c8 else torch.bfloat16,
        scale_dim=1 if indexer_c8 else 0,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=indexer_c8,
    )
    num_blocks = 3
    backing_size = (
        max(len(main_names) * main_spec.page_size_bytes, len(indexer_names) * indexer_spec.page_size_bytes) * num_blocks
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=backing_size,
                layers=names,
                layer_stride=num_blocks * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
            for names, spec in ((main_names, main_spec), (indexer_names, indexer_spec))
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)
            for names, spec in ((main_names, main_spec), (indexer_names, indexer_spec))
        ],
    )
    extra_config = {"layerwise_num_shared_buffers": 2} if reuse else {}
    transfer = KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={"backend": "memcache", "use_layerwise": True, **extra_config},
    )
    runtime = SimpleNamespace(model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: 4))
    layout = LayerwiseCachePlanner(
        config, runtime, extra_config, spec_reuse_rules={AscendMLAAttentionSpec: True}
    ).build()
    original_tensors = list(config.kv_cache_tensors)
    assert layout.has_layer_reuse is reuse
    if not reuse:
        assert not layout.kv_cache_tensors
        assert config.kv_cache_tensors == original_tensors
        return

    worker_config = apply_layerwise_kv_cache_plan(config, transfer, layout)
    assert config.kv_cache_tensors == original_tensors
    descriptors = worker_config.kv_cache_tensors
    assert len({descriptor.offset for descriptor in descriptors}) == len(descriptors)
    expected_bytes = num_blocks * (4 * main_spec.page_size_bytes + 3 * indexer_spec.page_size_bytes)
    assert {descriptor.size for descriptor in descriptors} == {expected_bytes}
    assert all(
        descriptor.layer_stride == 0 and descriptor.offset + num_blocks * descriptor.block_stride <= expected_bytes
        for descriptor in descriptors
    )
    covered = [name for descriptor in descriptors for name in descriptor.layers]
    assert sorted(covered) == sorted(main_names + indexer_names)
    assert len(covered) == len(set(covered))

    assert layout.reused_cache_names == frozenset((main_names[1], main_names[3]))
    assert layout.draft_entry_names == frozenset((main_names[-1],))
    main_slots = [d for d in descriptors if set(d.layers) <= set(main_names)]
    indexer_slots = [d for d in descriptors if set(d.layers) <= set(indexer_names)]
    assert len(main_slots) + len(indexer_slots) == len(descriptors)
    assert all(d.block_stride == main_spec.page_size_bytes for d in main_slots)
    assert all(d.block_stride == indexer_spec.page_size_bytes for d in indexer_slots)
