# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from tests.ut.distributed.ascend_store.cache_layout_utils import make_vllm_config
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import plan_layerwise_cache_reuse
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
)


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("indexer_c8", [False, True])
def test_layerwise_plan_shares_each_physical_slot(reuse, indexer_c8):
    main_names = [f"model.layers.{i}.self_attn.attn" for i in range(4)] + ["model.mtp.0.self_attn.attn"]
    indexer_names = [f"model.layers.{i}.self_attn.indexer.k_cache" for i in (1, 2, 3)]
    main_spec = AscendMLAAttentionSpec(
        block_size=2, num_kv_heads=1, head_size=12, dtype=torch.bfloat16, model_version="glm5_next"
    )
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
    vllm_config = make_vllm_config(4, 2 if reuse else None)
    original_tensors = list(config.kv_cache_tensors)
    layout = plan_layerwise_cache_reuse(config, vllm_config)
    if layout is not None:
        config = apply_layerwise_kv_cache_plan(config, vllm_config.kv_transfer_config, layout)

    if not reuse:
        assert config.kv_cache_tensors == original_tensors
        return

    # Descriptors select slots within one common physical backing.
    descriptors = config.kv_cache_tensors
    assert len({d.size for d in descriptors}) == 1
    assert descriptors != original_tensors
    for descriptor in descriptors:
        assert descriptor.layer_stride == 0
        assert descriptor.offset >= 0
        assert descriptor.block_stride > 0
        assert descriptor.offset + config.num_blocks * descriptor.block_stride <= descriptor.size
        assert descriptor.layers
    covered = [name for descriptor in descriptors for name in descriptor.layers]
    assert sorted(covered) == sorted(main_names + indexer_names)
    assert len(covered) == len(set(covered))

    # A slot never mixes main caches with indexer caches: their page layouts
    # differ, and only the indexer is allowed to keep a per-layer dtype.
    main_slots = [d for d in descriptors if set(d.layers) <= set(main_names)]
    indexer_slots = [d for d in descriptors if set(d.layers) <= set(indexer_names)]
    assert len(main_slots) + len(indexer_slots) == len(descriptors)
    assert all(d.block_stride == main_spec.page_size_bytes for d in main_slots)
    assert all(d.block_stride == indexer_spec.page_size_bytes for d in indexer_slots)
