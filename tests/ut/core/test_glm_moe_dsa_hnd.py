# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.worker.v2 import attn_utils


def test_glm_dflash_hnd_placement(monkeypatch):
    common = dict(block_size=128, num_kv_heads=1)
    index = AscendSFAIndexerCacheSpec(
        **common,
        head_size=128,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float32,
        sfa_dcp_replicated_indexer_size=8,
    )
    main = AscendMLAAttentionSpec(**common, num_heads=8, head_size=656, dtype=torch.int8, cache_sparse_sfa_c8=True)
    draft = SlidingWindowSpec(
        **common, head_size=128, dtype=torch.bfloat16, sliding_window=2048, is_draft=True, dcp_sharded=False
    )
    # Indexer registered first; one target layer has no corresponding indexer.
    specs = {"index": index, "mla0": main, "mla1": main, "draft": draft}
    model = SimpleNamespace(model_type="glm_moe_dsa")
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=model, hf_text_config=model, max_model_len=8192),
        speculative_config=SimpleNamespace(method="dflash"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=8),
        cache_config=CacheConfig(),
        kv_transfer_config=None,
        additional_config={},
    )
    config.cache_config.kv_cache_layout = "LBHNC"
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    builder = AscendKVCacheConfigBuilder()
    groups = builder.get_kv_cache_groups(config, specs)
    assert [g.layer_names for g in groups] == [["mla0", "mla1", "index"], ["draft"]]
    assert [g.kv_cache_spec.max_num_blocks_per_req(config, 8192) for g in groups] == [8, 64]
    for local_groups in (groups, [groups[0], replace(groups[1], layer_names=[])]):
        placement = builder.get_kv_cache_config_from_groups(config, local_groups, num_blocks=2)
        local_specs = {name: specs[name] for g in local_groups for name in g.layer_names}
        layer_bytes = {name: 2 * spec.page_size_bytes for name, spec in local_specs.items()}
        assert 2 * builder._get_pool_bytes_per_block(config, local_groups) == sum(layer_bytes.values())
        views = attn_utils._allocate_kv_cache(placement, {}, torch.device("cpu"))
        assert len({v.untyped_storage().data_ptr() for v in views.values()}) == 1
        for tensor in placement.kv_cache_tensors:
            assert tensor.size == sum(layer_bytes.values())
            for name in tensor.layers:
                assert tensor.block_stride == specs[name].page_size_bytes
                assert tensor.layer_stride == layer_bytes[name]
        for i, name in enumerate(local_specs):
            view = views[name].view(2, -1)
            view[0].fill_(2 * i)
            view[1].fill_(2 * i + 1)
        for i, name in enumerate(local_specs):
            view = views[name].view(2, -1)
            assert (view[0] == 2 * i).all() and (view[1] == 2 * i + 1).all()
    with pytest.raises(ValueError, match="compatible"):
        builder.get_kv_cache_groups(config, {**specs, "mla1": replace(main, dcp_sharded=False)})


def test_kvpp_promoted_draft_shares_target_block_ids():
    model = SimpleNamespace(model_type="glm_moe_dsa")
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=model, hf_text_config=model, max_model_len=1048576),
        speculative_config=SimpleNamespace(method="dflash"),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=2, tensor_parallel_size=4
        ),
        cache_config=CacheConfig(),
        kv_transfer_config=None,
        additional_config={},
    )
    config.cache_config.kv_cache_layout = "LBHNC"
    common = dict(block_size=128, num_kv_heads=1)
    main = AscendMLAAttentionSpec(**common, num_heads=8, head_size=656, dtype=torch.int8, cache_sparse_sfa_c8=True)
    draft = FullAttentionSpec(**common, head_size=128, dtype=torch.bfloat16, sliding_window=2048, is_draft=True)
    specs = {"main": main, "draft": draft}
    builder = AscendKVCacheConfigBuilder()
    with pytest.raises(ValueError, match="Unexpected.*FullAttentionSpec"):
        builder.get_kv_cache_groups(config, specs)
    config.additional_config["enable_kvpp"] = True
    groups = builder.get_kv_cache_groups(config, specs)
    assert [group.layer_names for group in groups] == [["main", "draft"]]
    placement = builder.get_kv_cache_config_from_groups(config, groups, num_blocks=2)
    assert {t.size for t in placement.kv_cache_tensors} == {2 * (main.page_size_bytes + draft.page_size_bytes)}
    with pytest.raises(ValueError, match="Unexpected.*FullAttentionSpec"):
        builder.get_kv_cache_groups(config, {**specs, "draft": replace(draft, is_draft=False)})
