# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from tests.ut.distributed.ascend_store.test_layerwise_cache_layout import _native_config, _runtime
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
    LayerwiseCachePlanner,
    get_spec_reuse_rules,
    plan_layerwise_cache_reuse,
)


def _glm_runtime(num_buffers=3, model_type="glm_moe_dsa"):
    runtime = _runtime(78, num_buffers, model_type=model_type)
    runtime.model_config.hf_config = runtime.model_config.hf_text_config
    runtime.cache_config.get_resolved_kv_cache_layout = lambda: KVCacheLayout.LBNHC
    runtime.cache_config.prefix_cache_retention_interval = 0
    runtime.attention_config = SimpleNamespace(hisparse_config=None)
    runtime.speculative_config = None
    return runtime


def _glm_specs(c8=True, query_heads=64):
    mla = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        num_heads=query_heads,
        head_size=656 if c8 else 576,
        dtype=torch.int8 if c8 else torch.bfloat16,
        cache_sparse_sfa_c8=c8,
    )
    indexer = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8 if c8 else torch.bfloat16,
        scale_dim=1 if c8 else 0,
        scale_dtype=torch.float32,
        cache_sparse_li_c8=c8,
    )
    specs = {f"model.layers.{i}.self_attn.attn": mla for i in range(78)}
    specs.update({f"model.layers.{i}.indexer.k_cache": indexer for i in range(21)})
    return mla, indexer, specs


@pytest.mark.parametrize("blocks", [1, 3, 7881])
def test_glm5_full_layer_reuse_budget(blocks):
    mla, indexer, _ = _glm_specs()
    groups = [
        KVCacheGroupSpec([f"model.layers.{i}.self_attn" for i in range(78)], mla),
        KVCacheGroupSpec([f"model.layers.{i}.indexer" for i in range(0, 78, 4)], indexer),
    ]
    native = _native_config(groups, blocks)
    runtime = _glm_runtime()
    layout = plan_layerwise_cache_reuse(native, runtime)
    assert layout is not None and layout.has_layer_reuse
    assert "model.layers.0.self_attn" not in layout.reused_cache_names
    assert "model.layers.0.indexer" not in layout.reused_cache_names
    assert all(f"model.layers.{i}.self_attn" in layout.reused_cache_names for i in range(1, 78))
    assert layout.kv_cache_tensors[0].size == 4 * blocks * (mla.page_size_bytes + indexer.page_size_bytes)
    assert layout.kv_cache_tensors[0].size < sum(t.size for t in native.kv_cache_tensors)
    assert all(source < layer for layer, sources in layout.prefetch_layer_map.items() for source in sources)
    runtime.kv_transfer_config.kv_connector_extra_config.pop("layerwise_num_shared_buffers")
    assert plan_layerwise_cache_reuse(native, runtime) is None


def test_glm5_rules_do_not_enable_unknown_specs_or_models():
    rules = get_spec_reuse_rules(_glm_runtime())
    dense = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    assert not LayerwiseCachePlanner.can_reuse_spec(dense, rules)
    assert get_spec_reuse_rules(_glm_runtime(model_type="unknown")) is None


def test_glm5_rules_register_both_physical_cache_types():
    mla, indexer, _ = _glm_specs()
    rules = get_spec_reuse_rules(_glm_runtime())
    assert rules is not None
    assert LayerwiseCachePlanner.can_reuse_spec(mla, rules)
    assert LayerwiseCachePlanner.can_reuse_spec(indexer, rules)
