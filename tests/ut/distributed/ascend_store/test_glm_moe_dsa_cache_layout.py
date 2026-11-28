# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_layout import KVCacheLayout

from tests.ut.distributed.ascend_store.test_layerwise_cache_layout import _runtime
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import plan_layerwise_cache_reuse
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import get_cache_views
from vllm_ascend.models.glm_moe_dsa_cache_layout import normalize_offload_descriptors


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


def _glm_group(specs):
    return KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=specs))


def _builder():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    return AscendKVCacheConfigBuilder()


@pytest.mark.parametrize("blocks", [1, 7881])
@pytest.mark.parametrize("c8", [False, True])
@pytest.mark.parametrize("query_heads", [1, 64])
def test_real_builder_uses_physical_mla_and_li_pages(blocks, c8, query_heads):
    _, _, specs = _glm_specs(c8, query_heads)
    config = _builder().get_kv_cache_config_from_groups(_glm_runtime(), [_glm_group(specs)], blocks)
    views = get_cache_views(config, specs)
    assert set(views) == set(specs)
    assert all(t.block_stride == specs[n].page_size_bytes for t in config.kv_cache_tensors for n in t.layers)
    assert specs["model.layers.0.self_attn.attn"].num_heads == query_heads
    windows = sorted((start, end) for _, start, end in views.values())
    assert all(left[1] <= right[0] for left, right in zip(windows, windows[1:]))


def _native_glm_config():
    _, _, specs = _glm_specs()
    runtime = _glm_runtime(num_buffers=None)
    config = _builder().get_kv_cache_config_from_groups(runtime, [_glm_group(specs)], 3)
    return runtime, config, specs


def test_normalization_preserves_query_heads_and_is_idempotent():
    runtime, config, specs = _native_glm_config()
    normalize_offload_descriptors(runtime, config, specs)
    tensors = config.kv_cache_tensors
    assert set(get_cache_views(config, specs)) == set(specs)
    assert specs["model.layers.0.self_attn.attn"].num_heads == 64
    normalize_offload_descriptors(runtime, config, specs)
    assert config.kv_cache_tensors is tensors


@pytest.mark.parametrize("mode", ["disabled", "unknown_model", "other_layout", "empty"])
def test_non_offload_and_unsupported_layouts_are_unchanged(mode):
    runtime, config, specs = _native_glm_config()
    if mode == "unknown_model":
        runtime = _glm_runtime(model_type="unknown")
    elif mode == "other_layout":
        runtime.cache_config.get_resolved_kv_cache_layout = lambda: SimpleNamespace(name="NBLHC")
    elif mode == "empty":
        config.num_blocks = 0
        config.kv_cache_tensors = []
    tensors = config.kv_cache_tensors
    if mode in ("other_layout", "empty"):
        normalize_offload_descriptors(runtime, config, specs)
    else:
        assert plan_layerwise_cache_reuse(config, runtime) is None
    assert config.kv_cache_tensors is tensors


@pytest.mark.parametrize("malformation", ["duplicate", "missing", "empty_layers", "size", "stride"])
def test_unrecognized_native_descriptors_fail_without_partial_mutation(malformation):
    runtime, config, specs = _native_glm_config()
    tensor = config.kv_cache_tensors[0]
    if malformation == "duplicate":
        config.kv_cache_tensors.append(tensor)
    elif malformation == "missing":
        config.kv_cache_tensors.pop()
    elif malformation == "empty_layers":
        config.kv_cache_tensors[0] = replace(tensor, layers=[])
    elif malformation == "size":
        config.kv_cache_tensors[-1] = replace(config.kv_cache_tensors[-1], size=1)
    else:
        config.kv_cache_tensors[-1] = replace(config.kv_cache_tensors[-1], block_stride=7)
    tensors = config.kv_cache_tensors
    with pytest.raises(ValueError, match="GLM-5 offload"):
        normalize_offload_descriptors(runtime, config, specs)
    assert config.kv_cache_tensors is tensors
