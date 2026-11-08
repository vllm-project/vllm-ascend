# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Cache owners publish the formats used by their execution implementations."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SparseCacheRole

from vllm_ascend.attention import indexer as indexer_module
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend, AscendSFAIndexerCache
from vllm_ascend.attention.mla_v1 import AscendMLAAttention, AscendMLAImpl
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec


@pytest.mark.parametrize("model_type,heads", [("glm_moe_dsa", 12), ("kimi_linear", 1)])
@pytest.mark.parametrize("use_v2", [False, True])
def test_mla_owner_preserves_upstream_spec_metadata(monkeypatch, model_type, heads, use_v2):
    source = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        is_draft=True,
        dcp_sharded=True,
        state_content_bytes=1152,
        block_stride_alignment=256,
        non_causal_multi_token_decode=True,
        cache_role=SparseCacheRole.SPARSE,
        is_index_group_leader=True,
    )
    monkeypatch.setattr(MLAAttention, "get_kv_cache_spec", lambda *_: source)
    layer = AscendMLAAttention.__new__(AscendMLAAttention)
    torch.nn.Module.__init__(layer)
    layer.num_heads = 12
    layer.model_version = None
    layer.indexes_kv_by_block_stride = True
    layer.impl = AscendMLAImpl.__new__(AscendMLAImpl)
    layer.impl.fa_quant_layer = False
    config = SimpleNamespace(
        use_v2_model_runner=use_v2,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type)),
    )
    actual = layer.get_kv_cache_spec(config)
    if not use_v2:
        assert actual is source
        return
    assert isinstance(actual, AscendMLAAttentionSpec)
    assert actual.num_heads == heads
    for field in (
        "head_size",
        "dtype",
        "is_draft",
        "dcp_sharded",
        "state_content_bytes",
        "block_stride_alignment",
        "non_causal_multi_token_decode",
        "cache_role",
        "is_index_group_leader",
    ):
        assert getattr(actual, field) == getattr(source, field)


@pytest.mark.parametrize("format_name,width", [("bf16", 576), ("c8", 656), ("tq", 386)])
@pytest.mark.parametrize("offload", [False, True])
def test_sfa_spec_uses_execution_format_and_offload_decision(format_name, width, offload):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.enable_sparse_sfa_c8 = format_name == "c8"
    impl.enable_sparse_sfa_turboquant = format_name == "tq"
    impl.sfa_qsfa_packed_kv_head_dim = width
    impl.c8_cache_dtype = torch.int8
    impl.kv_cache_dtype = "turboquant" if format_name == "tq" else "int8"
    impl.store_kv_cache_on_host = offload
    source = AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
    spec = impl.customize_kv_cache_spec(source)
    assert spec.head_size == width
    assert spec.dtype == (torch.bfloat16 if format_name == "bf16" else torch.int8)
    assert spec.store_on_host is offload
    assert spec.cache_sparse_sfa_c8 is (format_name == "c8")
    assert spec.cache_sparse_sfa_turboquant is (format_name == "tq")
    assert spec.page_size_bytes == 128 * (1152 if format_name == "bf16" else width)


def test_fa_quant_spec_uses_execution_dtype():
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.fa_quant_layer = True
    impl.head_size = 576
    impl.qk_rope_head_dim = 64
    impl.dtype = torch.int8
    spec = impl.customize_kv_cache_spec(
        AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
    )
    assert spec.head_size == 640
    assert spec.dtype == torch.int8
    assert spec.cache_dtype_str is None
    assert spec.page_size_bytes == 128 * 640


@pytest.mark.parametrize("c4", [False, True])
@pytest.mark.parametrize("c8", [False, True])
@pytest.mark.parametrize("replicas", [1, 4])
def test_indexer_owner_reports_data_scale_and_replication(monkeypatch, c4, c8, replicas):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128, cache_dtype="auto"),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(decode_context_parallel_size=replicas),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(indexer_module, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(indexer_module, "enable_sfa_dcp_replicated_indexer", lambda _: replicas > 1)
    backend = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    torch.nn.Module.__init__(backend)
    backend.head_dim = 128
    backend.enable_sparse_li_c4, backend.enable_sparse_li_c8 = c4, c8
    backend.c4_k_cache_dtype, backend.c4_k_scale_cache_dtype = torch.uint8, torch.float8_e8m0fnu
    backend.c8_k_cache_dtype, backend.c8_k_scale_cache_dtype = torch.int8, torch.float16
    source = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        cache_role=SparseCacheRole.INDEXER,
        is_draft=True,
    )
    spec = backend.get_kv_cache_spec(source, config)
    assert isinstance(spec, AscendSFAIndexerCacheSpec)
    assert spec.cache_role is source.cache_role and spec.is_draft
    assert spec.sfa_dcp_replicated_indexer_size == replicas
    assert spec.scale_dim == (4 if c4 else 1 if c8 else 0)
    assert spec.head_size == (64 if c4 else 128)
    assert spec.page_size_bytes == replicas * 128 * (68 if c4 else 130 if c8 else 256)

    prefix = "model.layers.0.self_attn.indexer.k_cache"
    old_owner = SimpleNamespace(prefix=prefix, kv_cache=torch.empty(0), cache_config=config.cache_config)
    config.compilation_config.static_forward_context[prefix] = old_owner
    owner = AscendSFAIndexerCache(old_owner, spec)
    assert config.compilation_config.static_forward_context[prefix] is owner
    assert owner.get_kv_cache_spec(config) == spec
    owner.cache_enabled = False
    assert owner.get_kv_cache_spec(config) is None
    owner.cache_enabled = True
    assert owner.get_kv_cache_spec(config) == spec
    assert owner.get_attn_backend() is AscendSFAIndexerBackend


def test_worker_collects_complete_specs_without_inspecting_impl(monkeypatch):
    from vllm_ascend.worker.v2 import attn_utils

    spec = AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)

    class Owner:
        def get_kv_cache_spec(self, _config):
            return spec

        @property
        def impl(self):
            raise AssertionError("Spec collection must not inspect the execution implementation")

    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: {"main": Owner()})
    config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="auto"))
    assert attn_utils.get_kv_cache_spec(config)["main"] is spec
