# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Cache owners publish the formats used by their execution implementations."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.v1.kv_cache_interface import KVQuantMode, MLAAttentionSpec, SparseCacheRole

from vllm_ascend.attention.mla_v1 import AscendMLAAttention, AscendMLABackend, AscendMLAImpl
from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAImpl
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec


@pytest.mark.parametrize(
    "model_type,heads",
    [("glm_moe_dsa", 1), ("glm4_moe_lite", 1), ("glm5_next", 1), ("glm5_next_text", 1),
     ("kimi_linear", 1), ("deepseek_v3", 12)],
)
def test_mla_owner_preserves_upstream_spec_metadata(monkeypatch, model_type, heads):
    source = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        kv_quant_mode=KVQuantMode.NONE,
        page_size_padded=128 * 1152 + 256,
        tokens_per_state=2,
        topk_buffer_width=2048,
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
    layer.attn_backend = AscendMLABackend
    layer.kv_lora_rank = 512
    layer.qk_rope_head_dim = 64
    layer.num_heads = 12
    layer.model_version = None
    layer.indexes_kv_by_block_stride = True
    layer.impl = AscendMLAImpl.__new__(AscendMLAImpl)
    layer.impl.fa_quant_layer = False
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type)),
    )
    actual = layer.get_kv_cache_spec(config)
    assert isinstance(actual, AscendMLAAttentionSpec)
    assert actual.num_heads == heads
    assert layer.num_heads == 12
    if "glm" in model_type:
        assert actual.real_page_size_bytes == actual.num_heads * actual.num_states * actual.state_content_size_bytes
    for field in (
        "kv_quant_mode",
        "page_size_padded",
        "tokens_per_state",
        "topk_buffer_width",
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


def _make_owner(impl, backend, rope_dim=64):
    layer = AscendMLAAttention.__new__(AscendMLAAttention)
    torch.nn.Module.__init__(layer)
    layer.impl = impl
    layer.attn_backend = backend
    layer.num_heads = 12
    layer.kv_lora_rank = 512
    layer.qk_rope_head_dim = rope_dim
    return layer


def _get_owner_spec(monkeypatch, layer):
    source = MLAAttentionSpec(
        block_size=128, num_kv_heads=1,
        head_size=layer.kv_lora_rank + layer.qk_rope_head_dim, dtype=torch.bfloat16,
    )
    monkeypatch.setattr(MLAAttention, "get_kv_cache_spec", lambda *_: source)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")),
    )
    return layer.get_kv_cache_spec(config)


@pytest.mark.parametrize("format_name,width", [("bf16", 576), ("c8", 656), ("c8_nope", 528), ("tq", 386)])
@pytest.mark.parametrize("offload", [False, True])
def test_sfa_spec_uses_owner_decisions_and_backend_packing(monkeypatch, format_name, width, offload):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.enable_sparse_sfa_c8 = format_name in ("c8", "c8_nope")
    impl.enable_sparse_sfa_turboquant = format_name == "tq"
    impl.c8_cache_dtype = torch.int8
    impl.kv_cache_dtype = "turboquant" if format_name == "tq" else "int8"
    impl.store_kv_cache_on_host = offload
    layer = _make_owner(impl, AscendSFABackend, rope_dim=0 if format_name == "c8_nope" else 64)
    spec = _get_owner_spec(monkeypatch, layer)
    assert spec.head_size == width
    assert spec.dtype == (torch.bfloat16 if format_name == "bf16" else torch.int8)
    assert spec.store_on_host is offload
    assert spec.cache_sparse_sfa_c8 is (format_name in ("c8", "c8_nope"))
    assert spec.cache_sparse_sfa_turboquant is (format_name == "tq")
    assert spec.page_size_bytes == 128 * (1152 if format_name == "bf16" else width)
    # Framework callers may invoke the standard hook again.
    assert AscendSFABackend.customize_spec(spec) == spec


@pytest.mark.parametrize("quantized", [False, True])
def test_fa_quant_decision_stays_with_each_owner(monkeypatch, quantized):
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.fa_quant_layer = quantized
    impl.head_size = 576
    impl.dtype = torch.int8
    layer = _make_owner(impl, AscendMLABackend)
    spec = _get_owner_spec(monkeypatch, layer)
    assert spec.head_size == (640 if quantized else 576)
    assert spec.dtype == (torch.int8 if quantized else torch.bfloat16)
    assert spec.page_size_bytes == 128 * (640 if quantized else 1152)
    if quantized:
        assert spec.cache_dtype_str is None


def test_sfa_backend_requires_owner_component_dimensions():
    spec = AscendMLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=576,
        dtype=torch.int8, cache_sparse_sfa_c8=True,
    )
    with pytest.raises(ValueError, match="NoPE and RoPE dimensions"):
        AscendSFABackend.customize_spec(spec)


def test_worker_collects_mla_spec_without_reading_impl(monkeypatch):
    from vllm_ascend.worker.v2 import attn_utils

    source = AscendMLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=386, dtype=torch.int8, cache_sparse_sfa_turboquant=True
    )

    class Owner(MLAAttention):
        def get_kv_cache_spec(self, _config):
            return source

        @property
        def impl(self):
            raise AssertionError("Spec collection must not inspect the execution implementation")

    layer = Owner.__new__(Owner)
    torch.nn.Module.__init__(layer)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: {"main": layer})
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto"),
        attention_config=SimpleNamespace(indexer_kv_dtype="auto"),
    )
    assert attn_utils.get_kv_cache_spec(config)["main"] is source
