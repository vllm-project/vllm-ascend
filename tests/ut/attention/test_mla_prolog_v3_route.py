# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routing tests for PROLOG_V3 on strided MLA caches."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm_ascend.attention.mla_v1 as mla_v1
from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.quantization.methods import (
    AscendW8A8DynamicLinearMethod,
    AscendW8A8FP8DynamicLinearMethod,
    AscendW8A8LinearMethod,
)

NOPE_DIM = 512
ROPE_DIM = 64
FUSED_DIM = NOPE_DIM + ROPE_DIM
BLOCK_SIZE = 128


def _make_impl() -> AscendMLAImpl:
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.layer_name = "model.layers.0.self_attn.attn"
    impl.use_mla_rope = True
    impl.qk_rope_head_dim = ROPE_DIM
    impl.enable_kv_nz = False
    impl.fa_quant_layer = False
    impl.support_fp8_attention = False
    impl.num_kv_heads = 1
    impl.num_heads = 8
    impl.mlapo_num_heads = 8
    impl.kv_lora_rank = NOPE_DIM
    impl.qk_nope_head_dim = 128
    impl.qk_head_dim = impl.qk_nope_head_dim + ROPE_DIM
    impl.v_head_dim = 128
    impl.q_lora_rank = 128
    impl.fused_qkv_a_proj = SimpleNamespace(weight=MagicMock())
    impl.q_proj = SimpleNamespace(weight=MagicMock(), _chunk_size=0)
    impl.q_a_layernorm = SimpleNamespace(
        weight=SimpleNamespace(data=torch.ones(128)),
        variance_epsilon=1e-6,
    )
    impl.kv_a_layernorm = SimpleNamespace(
        weight=SimpleNamespace(data=torch.ones(NOPE_DIM)),
        variance_epsilon=1e-6,
    )
    impl.mlapo_W_UK_T = torch.empty(8, 128, NOPE_DIM)
    impl.kv_b_proj = MagicMock()
    impl.layerwise_kv_cache_hook = None
    impl._prolog_v3_weights = {
        "quant_type": None,
        "weight_dq": torch.empty(128, 128),
        "weight_uq_qr": torch.empty(128, 8 * 192),
        "weight_dkv_kr": torch.empty(128, FUSED_DIM),
        "weight_uk": torch.empty(8, 128, NOPE_DIM),
        "num_heads": 8,
    }
    return impl


def _component_major_cache(blocks: int = 4):
    page = BLOCK_SIZE * FUSED_DIM
    raw = torch.arange(blocks * 2 * page, dtype=torch.float32).reshape(blocks * 2, page)
    nope = raw.as_strided(
        (blocks, BLOCK_SIZE, 1, NOPE_DIM),
        (2 * page, NOPE_DIM, NOPE_DIM, 1),
        0,
    )
    rope = raw.as_strided(
        (blocks, BLOCK_SIZE, 1, ROPE_DIM),
        (2 * page, ROPE_DIM, ROPE_DIM, 1),
        BLOCK_SIZE * NOPE_DIM,
    )
    return nope, rope


def _token_fused_cache(blocks: int = 4):
    fused = torch.arange(blocks * BLOCK_SIZE * FUSED_DIM, dtype=torch.float32).reshape(blocks, BLOCK_SIZE, 1, FUSED_DIM)
    nope = fused[..., :NOPE_DIM]
    rope = fused[..., NOPE_DIM:]
    return nope, rope


@pytest.mark.parametrize("cache_factory", [_component_major_cache, _token_fused_cache])
def test_strided_mla_caches_require_prolog_v3(cache_factory):
    impl = _make_impl()
    nope, rope = cache_factory()
    assert impl._cache_requires_prolog_v3_writer((nope, rope))


def test_contiguous_mla_cache_does_not_force_prolog_v3():
    impl = _make_impl()
    nope, rope = _component_major_cache()
    assert not impl._cache_requires_prolog_v3_writer((nope.contiguous(), rope.contiguous()))


def test_prolog_v3_weights_pad_native_query_heads(monkeypatch):
    impl = _make_impl()
    impl._prolog_v3_weights = None
    impl.num_heads = 3
    impl.num_heads_padded = 4
    impl.head_padding = 1
    impl.qk_head_dim = impl.qk_nope_head_dim + ROPE_DIM
    impl.mlapo_W_UK_T = torch.arange(3 * impl.qk_nope_head_dim * NOPE_DIM, dtype=torch.float32).reshape(
        3, impl.qk_nope_head_dim, NOPE_DIM
    )
    impl.fused_qkv_a_proj = SimpleNamespace(
        weight=SimpleNamespace(data=torch.randn(128 + FUSED_DIM, 128)),
        quant_method=SimpleNamespace(quant_method=None),
    )
    impl.q_proj = SimpleNamespace(
        weight=SimpleNamespace(data=torch.randn(3 * impl.qk_head_dim, 128)),
        _chunk_size=0,
    )
    monkeypatch.setattr("vllm_ascend.attention.mla_v1.torch_npu.npu_format_cast", lambda tensor, _: tensor)

    prepared = impl._prepare_prolog_v3_weights()

    assert prepared["num_heads"] == 4
    assert prepared["weight_uq_qr"].shape == (128, 4 * impl.qk_head_dim)
    assert prepared["weight_uk"].shape == (4, impl.qk_nope_head_dim, NOPE_DIM)


def test_prolog_v3_weights_reuse_prepared_weights_after_source_free():
    impl = _make_impl()
    impl._prolog_v3_weights = None
    quant_method = object.__new__(AscendW8A8LinearMethod)
    impl._get_context_prolog_quant_method = lambda _layer: quant_method

    weight_dq = torch.empty(1)
    weight_dkv_kr = torch.empty(2)
    weight_uq_qr = torch.empty(3)
    impl.weight_dq = weight_dq
    impl.weight_dkv_kr = weight_dkv_kr
    impl.weight_uq_qr = weight_uq_qr
    impl.dequant_scale_w_dq = torch.empty(4)
    impl.dequant_scale_w_dkv_kr = torch.empty(5)
    impl.dequant_scale_w_uq_qr = torch.empty(6)
    impl.fused_qkv_a_proj.weight = None
    impl.q_proj.weight = None

    prepared = impl._prepare_prolog_v3_weights()

    assert prepared["weight_dq"] is weight_dq
    assert prepared["weight_dkv_kr"] is weight_dkv_kr
    assert prepared["weight_uq_qr"] is weight_uq_qr
    assert prepared["dequant_scale_w_dq"] is impl.dequant_scale_w_dq


def test_prolog_v3_weights_accept_static_w8a8(monkeypatch):
    impl = _make_impl()
    impl._prolog_v3_weights = None
    quant_method = object.__new__(AscendW8A8LinearMethod)
    impl._get_context_prolog_quant_method = lambda _layer: quant_method
    fused_output_dim = impl.q_lora_rank + FUSED_DIM
    q_output_dim = impl.num_heads * (impl.qk_nope_head_dim + ROPE_DIM)
    impl.fused_qkv_a_proj = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(impl.q_lora_rank, fused_output_dim)),
        weight_scale=torch.ones(fused_output_dim),
        quant_method=SimpleNamespace(quant_method=quant_method),
    )
    impl.q_proj = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(impl.q_lora_rank, q_output_dim)),
        weight_scale=torch.ones(q_output_dim),
        _chunk_size=0,
    )
    monkeypatch.setattr("vllm_ascend.attention.mla_v1.torch_npu.npu_format_cast", lambda tensor, _: tensor)

    prepared = impl._prepare_prolog_v3_weights()

    assert prepared["quant_type"] is AscendW8A8LinearMethod
    assert prepared["weight_dq"].shape == (impl.q_lora_rank, impl.q_lora_rank)
    assert prepared["weight_dkv_kr"].shape == (impl.q_lora_rank, FUSED_DIM)
    assert prepared["weight_uq_qr"].shape == (impl.q_lora_rank, q_output_dim)
    assert prepared["dequant_scale_w_dq"].shape == (1, impl.q_lora_rank)
    assert prepared["dequant_scale_w_dkv_kr"].shape == (1, FUSED_DIM)
    assert prepared["dequant_scale_w_uq_qr"].shape == (1, q_output_dim)


def test_prolog_v3_weights_keep_int8_dynamic_scale_layout_on_a5(monkeypatch):
    impl = _make_impl()
    impl._prolog_v3_weights = None
    impl.support_fp8_attention = True
    quant_method = object.__new__(AscendW8A8DynamicLinearMethod)
    impl._get_context_prolog_quant_method = lambda _layer: quant_method
    fused_output_dim = impl.q_lora_rank + FUSED_DIM
    q_output_dim = impl.num_heads * (impl.qk_nope_head_dim + ROPE_DIM)
    impl.fused_qkv_a_proj = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(impl.q_lora_rank, fused_output_dim)),
        weight_scale=torch.ones(fused_output_dim),
        quant_method=SimpleNamespace(quant_method=quant_method),
    )
    impl.q_proj = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(impl.q_lora_rank, q_output_dim)),
        weight_scale=torch.ones(q_output_dim),
        _chunk_size=0,
    )
    monkeypatch.setattr("vllm_ascend.attention.mla_v1.torch_npu.npu_format_cast", lambda tensor, _: tensor)

    prepared = impl._prepare_prolog_v3_weights()

    assert prepared["dequant_scale_w_dq"].shape == (1, impl.q_lora_rank)
    assert prepared["dequant_scale_w_dkv_kr"].shape == (1, FUSED_DIM)
    assert prepared["dequant_scale_w_uq_qr"].shape == (1, q_output_dim)


@pytest.mark.parametrize(
    "fused_method,q_method",
    [
        (object.__new__(AscendW8A8FP8DynamicLinearMethod), object.__new__(AscendW8A8FP8DynamicLinearMethod)),
        (object.__new__(AscendW8A8DynamicLinearMethod), None),
        (None, object.__new__(AscendW8A8DynamicLinearMethod)),
    ],
)
def test_unsupported_or_mixed_prolog_quantization_keeps_legacy_path(fused_method, q_method):
    impl = _make_impl()
    impl.fused_qkv_a_proj = SimpleNamespace(quant_method=SimpleNamespace(quant_method=fused_method))
    impl.q_proj = SimpleNamespace(quant_method=SimpleNamespace(quant_method=q_method), _chunk_size=0)
    nope, rope = _component_major_cache()

    assert not impl.supports_prolog_v3_quantization()
    assert not impl._cache_requires_prolog_v3_writer((nope, rope))


def test_mla_preprocess_prolog_v3_replaces_rmsnorm_rope_cache_writer():
    impl = _make_impl()
    total_tokens = 5
    qkv_lora = torch.zeros(total_tokens, impl.q_lora_rank + FUSED_DIM)
    impl.fused_qkv_a_proj = MagicMock(return_value=(qkv_lora,))
    impl.q_a_layernorm = MagicMock(side_effect=lambda tensor: tensor)
    impl.q_proj = MagicMock(
        side_effect=lambda tensor: (torch.zeros(tensor.shape[0], impl.num_heads * (impl.qk_nope_head_dim + ROPE_DIM)),)
    )
    impl.rope_single = MagicMock(side_effect=lambda tensor, _cos, _sin: tensor)
    nope, rope = _component_major_cache()
    nope.fill_(1.0)
    rope.fill_(2.0)

    num_decode = 2
    num_prefill = 3
    hidden = torch.randn(num_decode + num_prefill, 128)
    decode_cos = torch.randn(num_decode, ROPE_DIM)
    decode_sin = torch.randn(num_decode, ROPE_DIM)
    prefill_cos = torch.randn(num_prefill, ROPE_DIM)
    prefill_sin = torch.randn(num_prefill, ROPE_DIM)
    metadata = SimpleNamespace(
        num_decode_tokens=num_decode,
        num_actual_tokens=num_decode + num_prefill,
        slot_mapping=torch.tensor([0, 1, 129, 130, 131]),
        decode=SimpleNamespace(cos=decode_cos, sin=decode_sin),
        prefill=SimpleNamespace(cos=prefill_cos, sin=prefill_sin),
    )

    def fake_prolog(**kwargs):
        tokens = kwargs["token_x"].shape[0]
        return (
            torch.zeros(tokens, impl.num_heads * NOPE_DIM),
            torch.zeros(tokens, impl.num_heads * ROPE_DIM),
            None,
            None,
            None,
        )

    projected = torch.zeros(num_prefill, impl.num_heads * (impl.qk_nope_head_dim + impl.v_head_dim))
    impl.kv_b_proj.return_value = (projected,)
    impl._prepare_prolog_v3_weights = lambda: impl._prolog_v3_weights

    with (
        patch("torch_npu.npu_mla_prolog_v3", side_effect=fake_prolog) as mock_prolog,
        patch("torch_npu.npu_kv_rmsnorm_rope_cache") as mock_legacy,
    ):
        decode, prefill = impl.mla_preprocess_prolog_v3(hidden, (nope, rope), metadata)

    assert mock_prolog.call_count == 2
    mock_legacy.assert_not_called()
    assert decode is not None
    assert prefill is not None
    assert prefill.q_nope.shape == (num_prefill, impl.num_heads, impl.qk_nope_head_dim)
    assert prefill.q_pe.shape == (num_prefill, impl.num_heads, ROPE_DIM)
    assert prefill.k_pe.shape == (num_prefill, impl.num_heads, ROPE_DIM)
    assert torch.all(prefill.k_pe == 2.0)


def test_forward_routes_strided_cache_to_prolog_v3_preprocessing():
    impl = _make_impl()
    impl.use_output_gate = False
    impl.enable_mlapo = False
    impl.pcp_enabled = False
    impl.is_pcp_decode_sharded = False
    impl._decode_requires_current_kv = MagicMock(return_value=False)
    impl._cache_requires_prolog_v3_writer = MagicMock(return_value=True)
    impl.mla_preprocess_prolog_v3 = MagicMock(return_value=(None, None))
    impl._mla_preprocess = MagicMock()
    impl.o_proj = MagicMock(side_effect=lambda tensor, **_kwargs: (tensor,))

    hidden = torch.zeros(3, 128)
    output = torch.empty(3, impl.num_heads * impl.v_head_dim)
    cache = _component_major_cache()
    metadata = SimpleNamespace(
        num_actual_tokens=3,
        num_decodes=1,
        num_prefills=2,
        num_decode_tokens=1,
    )

    with (
        patch.object(mla_v1, "_EXTRA_CTX", SimpleNamespace(num_tokens=3)),
        patch.object(mla_v1, "maybe_save_kv_layer_to_connector"),
    ):
        assert impl.forward("layer", hidden, cache, metadata, output) is output

    impl._cache_requires_prolog_v3_writer.assert_called_once_with(cache)
    impl.mla_preprocess_prolog_v3.assert_called_once_with(hidden, cache, metadata)
    impl._mla_preprocess.assert_not_called()


def test_mla_preprocess_prolog_v3_keeps_prefill_query_unquantized_for_fa_quant():
    impl = _make_impl()
    total_tokens = 3
    qkv_lora = torch.zeros(total_tokens, impl.q_lora_rank + FUSED_DIM)
    impl.fused_qkv_a_proj = MagicMock(return_value=(qkv_lora,))
    impl.q_a_layernorm = MagicMock(side_effect=lambda tensor: tensor)
    impl.q_proj = MagicMock(
        side_effect=lambda tensor: (torch.zeros(tensor.shape[0], impl.num_heads * (impl.qk_nope_head_dim + ROPE_DIM)),)
    )
    impl.rope_single = MagicMock(side_effect=lambda tensor, _cos, _sin: tensor)
    impl.fa_quant_layer = True
    impl.support_fp8_attention = True
    impl.quant_kscale = torch.tensor([[2.0]])
    impl.fak_descale_reciprocal = torch.tensor(4.0)
    impl.fak_descale_float = torch.tensor(0.25)
    nope, rope = _token_fused_cache()

    num_decode = 1
    num_prefill = 2
    hidden = torch.randn(num_decode + num_prefill, 128)
    metadata = SimpleNamespace(
        num_decode_tokens=num_decode,
        num_actual_tokens=num_decode + num_prefill,
        slot_mapping=torch.tensor([0, 128, 129]),
        decode=SimpleNamespace(
            cos=torch.randn(num_decode, ROPE_DIM),
            sin=torch.randn(num_decode, ROPE_DIM),
        ),
        prefill=SimpleNamespace(
            cos=torch.randn(num_prefill, ROPE_DIM),
            sin=torch.randn(num_prefill, ROPE_DIM),
        ),
    )
    impl.kv_b_proj.return_value = (
        torch.zeros(num_prefill, impl.num_heads * (impl.qk_nope_head_dim + impl.v_head_dim)),
    )
    impl._prepare_prolog_v3_weights = lambda: impl._prolog_v3_weights

    def fake_prolog(**kwargs):
        tokens = kwargs["token_x"].shape[0]
        return (
            torch.zeros(tokens, impl.num_heads * NOPE_DIM),
            torch.zeros(tokens, impl.num_heads * ROPE_DIM),
            torch.ones(tokens, impl.num_heads),
            None,
            None,
        )

    with patch("torch_npu.npu_mla_prolog_v3", side_effect=fake_prolog) as mock_prolog:
        impl.mla_preprocess_prolog_v3(hidden, (nope, rope), metadata)

    assert mock_prolog.call_count == 2
    decode_kwargs = mock_prolog.call_args_list[0].kwargs
    prefill_kwargs = mock_prolog.call_args_list[1].kwargs
    assert decode_kwargs["query_quant_mode"] == 1
    assert prefill_kwargs["query_quant_mode"] == 0
    assert decode_kwargs["kv_cache_quant_mode"] == prefill_kwargs["kv_cache_quant_mode"] == 1
    assert decode_kwargs["cache_mode"] == prefill_kwargs["cache_mode"] == "PA_BSND"
    assert decode_kwargs["tile_size"] == prefill_kwargs["tile_size"] == BLOCK_SIZE
    assert torch.equal(decode_kwargs["quant_scale_ckv"], impl.fak_descale_reciprocal)
