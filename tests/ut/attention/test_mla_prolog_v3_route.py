# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routing tests for PROLOG_V3 on strided MLA caches."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu

import vllm_ascend.attention.mla_v1 as mla_v1
from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPImpl
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
    impl.use_flashmla = False
    impl.use_mla_rope = True
    impl.qk_rope_head_dim = ROPE_DIM
    impl.dtype = torch.bfloat16
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


def _construct_impl(pcp_size: int, dcp_size: int) -> AscendMLAImpl:
    """Construct the production MLA implementation without weight processing."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(runner_type="generate", dtype=torch.float16),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        speculative_config=None,
    )
    with (
        patch.object(mla_v1, "get_current_vllm_config", return_value=config),
        patch.object(
            mla_v1,
            "get_current_hardware_profile",
            return_value=SimpleNamespace(supports=lambda _capability: False),
        ),
        patch.object(mla_v1, "get_ascend_config", return_value=SimpleNamespace(enable_kv_nz=False)),
        patch.object(mla_v1, "enabling_mlapo", return_value=False),
        patch.object(mla_v1, "enable_fa_quant", return_value=False),
    ):
        return AscendMLAImpl(
            num_heads=8,
            head_size=FUSED_DIM,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type="decoded_attention",
            kv_sharing_target_layer_name=None,
            q_lora_rank=128,
            kv_lora_rank=NOPE_DIM,
            qk_nope_head_dim=128,
            qk_rope_head_dim=ROPE_DIM,
            qk_head_dim=128 + ROPE_DIM,
            v_head_dim=128,
            rotary_emb=None,
            fused_qkv_a_proj=MagicMock(),
            q_b_proj=MagicMock(),
            kv_b_proj=MagicMock(),
            o_proj=None,
            use_mla_rope=True,
        )


@pytest.mark.parametrize(
    ("pcp_size", "dcp_size", "expected_pcp", "expected_dcp"),
    [
        (1, 1, False, False),
        (1, 2, False, True),
        (2, 1, True, False),
        (2, 2, True, True),
    ],
)
def test_constructor_initializes_parallel_routing_flags(
    pcp_size,
    dcp_size,
    expected_pcp,
    expected_dcp,
):
    impl = _construct_impl(pcp_size, dcp_size)

    assert impl.pcp_enabled is expected_pcp
    assert impl.dcp_enabled is expected_dcp
    assert impl.is_pcp_decode_sharded is (expected_pcp and dcp_size == 1)


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


def test_mla_without_q_lora_keeps_legacy_writer():
    impl = _make_impl()
    impl.q_lora_rank = None
    impl.fused_qkv_a_proj = None
    impl.q_a_layernorm = None
    nope, rope = _component_major_cache()

    assert not impl.supports_prolog_v3_projection()
    assert not impl._cache_requires_prolog_v3_writer((nope, rope))


def test_prolog_v3_writer_requires_rope_and_operator(monkeypatch):
    impl = _make_impl()
    nope, rope = _component_major_cache()

    assert impl.supports_prolog_v3_writer()
    assert impl._cache_requires_prolog_v3_writer((nope, rope))

    impl.use_mla_rope = False
    assert not impl.supports_prolog_v3_writer()
    assert not impl._cache_requires_prolog_v3_writer((nope, rope))
    impl.use_mla_rope = True

    monkeypatch.delattr(torch_npu, "npu_mla_prolog_v3")
    assert not impl.supports_prolog_v3_writer()
    assert not impl._cache_requires_prolog_v3_writer((nope, rope))


def test_prolog_v3_writer_excludes_native_fp16():
    impl = _make_impl()
    nope, rope = _component_major_cache()
    impl.dtype = torch.float16

    assert not impl.supports_prolog_v3_dtype()
    assert not impl.supports_prolog_v3_writer()
    assert not impl._cache_requires_prolog_v3_writer((nope, rope))


def test_dcp_current_kv_capacity_uses_decode_query_len():
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=32, max_num_batched_tokens=16384),
    )
    impl.speculative_config = SimpleNamespace(num_speculative_tokens=3)

    assert impl._required_dcp_current_kv_capacity() == 128

    impl.vllm_config.scheduler_config.max_num_batched_tokens = 96
    assert impl._required_dcp_current_kv_capacity() == 96


def test_dcp_prefill_current_kv_capacity_uses_batched_tokens():
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=32, max_num_batched_tokens=4096),
    )

    assert impl._required_dcp_prefill_current_kv_capacity() == 4096


@pytest.mark.parametrize(
    ("dcp_enabled", "pcp_enabled", "decode_sharded", "use_rope", "expected_buffers"),
    [
        (False, False, False, True, "none"),
        (True, False, False, True, "both"),
        (True, True, False, True, "none"),
        (True, False, True, True, "none"),
        (False, False, False, False, "none"),
        (True, False, False, False, "decode"),
        (True, True, False, False, "decode"),
    ],
)
def test_dcp_post_load_allocates_only_consumed_workspaces(
    dcp_enabled, pcp_enabled, decode_sharded, use_rope, expected_buffers
):
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=32),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
    )
    impl.speculative_config = SimpleNamespace(num_speculative_tokens=7)
    impl.dcp_enabled = dcp_enabled
    impl.pcp_enabled = pcp_enabled
    impl.is_pcp_decode_sharded = decode_sharded
    impl.enable_mlapo = True
    impl.fa_quant_layer = False
    impl.use_mla_rope = use_rope
    impl.enable_kv_nz = False
    impl.num_kv_heads = 1
    impl.kv_lora_rank = NOPE_DIM
    impl.qk_rope_head_dim = ROPE_DIM
    impl.dtype = torch.bfloat16
    impl.fused_qkv_a_proj = SimpleNamespace(weight=torch.empty(1))
    impl._dcp_current_kv_buffers = None
    impl._dcp_prefill_current_kv_buffers = None
    impl.supports_prolog_v3_writer = lambda: use_rope

    with patch.object(AscendMLAImpl, "process_weights_after_loading"):
        impl.process_weights_after_loading(torch.bfloat16)

    assert (impl._dcp_current_kv_buffers is not None) is (expected_buffers != "none")
    assert (impl._dcp_prefill_current_kv_buffers is not None) is (expected_buffers == "both")
    if expected_buffers != "none":
        assert impl._dcp_current_kv_buffers is not None
        assert impl._dcp_current_kv_buffers[2].numel() == 256
    if expected_buffers == "both":
        assert impl._dcp_prefill_current_kv_buffers is not None
        assert impl._dcp_prefill_current_kv_buffers[2].numel() == 8192


def test_dcp_current_kv_buffers_survive_reloads():
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256, max_num_seqs=256),
        cache_config=SimpleNamespace(block_size=128),
    )
    impl.speculative_config = None
    impl.num_kv_heads = 1
    impl.kv_lora_rank = NOPE_DIM
    impl.qk_rope_head_dim = ROPE_DIM
    impl.dtype = torch.float32
    impl.fused_qkv_a_proj = SimpleNamespace(weight=torch.empty(1))
    impl._dcp_current_kv_buffers = None
    impl._dcp_prefill_current_kv_buffers = None
    impl.supports_prolog_v3_writer = lambda: False

    impl._prepare_dcp_current_kv_buffers()
    first_buffers = impl._dcp_current_kv_buffers
    assert first_buffers is not None
    assert first_buffers[0].shape == (2, 128, 1, NOPE_DIM)
    assert first_buffers[1].shape == (2, 128, 1, ROPE_DIM)
    assert first_buffers[2].tolist() == list(range(256))

    impl._prepare_dcp_current_kv_buffers()
    assert impl._dcp_current_kv_buffers is first_buffers
    assert all(
        first.data_ptr() == second.data_ptr()
        for first, second in zip(first_buffers, impl._dcp_current_kv_buffers, strict=True)
    )


def test_dcp_prolog_buffers_survive_reloads_and_use_disjoint_workspaces():
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096, max_num_seqs=32),
        cache_config=SimpleNamespace(block_size=128),
    )
    impl.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    impl.num_kv_heads = 1
    impl.kv_lora_rank = NOPE_DIM
    impl.qk_rope_head_dim = ROPE_DIM
    impl.dtype = torch.float32
    impl.fused_qkv_a_proj = SimpleNamespace(weight=torch.empty(1))
    impl._dcp_current_kv_buffers = None
    impl._dcp_prefill_current_kv_buffers = None
    impl.supports_prolog_v3_writer = lambda: True

    impl._prepare_dcp_current_kv_buffers()
    first_decode_buffers = impl._dcp_current_kv_buffers
    first_prefill_buffers = impl._dcp_prefill_current_kv_buffers
    assert first_decode_buffers is not None
    assert first_prefill_buffers is not None
    assert first_decode_buffers[0].shape == (1, 128, 1, NOPE_DIM)
    assert first_prefill_buffers[0].shape == (32, 128, 1, NOPE_DIM)
    assert first_decode_buffers[0].untyped_storage() is not first_prefill_buffers[0].untyped_storage()
    assert impl._dcp_current_kv_buffers_are_sufficient()

    impl._prepare_dcp_current_kv_buffers()
    assert impl._dcp_current_kv_buffers is first_decode_buffers
    assert impl._dcp_prefill_current_kv_buffers is first_prefill_buffers


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
        external_flashmla=None,
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


@pytest.mark.parametrize(
    (
        "dcp_enabled",
        "has_dcp_buffers",
        "pcp_enabled",
        "is_pcp_decode_sharded",
        "requires_current_kv",
        "expects_prolog",
    ),
    [
        (False, False, False, False, False, True),
        (True, True, False, False, False, True),
        (True, True, False, False, True, True),
        (True, False, False, False, False, False),
        (False, False, False, False, True, False),
        (False, False, True, False, False, False),
        (False, False, False, True, False, False),
    ],
)
def test_forward_routes_strided_cache_with_runtime_exclusions(
    dcp_enabled,
    has_dcp_buffers,
    pcp_enabled,
    is_pcp_decode_sharded,
    requires_current_kv,
    expects_prolog,
):
    impl = _make_impl()
    impl.use_output_gate = False
    impl.enable_mlapo = False
    impl.dcp_enabled = dcp_enabled
    if has_dcp_buffers:
        impl._dcp_current_kv_buffers = (
            torch.empty(1, BLOCK_SIZE, 1, NOPE_DIM),
            torch.empty(1, BLOCK_SIZE, 1, ROPE_DIM),
            torch.arange(BLOCK_SIZE),
        )
    impl.pcp_enabled = pcp_enabled
    impl.is_pcp_decode_sharded = is_pcp_decode_sharded
    impl._decode_requires_current_kv = MagicMock(return_value=requires_current_kv)
    impl._cache_requires_prolog_v3_writer = MagicMock(return_value=True)
    impl.mla_preprocess_prolog_v3 = MagicMock(return_value=(None, None))
    impl._mla_preprocess = MagicMock(return_value=(None, None))
    impl.o_proj = MagicMock(side_effect=lambda tensor, **_kwargs: (tensor,))

    hidden = torch.zeros(3, 128)
    output = torch.empty(3, impl.num_heads * impl.v_head_dim)
    cache = _component_major_cache()
    metadata = SimpleNamespace(
        external_flashmla=None,
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
    if expects_prolog:
        impl.mla_preprocess_prolog_v3.assert_called_once_with(hidden, cache, metadata)
        impl._mla_preprocess.assert_not_called()
    else:
        impl.mla_preprocess_prolog_v3.assert_not_called()
        impl._mla_preprocess.assert_called_once()


def test_forward_falls_back_when_dcp_current_buffer_is_too_small():
    impl = _make_impl()
    impl.use_output_gate = False
    impl.enable_mlapo = False
    impl.dcp_enabled = True
    impl.pcp_enabled = False
    impl.is_pcp_decode_sharded = False
    impl._dcp_current_kv_buffers = (
        torch.empty(1, 1, 1, NOPE_DIM),
        torch.empty(1, 1, 1, ROPE_DIM),
        torch.arange(1),
    )
    impl._decode_requires_current_kv = MagicMock(return_value=True)
    impl._cache_requires_prolog_v3_writer = MagicMock(return_value=True)
    impl.mla_preprocess_prolog_v3 = MagicMock(return_value=(None, None))
    impl._mla_preprocess = MagicMock(return_value=(None, None))
    impl.o_proj = MagicMock(side_effect=lambda tensor, **_kwargs: (tensor,))

    hidden = torch.zeros(2, 128)
    output = torch.empty(2, impl.num_heads * impl.v_head_dim)
    metadata = SimpleNamespace(
        external_flashmla=None,
        num_actual_tokens=2,
        num_decodes=2,
        num_prefills=0,
        num_decode_tokens=2,
    )

    with (
        patch.object(mla_v1, "_EXTRA_CTX", SimpleNamespace(num_tokens=2)),
        patch.object(mla_v1, "maybe_save_kv_layer_to_connector"),
    ):
        assert impl.forward("layer", hidden, _component_major_cache(), metadata, output) is output

    impl.mla_preprocess_prolog_v3.assert_not_called()
    impl._mla_preprocess.assert_called_once()


def test_forward_falls_back_when_dcp_prefill_current_buffer_is_too_small():
    impl = _make_impl()
    impl.use_output_gate = False
    impl.enable_mlapo = False
    impl.dcp_enabled = True
    impl.pcp_enabled = False
    impl.is_pcp_decode_sharded = False
    impl._dcp_current_kv_buffers = (
        torch.empty(1, 1, 1, NOPE_DIM),
        torch.empty(1, 1, 1, ROPE_DIM),
        torch.arange(1),
    )
    impl._dcp_prefill_current_kv_buffers = (
        torch.empty(1, 1, 1, NOPE_DIM),
        torch.empty(1, 1, 1, ROPE_DIM),
        torch.arange(1),
    )
    impl._decode_requires_current_kv = MagicMock(return_value=True)
    impl._prefill_requires_current_kv = MagicMock(return_value=True)
    impl._cache_requires_prolog_v3_writer = MagicMock(return_value=True)
    impl.mla_preprocess_prolog_v3 = MagicMock(return_value=(None, None))
    impl._mla_preprocess = MagicMock(return_value=(None, None))
    impl.o_proj = MagicMock(side_effect=lambda tensor, **_kwargs: (tensor,))

    hidden = torch.zeros(2, 128)
    output = torch.empty(2, impl.num_heads * impl.v_head_dim)
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        num_decodes=0,
        num_prefills=1,
        num_decode_tokens=0,
    )

    with (
        patch.object(mla_v1, "_EXTRA_CTX", SimpleNamespace(num_tokens=2)),
        patch.object(mla_v1, "maybe_save_kv_layer_to_connector"),
    ):
        assert impl.forward("layer", hidden, _component_major_cache(), metadata, output) is output

    impl.mla_preprocess_prolog_v3.assert_not_called()
    impl._mla_preprocess.assert_called_once()


def test_mla_preprocess_prolog_v3_preserves_dcp_prefill_current_kv():
    impl = _make_impl()
    dcp_impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    dcp_impl.__dict__.update(impl.__dict__)
    impl = dcp_impl
    impl.dcp_enabled = True
    impl.speculative_config = None
    num_prefill_tokens = 4
    current_nope_cache = torch.zeros(1, BLOCK_SIZE, 1, NOPE_DIM)
    current_pe_cache = torch.zeros(1, BLOCK_SIZE, 1, ROPE_DIM)
    current_indices = torch.arange(BLOCK_SIZE)
    impl._dcp_current_kv_buffers = (
        torch.empty(1, BLOCK_SIZE, 1, NOPE_DIM),
        torch.empty(1, BLOCK_SIZE, 1, ROPE_DIM),
        torch.arange(BLOCK_SIZE),
    )
    impl._dcp_prefill_current_kv_buffers = (current_nope_cache, current_pe_cache, current_indices)
    impl._prepare_prolog_v3_weights = lambda: impl._prolog_v3_weights

    qkv_lora = torch.zeros(num_prefill_tokens, impl.q_lora_rank + FUSED_DIM)
    impl.fused_qkv_a_proj = MagicMock(return_value=(qkv_lora,))
    impl.q_a_layernorm = MagicMock(side_effect=lambda tensor: tensor)
    impl.q_proj = MagicMock(
        side_effect=lambda tensor: (torch.zeros(tensor.shape[0], impl.num_heads * (impl.qk_nope_head_dim + ROPE_DIM)),)
    )
    impl.rope_single = MagicMock(side_effect=lambda tensor, _cos, _sin: tensor)
    impl.kv_b_proj = MagicMock(
        side_effect=lambda tensor: (
            torch.zeros(tensor.shape[0], impl.num_heads * (impl.qk_nope_head_dim + impl.v_head_dim)),
        )
    )

    nope, rope = _component_major_cache()
    hidden = torch.randn(num_prefill_tokens, 128)
    # Slots -1 are real current rows owned by another DCP rank, not padding.
    prefill_slots = torch.tensor([-1, BLOCK_SIZE, -1, BLOCK_SIZE + 1])
    metadata = SimpleNamespace(
        external_flashmla=None,
        causal=True,
        decode=None,
        num_decode_tokens=0,
        num_actual_tokens=num_prefill_tokens,
        slot_mapping=prefill_slots,
        prefill=SimpleNamespace(
            cos=torch.randn(num_prefill_tokens, ROPE_DIM),
            sin=torch.randn(num_prefill_tokens, ROPE_DIM),
        ),
    )

    expected_nope = torch.arange(num_prefill_tokens, dtype=torch.float32) + 3.0
    expected_pe = torch.arange(num_prefill_tokens, dtype=torch.float32) + 4.0

    def fake_prolog(**kwargs):
        kwargs["kv_cache"][0, :num_prefill_tokens, 0, 0] = expected_nope
        kwargs["kr_cache"][0, :num_prefill_tokens, 0, 0] = expected_pe
        return (
            torch.zeros(num_prefill_tokens, impl.num_heads * NOPE_DIM),
            torch.zeros(num_prefill_tokens, impl.num_heads * ROPE_DIM),
            None,
            None,
            None,
        )

    with (
        patch(
            "vllm_ascend.attention.context_parallel.mla_cp._EXTRA_CTX",
            SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False),
        ),
        patch("torch_npu.npu_mla_prolog_v3", side_effect=fake_prolog) as mock_prolog,
        patch("torch_npu.npu_kv_rmsnorm_rope_cache") as mock_legacy,
        patch.object(mla_v1.DeviceOperator, "reshape_and_cache") as mock_scatter,
    ):
        decode, prefill = impl.mla_preprocess_prolog_v3(hidden, (nope, rope), metadata)

    assert decode is None
    assert prefill is not None
    mock_legacy.assert_not_called()

    prolog_kwargs = mock_prolog.call_args.kwargs
    assert prolog_kwargs["kv_cache"].untyped_storage() is current_nope_cache.untyped_storage()
    assert prolog_kwargs["kr_cache"].untyped_storage() is current_pe_cache.untyped_storage()
    assert torch.equal(prolog_kwargs["cache_index"], current_indices[:num_prefill_tokens])

    scatter_kwargs = mock_scatter.call_args.kwargs
    assert scatter_kwargs["key_cache"] is nope
    assert scatter_kwargs["value_cache"] is rope
    assert torch.equal(scatter_kwargs["slot_mapping"], prefill_slots)
    assert scatter_kwargs["key"].untyped_storage() is current_nope_cache.untyped_storage()
    assert scatter_kwargs["value"].untyped_storage() is current_pe_cache.untyped_storage()

    nope_passed_to_kv_b = impl.kv_b_proj.call_args.args[0]
    assert torch.equal(nope_passed_to_kv_b[:, 0], expected_nope)
    assert torch.equal(prefill.k_pe[:, 0, 0], expected_pe)


def test_mla_preprocess_prolog_v3_keeps_mixed_dcp_current_kv_disjoint():
    impl = _make_impl()
    dcp_impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    dcp_impl.__dict__.update(impl.__dict__)
    impl = dcp_impl
    impl.dcp_enabled = True
    impl.speculative_config = object()
    num_decode_tokens = 2
    num_prefill_tokens = 4
    decode_nope_cache = torch.zeros(1, BLOCK_SIZE, 1, NOPE_DIM)
    decode_pe_cache = torch.zeros(1, BLOCK_SIZE, 1, ROPE_DIM)
    prefill_nope_cache = torch.zeros(1, BLOCK_SIZE, 1, NOPE_DIM)
    prefill_pe_cache = torch.zeros(1, BLOCK_SIZE, 1, ROPE_DIM)
    current_indices = torch.arange(BLOCK_SIZE)
    impl._dcp_current_kv_buffers = (decode_nope_cache, decode_pe_cache, current_indices)
    impl._dcp_prefill_current_kv_buffers = (prefill_nope_cache, prefill_pe_cache, current_indices)
    impl._prepare_prolog_v3_weights = lambda: impl._prolog_v3_weights
    impl.reorg_decode_q = MagicMock(side_effect=lambda q_nope, q_pe: (q_nope, q_pe))

    total_tokens = num_decode_tokens + num_prefill_tokens
    qkv_lora = torch.zeros(total_tokens, impl.q_lora_rank + FUSED_DIM)
    impl.fused_qkv_a_proj = MagicMock(return_value=(qkv_lora,))
    impl.q_a_layernorm = MagicMock(side_effect=lambda tensor: tensor)
    impl.q_proj = MagicMock(
        side_effect=lambda tensor: (torch.zeros(tensor.shape[0], impl.num_heads * (impl.qk_nope_head_dim + ROPE_DIM)),)
    )
    impl.rope_single = MagicMock(side_effect=lambda tensor, _cos, _sin: tensor)
    impl.kv_b_proj = MagicMock(
        side_effect=lambda tensor: (
            torch.zeros(tensor.shape[0], impl.num_heads * (impl.qk_nope_head_dim + impl.v_head_dim)),
        )
    )

    nope, rope = _component_major_cache()
    hidden = torch.randn(total_tokens, 128)
    metadata = SimpleNamespace(
        external_flashmla=None,
        causal=True,
        decode=SimpleNamespace(
            cos=torch.randn(num_decode_tokens, ROPE_DIM),
            sin=torch.randn(num_decode_tokens, ROPE_DIM),
            actual_seq_lengths_q=[2, 2],
        ),
        prefill=SimpleNamespace(
            cos=torch.randn(num_prefill_tokens, ROPE_DIM),
            sin=torch.randn(num_prefill_tokens, ROPE_DIM),
        ),
        num_decode_tokens=num_decode_tokens,
        num_actual_tokens=total_tokens,
        slot_mapping=torch.tensor([10, 11, -1, BLOCK_SIZE, -1, BLOCK_SIZE + 1]),
    )

    def fake_prolog(**kwargs):
        is_decode = kwargs["kv_cache"].untyped_storage() is decode_nope_cache.untyped_storage()
        kwargs["kv_cache"].fill_(3.0 if is_decode else 5.0)
        kwargs["kr_cache"].fill_(4.0 if is_decode else 6.0)
        return (
            torch.zeros(kwargs["token_x"].shape[0], impl.num_heads * NOPE_DIM),
            torch.zeros(kwargs["token_x"].shape[0], impl.num_heads * ROPE_DIM),
            None,
            None,
            None,
        )

    with (
        patch(
            "vllm_ascend.attention.context_parallel.mla_cp._EXTRA_CTX",
            SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False),
        ),
        patch.object(
            mla_v1,
            "_EXTRA_CTX",
            SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False),
        ),
        patch("torch_npu.npu_mla_prolog_v3", side_effect=fake_prolog),
        patch("torch_npu.npu_kv_rmsnorm_rope_cache") as mock_legacy,
        patch.object(mla_v1.DeviceOperator, "reshape_and_cache") as mock_scatter,
    ):
        decode, prefill = impl.mla_preprocess_prolog_v3(hidden, (nope, rope), metadata)

    mock_legacy.assert_not_called()
    assert decode is not None
    assert prefill is not None
    assert decode.current_k_nope is not None
    assert decode.current_k_pe is not None
    assert torch.all(decode.current_k_nope == 3.0)
    assert torch.all(decode.current_k_pe == 4.0)
    assert torch.all(prefill.k_pe == 6.0)
    assert len(mock_scatter.call_args_list) == 2
    assert mock_scatter.call_args_list[0].kwargs["key"].untyped_storage() is decode_nope_cache.untyped_storage()
    assert mock_scatter.call_args_list[1].kwargs["key"].untyped_storage() is prefill_nope_cache.untyped_storage()


def test_mla_preprocess_prolog_v3_supports_dcp_current_kv_split(monkeypatch):
    impl = _make_impl()
    num_decode_tokens = 1
    current_nope_cache = torch.zeros(1, BLOCK_SIZE, 1, NOPE_DIM)
    current_pe_cache = torch.zeros(1, BLOCK_SIZE, 1, ROPE_DIM)
    current_indices = torch.arange(BLOCK_SIZE)
    impl._dcp_current_kv_buffers = (current_nope_cache, current_pe_cache, current_indices)
    impl._decode_requires_current_kv = MagicMock(return_value=True)
    impl._prepare_prolog_v3_weights = lambda: impl._prolog_v3_weights
    impl.reorg_decode_q = MagicMock(side_effect=lambda q_nope, q_pe: (q_nope, q_pe))

    nope, rope = _component_major_cache()
    hidden = torch.randn(num_decode_tokens, 128)
    metadata = SimpleNamespace(
        external_flashmla=None,
        num_decode_tokens=num_decode_tokens,
        num_actual_tokens=num_decode_tokens,
        slot_mapping=torch.tensor([10]),
        decode=SimpleNamespace(
            cos=torch.randn(num_decode_tokens, ROPE_DIM),
            sin=torch.randn(num_decode_tokens, ROPE_DIM),
        ),
    )

    def fake_prolog(**kwargs):
        kwargs["kv_cache"].fill_(3.0)
        kwargs["kr_cache"].fill_(4.0)
        return (
            torch.zeros(num_decode_tokens, impl.num_heads * NOPE_DIM),
            torch.zeros(num_decode_tokens, impl.num_heads * ROPE_DIM),
            None,
            None,
            None,
        )

    with (
        patch("torch_npu.npu_mla_prolog_v3", side_effect=fake_prolog) as mock_prolog,
        patch("torch_npu.npu_kv_rmsnorm_rope_cache") as mock_legacy,
        patch.object(mla_v1.DeviceOperator, "reshape_and_cache") as mock_scatter,
    ):
        decode, prefill = impl.mla_preprocess_prolog_v3(hidden, (nope, rope), metadata)

    assert prefill is None
    prolog_kwargs = mock_prolog.call_args.kwargs
    assert prolog_kwargs["kv_cache"].untyped_storage() is current_nope_cache.untyped_storage()
    assert prolog_kwargs["kr_cache"].untyped_storage() is current_pe_cache.untyped_storage()
    assert torch.equal(prolog_kwargs["cache_index"], current_indices[:num_decode_tokens])
    mock_legacy.assert_not_called()
    impl.reorg_decode_q.assert_called_once()

    assert decode is not None
    assert decode.current_k_nope is not None
    assert decode.current_k_pe is not None
    assert torch.all(decode.current_k_nope == 3.0)
    assert torch.all(decode.current_k_pe == 4.0)
    assert mock_scatter.call_args.kwargs["key_cache"] is nope
    assert mock_scatter.call_args.kwargs["value_cache"] is rope
    assert torch.equal(mock_scatter.call_args.kwargs["slot_mapping"], torch.tensor([10]))


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
        external_flashmla=None,
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
    kv_b_proj_input = impl.kv_b_proj.call_args.args[0]
    assert kv_b_proj_input.dtype == nope.dtype
    assert torch.equal(decode_kwargs["quant_scale_ckv"], impl.fak_descale_reciprocal)
