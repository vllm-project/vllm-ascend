# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.quantization.methods import AscendW8A8LinearMethod
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD


def _static_projection(input_size, output_size):
    scheme = AscendW8A8LinearMethod.__new__(AscendW8A8LinearMethod)
    scheme.quant_method = ASCEND_QUANTIZATION_METHOD
    return SimpleNamespace(
        quant_method=SimpleNamespace(quant_method=scheme),
        weight=torch.arange(input_size * output_size).reshape(input_size, output_size).to(torch.int8),
        deq_scale=torch.linspace(0.001, 0.005, output_size),
        quant_bias=torch.arange(output_size, dtype=torch.int32) - output_size // 2,
        input_scale=torch.tensor([0.25], dtype=torch.bfloat16),
        input_offset=torch.tensor([-2], dtype=torch.int8),
    )


@pytest.fixture
def static_mla():
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.enable_mlapo = True
    impl.fa_quant_layer = False
    impl.support_fp8_attention = False
    impl._static_mlapo_weights = None
    impl.q_lora_rank = 32
    impl.kv_lora_rank = 512
    impl.qk_nope_head_dim = 128
    impl.qk_rope_head_dim = 64
    impl.qk_head_dim = 192
    impl.v_head_dim = 128
    impl.num_heads = 2
    impl.num_heads_padded = 2
    impl.mlapo_num_heads = 2
    impl.num_kv_heads = 1
    impl.head_padding = 0
    impl.enable_kv_nz = False
    impl.dtype = torch.bfloat16
    impl.use_mla_rope = True
    impl.use_output_gate = False
    impl.is_pcp_decode_sharded = False
    impl.pcp_enabled = False
    impl.layerwise_kv_cache_hook = None
    impl.layer_name = "static_mla"
    impl.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=False),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        lora_config=None,
    )
    impl.fused_qkv_a_proj = _static_projection(32, 32 + 512 + 64)
    impl.q_proj = _static_projection(32, 2 * 192)
    impl.kv_b_proj = SimpleNamespace(
        weight=torch.randn(2 * (128 + 128), 512, dtype=torch.bfloat16),
        quant_method=UnquantizedLinearMethod(),
    )
    impl.q_a_layernorm = SimpleNamespace(
        weight=torch.ones(32, dtype=torch.bfloat16),
        bias=torch.full((32,), 0.5, dtype=torch.bfloat16),
        bias_loaded=True,
        variance_epsilon=1e-6,
    )
    impl.kv_a_layernorm = SimpleNamespace(
        weight=torch.ones(512, dtype=torch.bfloat16),
        bias=torch.zeros(512, dtype=torch.bfloat16),
        bias_loaded=True,
        variance_epsilon=1e-6,
    )
    ascend_config = SimpleNamespace(rl_config=SimpleNamespace(enabled=False))
    with (
        patch(
            "vllm_ascend.attention.mla_v1.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType.A3),
        ),
        patch("vllm_ascend.attention.mla_v1.get_ascend_config", return_value=ascend_config),
        patch("vllm_ascend.attention.mla_v1.enable_custom_op", return_value=True),
        patch("torch.ops._C_ascend.mla_preprocess", create=True) as kernel,
        patch("torch_npu.npu_format_cast", side_effect=lambda weight, fmt: weight),
        patch("vllm_ascend.attention.mla_v1.maybe_trans_nz", side_effect=lambda weight: weight),
    ):
        yield impl, ascend_config, kernel


def _decode_inputs(impl, num_tokens=8):
    hidden = torch.randn(num_tokens, 32, dtype=torch.bfloat16)
    num_blocks = max(2, (num_tokens + 127) // 128)
    caches = (
        torch.zeros(num_blocks, 128, 1, impl.kv_lora_rank, dtype=torch.bfloat16),
        torch.zeros(num_blocks, 128, 1, impl.qk_rope_head_dim, dtype=torch.bfloat16),
    )
    metadata = SimpleNamespace(
        num_decodes=num_tokens,
        num_decode_tokens=num_tokens,
        num_actual_tokens=num_tokens,
        num_prefills=0,
        slot_mapping=torch.arange(num_tokens, dtype=torch.int32),
        decode=SimpleNamespace(
            cos=torch.ones(num_tokens, 1, 64, dtype=torch.bfloat16),
            sin=torch.zeros(num_tokens, 1, 64, dtype=torch.bfloat16),
        ),
    )
    return hidden, caches, metadata


def test_static_mlapo_load_keeps_sources_and_passes_compensation(static_mla):
    impl, _, kernel = static_mla
    sources = [
        (projection, name, getattr(projection, name), getattr(projection, name).clone())
        for projection in (impl.fused_qkv_a_proj, impl.q_proj)
        for name in ("weight", "deq_scale", "quant_bias")
    ]
    with patch.object(impl, "_process_weights_for_fused") as dynamic:
        impl.process_weights_after_loading(torch.bfloat16)
    dynamic.assert_not_called()
    assert impl.enable_mlapo
    assert impl._static_mlapo_weights is not None
    for projection, name, parameter, value in sources:
        assert getattr(projection, name) is parameter
        torch.testing.assert_close(parameter, value)

    hidden, caches, metadata = _decode_inputs(impl)
    assert impl._can_use_static_mlapo(hidden, caches, metadata)
    with patch("vllm_ascend.attention.mla_v1.notify_kv_cache_written") as notify:
        result, prefill = impl.mla_preprocess_only_decode(hidden, caches, metadata)
    notify.assert_called_once_with(impl.layer_name)
    assert prefill is None
    assert result.k_nope is caches[0] and result.k_pe is caches[1]
    assert result.ql_nope.shape == (8, 2, 512)
    args, kwargs = kernel.call_args
    torch.testing.assert_close(args[4], impl.q_a_layernorm.bias)
    torch.testing.assert_close(kwargs["bias0"], impl._static_mlapo_weights.quant_bias_qkv)
    torch.testing.assert_close(kwargs["bias1"], impl._static_mlapo_weights.qb_qt_bias)
    torch.testing.assert_close(kwargs["quant_scale0"], impl.fused_qkv_a_proj.input_scale)
    torch.testing.assert_close(kwargs["quant_offset1"], impl.q_proj.input_offset)
    assert kwargs["quant_mode"] == "per_tensor_quant_asymm"
    assert kwargs["cache_mode"] == "krope_ctkv"
    assert kwargs["enable_inner_out"] is False
    assert kwargs["inner_out"].shape == (8, impl.q_lora_rank)


def test_static_mlapo_unloaded_q_bias_uses_zero_tensor(static_mla):
    impl, _, _ = static_mla
    impl.q_a_layernorm.bias_loaded = False
    impl.q_a_layernorm.bias.fill_(float("nan"))
    impl.process_weights_after_loading(torch.bfloat16)
    torch.testing.assert_close(impl._static_mlapo_q_beta, torch.zeros_like(impl.q_a_layernorm.weight))


def test_static_mlapo_respects_disabled_configuration(static_mla):
    impl, _, _ = static_mla
    impl.enable_mlapo = False
    with patch.object(impl, "_process_weights_for_static_mlapo") as static:
        impl.process_weights_after_loading(torch.bfloat16)
    static.assert_not_called()
    assert impl._static_mlapo_weights is None


@pytest.mark.parametrize(
    "unsupported",
    [
        "sleep",
        "reload",
        "lora",
        "pcp",
        "dcp",
        "nz",
        "no_rope",
        "kv_beta",
        "epsilon",
        "q_beta_dtype",
        "compressed",
        "q_up_unquantized",
        "scale_zero",
        "misaligned",
        "fp16",
        "hardware",
        "missing_op",
    ],
)
def test_static_mlapo_unsupported_layers_keep_native_weights(static_mla, unsupported):
    impl, ascend_config, _ = static_mla
    dtype = torch.bfloat16
    if unsupported == "sleep":
        impl.vllm_config.model_config.enable_sleep_mode = True
    elif unsupported == "reload":
        ascend_config.rl_config.enabled = True
    elif unsupported == "lora":
        impl.vllm_config.lora_config = object()
    elif unsupported in ("pcp", "dcp"):
        attr = "prefill_context_parallel_size" if unsupported == "pcp" else "decode_context_parallel_size"
        setattr(impl.vllm_config.parallel_config, attr, 2)
    elif unsupported == "nz":
        impl.enable_kv_nz = True
    elif unsupported == "no_rope":
        impl.use_mla_rope = False
    elif unsupported == "kv_beta":
        impl.kv_a_layernorm.bias[0] = 0.25
    elif unsupported == "epsilon":
        impl.q_a_layernorm.variance_epsilon = 1e-5
    elif unsupported == "q_beta_dtype":
        impl.q_a_layernorm.bias = impl.q_a_layernorm.bias.float()
    elif unsupported == "compressed":
        impl.fused_qkv_a_proj.quant_method.quant_method.quant_method = COMPRESSED_TENSORS_METHOD
    elif unsupported == "q_up_unquantized":
        impl.q_proj.quant_method = UnquantizedLinearMethod()
    elif unsupported == "scale_zero":
        impl.q_proj.input_scale.zero_()
    elif unsupported == "misaligned":
        impl.q_lora_rank = 31
    elif unsupported == "fp16":
        dtype = torch.float16
    with (
        patch.object(impl, "_process_weights_for_fused") as dynamic,
        patch("vllm_ascend.attention.mla_v1.enable_custom_op", return_value=unsupported != "missing_op"),
        patch(
            "vllm_ascend.attention.mla_v1.get_current_hardware_profile",
            return_value=get_hardware_profile(
                AscendDeviceType.A5 if unsupported == "hardware" else AscendDeviceType.A3
            ),
        ),
    ):
        impl.process_weights_after_loading(dtype)
    assert impl.enable_mlapo is False
    assert impl._static_mlapo_weights is None
    assert impl.fused_qkv_a_proj.weight is not None and impl.q_proj.weight is not None
    dynamic.assert_not_called()


@pytest.mark.parametrize(
    "batch",
    [
        "decode",
        "one_decode",
        "max_decode",
        "mixed",
        "empty",
        "padded_overflow",
        "strided_slot",
        "fp16_cache",
        "strided_cache",
        "inner_strided_cache",
        "overlapping_cache",
    ],
)
def test_static_mlapo_runtime_falls_back_for_unsupported_batches(static_mla, batch):
    impl, _, _ = static_mla
    impl.process_weights_after_loading(torch.bfloat16)
    num_tokens = 1024 if batch == "max_decode" else (1 if batch == "one_decode" else 8)
    hidden, caches, metadata = _decode_inputs(impl, num_tokens)
    expected = "static"
    if batch == "mixed":
        metadata.num_prefills = 1
        expected = "native"
    elif batch == "empty":
        metadata.num_decode_tokens = metadata.num_decodes = metadata.num_actual_tokens = 0
        expected = "native"
    elif batch == "padded_overflow":
        hidden = torch.zeros(1025, 32, dtype=torch.bfloat16)
        expected = "native"
    elif batch == "strided_slot":
        metadata.slot_mapping = torch.arange(16, dtype=torch.int32)[::2]
        expected = "native"
    elif batch == "fp16_cache":
        caches = tuple(cache.half() for cache in caches)
        expected = "native"
    elif batch == "strided_cache":
        # Row-strided blocks are legal; only inner dimensions must be dense.
        caches = tuple(torch.zeros(4, *cache.shape[1:], dtype=cache.dtype)[::2] for cache in caches)
    elif batch == "inner_strided_cache":
        caches = tuple(
            torch.zeros(*cache.shape[:-1], cache.shape[-1] * 2, dtype=cache.dtype)[..., ::2] for cache in caches
        )
        expected = "native"
    elif batch == "overlapping_cache":
        caches = tuple(cache[:1].expand_as(cache) for cache in caches)
        expected = "native"
    output = torch.empty(hidden.shape[0], 32, dtype=hidden.dtype)
    with (
        patch("vllm_ascend.attention.mla_v1._EXTRA_CTX", SimpleNamespace(num_tokens=hidden.shape[0])),
        patch.object(impl, "_decode_requires_current_kv", return_value=False),
        patch.object(impl, "_mla_preprocess", side_effect=RuntimeError("native")),
        patch.object(impl, "mla_preprocess_static_only_decode", side_effect=RuntimeError("static")),
        pytest.raises(RuntimeError, match=f"^{expected}$"),
    ):
        impl.forward(impl.layer_name, hidden, caches, metadata, output)
