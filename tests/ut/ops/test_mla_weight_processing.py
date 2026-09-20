# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention


def make_attention(impl):
    # Run the installed upstream post-load method, not a mock of its dispatch.
    inner = MLAAttention.__new__(MLAAttention)
    nn.Module.__init__(inner)
    set_default_quant_scales(inner, register_buffer=True)
    inner.impl = impl
    inner.num_heads = 2
    inner.kv_lora_rank = 4
    inner.qk_nope_head_dim = 2
    inner.v_head_dim = 4
    inner.kv_b_proj = nn.Linear(4, 12, bias=False)
    inner.kv_b_proj.quant_method = UnquantizedLinearMethod()
    inner.is_amx_bmm_enabled = False
    inner.dcp_q_replicate = False
    inner.is_aiter_triton_fp4_bmm_enabled = False
    inner.is_aiter_triton_fp8_bmm_enabled = False
    inner.quant_config = None
    upstream = MagicMock(wraps=inner.process_weights_after_loading)
    inner.process_weights_after_loading = upstream
    modules = MagicMock()
    modules.indexer = None
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=1)),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with (
        patch("vllm_ascend.ops.mla.MLAAttention", return_value=inner),
        patch("vllm_ascend.ops.mla.get_current_vllm_config", return_value=config),
        patch("vllm_ascend.ops.mla.get_tensor_model_parallel_world_size", return_value=1),
        patch("vllm_ascend.ops.mla.mark_fused_preprocess_weights"),
    ):
        wrapper = AscendMultiHeadLatentAttention(
            hidden_size=8,
            num_heads=2,
            scale=1.0,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=4,
            q_lora_rank=4,
            kv_lora_rank=4,
            mla_modules=modules,
        )
    return wrapper.mla_attn, upstream


@pytest.mark.parametrize("is_sfa", [False, True])
@pytest.mark.parametrize("act_dtype", [torch.float16, torch.bfloat16])
def test_impl_post_load_called_once(is_sfa, act_dtype):
    impl = MagicMock(spec=AscendSFAImpl if is_sfa else AscendMLAImpl)
    inner, upstream = make_attention(impl)
    if is_sfa:
        # SFA disposes this projection; upstream dense packing must be bypassed.
        impl.process_weights_after_loading.side_effect = lambda _: setattr(inner.kv_b_proj, "weight", None)

    inner.process_weights_after_loading(act_dtype)

    impl.process_weights_after_loading.assert_called_once_with(act_dtype)
    if is_sfa:
        upstream.assert_not_called()
        assert inner.kv_b_proj.weight is None
    else:
        upstream.assert_called_once_with(act_dtype)
        assert inner.W_UV.shape == (2, 4, 4)
        assert inner.W_UK_T.shape == (2, 2, 4)


@pytest.mark.parametrize("is_kv_consumer", [False, True])
def test_mlapo_post_load_does_not_reprocess_released_weights(is_kv_consumer):
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.fa_quant_layer = False
    impl.enable_mlapo = True
    impl._mlapo_uses_native_weights = False
    impl.q_lora_rank = 4
    impl.fused_qkv_a_proj = SimpleNamespace(weight=torch.ones(8, 10), weight_scale=torch.ones(10))
    impl.q_proj = SimpleNamespace(weight=torch.ones(4, 8), weight_scale=torch.ones(8))
    impl.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=is_kv_consumer),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
    )
    # Execute the real fused transformation and decoder weight-release branch.
    impl.process_weights_after_loading = MagicMock(side_effect=impl._process_weights_for_fused)
    inner, upstream = make_attention(impl)
    profile = MagicMock()
    profile.supports.return_value = False
    with (
        patch("vllm_ascend.attention.mla_v1.get_current_hardware_profile", return_value=profile),
        patch("vllm_ascend.attention.mla_v1.enable_dcp", return_value=False),
        patch(
            "vllm_ascend.attention.mla_v1.get_ascend_config",
            return_value=SimpleNamespace(mlapo_keep_prefill_weights=False),
        ),
        patch("torch_npu.npu_format_cast", side_effect=lambda weight, fmt: weight),
        patch.object(torch.npu, "empty_cache") as empty_cache,
    ):
        inner.process_weights_after_loading(torch.bfloat16)

    upstream.assert_called_once_with(torch.bfloat16)
    impl.process_weights_after_loading.assert_called_once_with(torch.bfloat16)
    assert (impl.fused_qkv_a_proj.weight is None) == is_kv_consumer
    assert (impl.q_proj.weight is None) == is_kv_consumer
    assert empty_cache.call_count == int(is_kv_consumer)
    torch.testing.assert_close(impl.weight_dq, torch.ones(8, 4))
    torch.testing.assert_close(impl.weight_dkv_kr, torch.ones(8, 6))
    torch.testing.assert_close(impl.weight_uq_qr, torch.ones(4, 8))
