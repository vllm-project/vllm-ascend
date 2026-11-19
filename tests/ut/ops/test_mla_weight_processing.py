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
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, PreprocessType
from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention, IndexerWrapper
from vllm_ascend.utils import vllm_version_is


def make_attention(impl):
    # Run the installed upstream post-load method, not a mock of its dispatch.
    inner = MLAAttention.__new__(MLAAttention)
    nn.Module.__init__(inner)
    set_default_quant_scales(inner, register_buffer=True)
    inner.impl = impl
    inner.act_dtype = torch.bfloat16
    inner.num_heads = 2
    inner.kv_lora_rank = 4
    inner.qk_nope_head_dim = 2
    inner.v_head_dim = 4
    inner.kv_b_proj = nn.Linear(4, 12, bias=False)
    inner.kv_b_proj.quant_method = UnquantizedLinearMethod()
    inner.is_amx_bmm_enabled = False
    inner.dcp_q_replicate = False
    inner.dcp_manager = None
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
        patch("vllm_ascend.ops.mla.AscendMLAAttention", return_value=inner),
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

    inner.act_dtype = act_dtype
    if vllm_version_is("0.33.0"):
        inner.process_weights_after_loading(act_dtype)
    else:
        inner.process_weights_after_loading()

    impl.process_weights_after_loading.assert_called_once_with(act_dtype)
    if is_sfa:
        upstream.assert_not_called()
        assert inner.kv_b_proj.weight is None
    else:
        upstream.assert_called_once_with(*([act_dtype] if vllm_version_is("0.33.0") else []))
        assert inner.W_UV.shape == (2, 4, 4)
        assert inner.W_UK_T.shape == (2, 2, 4)


@pytest.mark.parametrize("is_kv_consumer", [False, True])
def test_mlapo_post_load_does_not_reprocess_released_weights(is_kv_consumer):
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.is_pcp_decode_sharded = False
    impl.fa_quant_layer = False
    impl.enable_mlapo = True
    impl.use_dcp = False
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
        patch(
            "vllm_ascend.attention.mla_v1.get_ascend_config",
            return_value=SimpleNamespace(mlapo_keep_prefill_weights=False),
        ),
        patch("torch_npu.npu_format_cast", side_effect=lambda weight, fmt: weight),
        patch.object(torch.npu, "empty_cache") as empty_cache,
    ):
        inner.process_weights_after_loading(torch.bfloat16)

    upstream.assert_called_once_with(*([torch.bfloat16] if vllm_version_is("0.33.0") else []))
    impl.process_weights_after_loading.assert_called_once_with(torch.bfloat16)
    assert (impl.fused_qkv_a_proj.weight is None) == is_kv_consumer
    assert (impl.q_proj.weight is None) == is_kv_consumer
    assert empty_cache.call_count == int(is_kv_consumer)
    torch.testing.assert_close(impl.weight_dq, torch.ones(8, 4))
    torch.testing.assert_close(impl.weight_dkv_kr, torch.ones(8, 6))
    torch.testing.assert_close(impl.weight_uq_qr, torch.ones(4, 8))


@pytest.fixture
def native_sfa(monkeypatch):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.has_indexer = False
    impl.local_num_heads = 2
    impl.kv_lora_rank = 4
    impl.qk_nope_head_dim = 2
    impl.v_head_dim = 4
    impl.kv_b_proj = nn.Linear(4, 12, bias=False, dtype=torch.bfloat16)
    impl.vllm_config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16, quantization=None))
    impl.rl_weight_update_enabled = False
    impl.enable_sparse_sfa_c8 = False
    impl._parallel_o_proj = None
    impl._resolve_preprocess_type = MagicMock(return_value=PreprocessType.NATIVE)
    monkeypatch.setattr("torch_npu.npu_format_cast", lambda weight, fmt: weight)
    monkeypatch.setattr("vllm_ascend.attention.sfa_v1.maybe_trans_nz", lambda weight: weight)
    return impl


@pytest.mark.parametrize("indexer_layout", ["module", "plain_backend", "plain_indexer"])
def test_indexer_finalized_once_per_complete_load(indexer_layout, native_sfa, monkeypatch):
    from vllm.model_executor.model_loader.weight_utils import finalize_model_weights

    events = []

    class Backend(nn.Module):
        def process_weights_after_loading(self):
            events.append("backend")

    backend = (
        Backend()
        if indexer_layout == "module"
        else SimpleNamespace(process_weights_after_loading=lambda: events.append("backend"))
    )
    indexer = IndexerWrapper.__new__(IndexerWrapper)
    nn.Module.__init__(indexer)
    indexer.impl = backend
    if indexer_layout == "plain_indexer":
        indexer = backend
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    native_sfa.has_indexer = True
    native_sfa.indexer = indexer

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.indexer = indexer

        def process_weights_after_loading(self):
            events.append("attention")
            native_sfa.process_weights_after_loading(torch.bfloat16)

    attention = Attention()
    root = nn.Module()
    root.attention = attention
    root.shared_indexer_alias = indexer
    for _ in range(2):
        events.clear()
        finalize_model_weights(root)
        expected = ["attention", "backend"] if indexer_layout == "plain_indexer" else ["backend", "attention"]
        assert events == expected
    assert native_sfa._resolve_preprocess_type.call_count == 2


@pytest.mark.parametrize("mode", ["layerwise", "direct"])
def test_sfa_disk_only_keeps_loadable_source_for_direct(native_sfa, monkeypatch, mode):
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    source = native_sfa.kv_b_proj.weight
    with torch.no_grad():
        source.fill_(1)
        native_sfa.process_weights_after_loading(torch.bfloat16)
        if mode == "layerwise":
            assert source.numel() == 0
            return
        captured = native_sfa.W_UV.detach()
        source.fill_(3)
        native_sfa.process_weights_after_loading(torch.bfloat16)
        assert native_sfa.kv_b_proj.weight is source
        assert native_sfa.W_UV.data_ptr() == captured.data_ptr()
        torch.testing.assert_close(captured, torch.full_like(captured, 3))
