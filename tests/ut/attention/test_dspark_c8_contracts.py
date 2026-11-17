# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.compilation import backends
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.attention import mla_v1
from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.quantization.utils import resolve_k3_dspark_quant_prefix


@pytest.mark.parametrize(
    "prefix, expected",
    [
        ("model.layers.93.self_attn.attn", "layers.0.self_attn.attn"),
        ("model.layers.97.self_attn.attn", "layers.4.self_attn.attn"),
        ("model.layers.92.self_attn.attn", "model.layers.92.self_attn.attn"),
        ("model.layers.98.self_attn.attn", "model.layers.98.self_attn.attn"),
        ("layers.0.self_attn.attn", "layers.0.self_attn.attn"),
    ],
)
def test_dspark_prefix_mapping_is_scoped_and_idempotent(monkeypatch, prefix, expected):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="k3_dspark", num_hidden_layers=5)),
        ),
        model_config=SimpleNamespace(get_num_layers=lambda _: 93),
        parallel_config=SimpleNamespace(),
    )
    monkeypatch.setattr(backends, "model_tag", "dspark_head")
    mapped = resolve_k3_dspark_quant_prefix(config, prefix)
    assert mapped == expected
    assert resolve_k3_dspark_quant_prefix(config, mapped) == expected
    monkeypatch.setattr(backends, "model_tag", "backbone")
    assert resolve_k3_dspark_quant_prefix(config, prefix) == prefix
    assert resolve_k3_dspark_quant_prefix(None, prefix) == prefix


@pytest.mark.parametrize("quant_backend", [False, True])
def test_joint_query_quantization_only_changes_quant_flashmla(monkeypatch, quant_backend):
    tokens, heads = 2, 1
    latent_dim = mla_v1.FLASHMLA_V_DIM
    rope_dim = mla_v1.FLASHMLA_QK_DIM - latent_dim
    latent = torch.full((tokens, heads, latent_dim), 2.0, dtype=torch.bfloat16)
    rope = torch.full((tokens, heads, rope_dim), 4.0, dtype=torch.bfloat16)
    kv_descale = torch.tensor([0.5])
    q_descale = torch.full((tokens, heads), 2.0)
    captured = []

    def quantize(x, *, dst_type):
        captured.append(x.clone())
        return (x.float() / q_descale.unsqueeze(-1)).to(dst_type), q_descale

    monkeypatch.setattr(mla_v1.torch_npu, "npu_dynamic_quant", quantize)
    impl = SimpleNamespace(
        _q_proj_and_k_up_proj=lambda _: (latent, rope),
        reorg_decode_q=lambda a, b: (a, b),
        rope_single=lambda x, *_: x,
        fa_quant_layer=True,
        support_fp8_attention=True,
        use_quant_flashmla=quant_backend,
        fak_descale_float=kv_descale,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        _decode_requires_current_kv=lambda _: False,
        exec_kv_decode=lambda *args, **kwargs: (None, None),
    )
    metadata = SimpleNamespace(
        num_decode_tokens=tokens,
        decode=SimpleNamespace(cos=None, sin=None),
        slot_mapping=torch.arange(tokens),
        external_flashmla=None,
    )
    result = AscendMLAImpl.mla_preprocess_decode(impl, torch.empty(tokens, 1), torch.empty(tokens, 1), None, metadata)
    assert len(captured) == 1
    assert result.dequant_scale_q_nope is q_descale
    if quant_backend:
        assert captured[0].shape[-1] == mla_v1.FLASHMLA_QK_DIM
        torch.testing.assert_close(captured[0][..., :latent_dim], latent)
        torch.testing.assert_close(captured[0][..., latent_dim:], (rope.float() / kv_descale).bfloat16())
        assert result.ql_nope.dtype == result.q_pe.dtype == torch.float8_e4m3fn
    else:
        torch.testing.assert_close(captured[0], latent)
        assert result.q_pe.dtype == torch.bfloat16
    torch.testing.assert_close(result.q_pe.float(), rope.float() / q_descale.unsqueeze(-1) / kv_descale)


def test_native_fused_weights_do_not_require_projection_scales(monkeypatch):
    native = UnquantizedLinearMethod()
    q_weight = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
    fused_weight = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4)
    impl = SimpleNamespace(
        fa_quant_layer=True,
        _load_fa_quant_scales=MagicMock(),
        q_proj=SimpleNamespace(quant_method=native, weight=q_weight),
        fused_qkv_a_proj=SimpleNamespace(quant_method=native, weight=fused_weight),
        _mlapo_uses_native_weights=False,
        head_padding=0,
        q_lora_rank=2,
        _fused_qkv_a_kv_end=5,
        num_heads_padded=1,
        support_fp8_attention=True,
        enable_mlapo=False,
    )
    monkeypatch.setattr(mla_v1.torch_npu, "npu_format_cast", lambda x, _: x)
    monkeypatch.setattr(mla_v1, "get_ascend_config", lambda: SimpleNamespace())
    monkeypatch.setattr(mla_v1, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: True))
    AscendMLAImpl._process_weights_for_fused(impl, torch.bfloat16)
    assert impl._mlapo_uses_native_weights
    torch.testing.assert_close(impl.weight_uq_qr, q_weight.T.contiguous())
    torch.testing.assert_close(impl.weight_dq, fused_weight.T[:, : impl.q_lora_rank].contiguous())
    impl._load_fa_quant_scales.assert_called_once_with()


def test_mixed_projection_methods_are_rejected_before_weight_conversion():
    impl = SimpleNamespace(
        fa_quant_layer=False,
        q_proj=SimpleNamespace(quant_method=UnquantizedLinearMethod()),
        fused_qkv_a_proj=SimpleNamespace(quant_method=SimpleNamespace(quant_method=object())),
    )
    with pytest.raises(RuntimeError, match="compatible weight methods"):
        AscendMLAImpl._process_weights_for_fused(impl, torch.bfloat16)
