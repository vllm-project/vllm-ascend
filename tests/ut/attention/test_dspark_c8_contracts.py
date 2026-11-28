# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from vllm.compilation import backends
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.models import utils as model_utils

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.attention import mla_v1
from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.models.kimi_k3 import kimi_k3_dspark
from vllm_ascend.quantization import utils as quant_utils
from vllm_ascend.quantization.configs import modelslim_config


@pytest.mark.parametrize("layer_offset,precision", [(0, "c8"), (1, "c8"), (93, "c8"), (93, "bf16"), (93, "w8a8")])
def test_dspark_maps_its_recipe_before_building_layers(monkeypatch, layer_offset, precision):
    weight_type = "W8A8_DYNAMIC" if precision == "w8a8" else "FLOAT"
    recipe = {
        "layers.0.self_attn.q_a_proj.weight": weight_type,
        "layers.0.self_attn.kv_a_proj_with_mqa.weight": weight_type,
        "layers.1.mlp.gate_proj.weight": "W8A8_DYNAMIC",
    }
    if precision == "c8":
        recipe.update({"fa_quant_type": "FAKQuant", "layers.0.self_attn.fa_k.scale": "FAQuant"})
    draft_quant = modelslim_config.AscendModelSlimConfig(recipe.copy()) if precision != "bf16" else None
    target_quant = modelslim_config.AscendModelSlimConfig({"model.layers.0.self_attn.q_a_proj.weight": "FLOAT"})
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            kv_cache_dtype=None,
            draft_model_config=SimpleNamespace(
                use_mla=True,
                quantization="ascend" if draft_quant is not None else None,
                hf_config=SimpleNamespace(
                    model_type="k3_dspark",
                    num_hidden_layers=2,
                    hidden_size=8,
                    target_hidden_size=8,
                    num_target_layers=1,
                    vocab_size=8,
                    draft_vocab_size=8,
                    markov_rank=2,
                    rms_norm_eps=1e-6,
                    num_attention_heads=2,
                    qk_nope_head_dim=2,
                    qk_rope_head_dim=2,
                    v_head_dim=2,
                    q_lora_rank=2,
                    kv_lora_rank=2,
                    intermediate_size=8,
                    hidden_act="silu",
                ),
            ),
        ),
        model_config=SimpleNamespace(use_mla=True, hf_text_config=SimpleNamespace(index_topk=2048)),
        quant_config=target_quant,
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        kv_transfer_config=None,
    )
    monkeypatch.setattr(kimi_k3_dspark, "get_draft_quant_config", lambda _: draft_quant)
    fallback_quant = MagicMock(return_value=draft_quant)
    monkeypatch.setattr(model_utils, "get_draft_quant_config", fallback_quant)
    monkeypatch.setattr(backends, "model_tag", "dspark_head")
    profile = SimpleNamespace(supports=lambda _: True)
    monkeypatch.setattr(modelslim_config, "get_current_hardware_profile", lambda: profile)
    monkeypatch.setattr(quant_utils, "get_current_hardware_profile", lambda: profile)
    monkeypatch.setattr(kimi_k3_dspark, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(kimi_k3_dspark, "ColumnParallelLinear", lambda *args, **kwargs: nn.Module())
    monkeypatch.setattr(kimi_k3_dspark, "RMSNorm", lambda *args, **kwargs: nn.Module())
    monkeypatch.setattr(kimi_k3_dspark, "DSparkMarkovHead", lambda *args, **kwargs: nn.Module())

    def capture_layer(*, prefix, quant_config, **kwargs):
        layer = nn.Module()
        layer.prefix = prefix
        layer.quant_config = quant_config
        return layer

    monkeypatch.setattr(kimi_k3_dspark, "AscendKimiMLAAttention", capture_layer)
    monkeypatch.setattr(kimi_k3_dspark, "KimiMLP", capture_layer)
    draft = kimi_k3_dspark.AscendK3DSparkModel(vllm_config=config, start_layer_id=layer_offset, prefix="model")
    attn = draft.layers[0].self_attn
    mlp = draft.layers[1].mlp
    assert attn.prefix == f"model.layers.{layer_offset}.self_attn"
    assert mlp.prefix == f"model.layers.{layer_offset + 1}.mlp"
    assert attn.quant_config is mlp.quant_config is draft.quant_config
    draft_fa_quant = quant_utils.enable_fa_quant(config, f"{attn.prefix}.attn", draft_quant_config=draft.quant_config)
    assert draft_fa_quant is (precision == "c8")
    assert not quant_utils.enable_fa_quant(
        config, f"{draft.layers[1].self_attn.prefix}.attn", draft_quant_config=draft.quant_config
    )
    assert target_quant.quant_description == {"model.layers.0.self_attn.q_a_proj.weight": "FLOAT"}
    if draft_quant is not None:
        fallback_quant.assert_not_called()
        assert draft_quant.quant_description == recipe
        if precision == "c8":
            assert draft.quant_config.quant_description[f"{attn.prefix}.fa_k.scale"] == "FAQuant"
        assert draft.quant_config.quant_description[f"{mlp.prefix}.gate_proj.weight"] == "W8A8_DYNAMIC"
        assert modelslim_config.get_quant_type_for_layer(
            draft.quant_config.quant_description,
            f"{attn.prefix}.fused_qkv_a_proj",
            {"fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]},
        ) == ("W8A8_DYNAMIC" if precision == "w8a8" else None)
    if precision == "w8a8":
        config.speculative_config.kv_cache_dtype = "fp8"
        with pytest.raises(ValueError, match="corresponding quantized weights are required"):
            quant_utils.enable_fa_quant(config, f"{attn.prefix}.attn", draft_quant_config=draft.quant_config)


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
        _is_k3_dspark_draft=True,
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
        fa_quant_layer=True,
        _is_k3_dspark_draft=True,
        _mlapo_uses_native_weights=False,
        _load_fa_quant_scales=MagicMock(),
        q_proj=SimpleNamespace(quant_method=UnquantizedLinearMethod()),
        fused_qkv_a_proj=SimpleNamespace(quant_method=SimpleNamespace(quant_method=object())),
    )
    with pytest.raises(RuntimeError, match="compatible weight methods"):
        AscendMLAImpl._process_weights_for_fused(impl, torch.bfloat16)
