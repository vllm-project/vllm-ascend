# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real Kimi projection construction and checkpoint loaders on CPU."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import KimiK3MergedQKVGateLinear
from vllm.models.kimi_k3.amd import linear as upstream_kimi

from vllm_ascend.models.kimi_k3.kimi_k3 import AscendKimiLinearModel, AscendKimiMLAAttention
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig

PREFIX = "model.layers.0.self_attn"


def _quant_config(qkv_type="W8A8_DYNAMIC", gate_type="FLOAT"):
    # Eco-Tech/Kimi-K3-w4a8 describes MLA q_a/kv_a as W8A8_DYNAMIC and
    # g_proj as FLOAT. These tiny tensors model that layout, not real weights.
    description = {
        f"{PREFIX}.{name}.weight": quant_type
        for name, quant_type in (
            ("q_a_proj", qkv_type),
            ("kv_a_proj_with_mqa", qkv_type),
            ("g_proj", gate_type),
            ("q_b_proj", "FLOAT"),
            ("kv_b_proj", "FLOAT"),
            ("o_proj", "FLOAT"),
        )
    }
    return AscendModelSlimConfig(description)


class _AttentionBoundary(nn.Module):
    """Replace only the NPU attention backend, leaving projection code real."""

    def __init__(self, *args):
        super().__init__()
        modules = args[8]
        fused = modules.fused_qkv_a_proj
        self.mla_attn = SimpleNamespace(
            impl=SimpleNamespace(
                fused_qkv_a_proj=fused,
                g_proj=modules.g_proj,
                qkv_a_gate_fused=len(fused.output_sizes) == 3,
                use_output_gate=modules.g_proj is not None or len(fused.output_sizes) == 3,
            )
        )


@pytest.fixture(params=[(1, 0), (2, 1), (16, 15)])
def projection_environment(request):
    tp_size, rank = request.param
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k3")))
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with (
        patch("vllm_ascend.quantization.configs.modelslim_config.get_current_vllm_config", return_value=config),
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=tp_size),
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=rank),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=rank),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=tp_size),
        patch.object(upstream_kimi, "get_tensor_model_parallel_world_size", return_value=tp_size),
        patch(
            "vllm_ascend.ops.linear.get_parallel_op",
            side_effect=lambda disable_tp, *args: (None, 0, 1) if disable_tp else (None, rank, tp_size),
        ),
        patch.object(upstream_kimi, "RMSNorm", lambda *args, **kwargs: nn.Identity()),
        patch.object(upstream_kimi, "KimiK3MultiHeadLatentAttentionWrapper", _AttentionBoundary),
    ):
        yield tp_size, rank
    torch.set_default_dtype(dtype)


def _attention(quant_config, gated=True):
    return AscendKimiMLAAttention(
        config=SimpleNamespace(mla_use_output_gate=gated, rms_norm_eps=1e-6),
        hidden_size=8,
        num_heads=32,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=4,
        q_lora_rank=4,
        kv_lora_rank=2,
        use_output_gate=gated,
        use_rope=False,
        quant_config=quant_config,
        prefix=PREFIX,
    )


def _model(attention, gated=True):
    model = AscendKimiLinearModel.__new__(AscendKimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        linear_attn_config=None,
        q_lora_rank=4,
        mla_use_output_gate=gated,
        is_moe=False,
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
    )
    layer = nn.Module()
    layer.self_attn = attention
    model.layers = nn.ModuleList([layer])
    model.n_redundant_experts = 0
    return model


def test_mixed_mla_preserves_float_gate_and_replicated_latents(projection_environment):
    attention = _attention(_quant_config())
    assert attention.fused_qkv_a_proj.output_sizes == [4, 4]
    assert attention.fused_qkv_a_proj.disable_tp
    assert attention.fused_qkv_a_proj.weight.dtype == torch.int8
    assert attention.g_proj.weight.dtype == torch.bfloat16
    tp_size, rank = projection_environment
    gate_rows = 128 // tp_size
    assert attention.g_proj.tp_rank == rank
    assert attention.g_proj.tp_size == tp_size
    assert attention.g_proj.gather_output is False
    assert attention.impl.g_proj is attention.g_proj
    assert attention.impl.use_output_gate and attention.use_output_gate
    assert not attention.impl.qkv_a_gate_fused
    assert attention.impl.fused_qkv_a_proj is attention.fused_qkv_a_proj

    q = torch.full((4, 8), 3, dtype=torch.int8)
    kv = torch.full((4, 8), 5, dtype=torch.int8)
    gate = (torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8) - 512).to(torch.bfloat16) / 2048
    gate_slice = gate[rank * gate_rows : (rank + 1) * gate_rows]
    model = _model(attention)
    weights = [("q_a_proj", q), ("kv_a_proj_with_mqa", kv), ("g_proj", gate)]
    checkpoint = [(f"layers.0.self_attn.{name}.weight", value) for name, value in weights]
    scale = torch.arange(4, dtype=torch.bfloat16).reshape(4, 1) + 1
    checkpoint += [
        (f"layers.0.self_attn.{name}.{suffix}", scale)
        for name in ("q_a_proj", "kv_a_proj_with_mqa")
        for suffix in ("weight_scale", "weight_offset")
    ]
    loaded = model.load_weights(checkpoint)
    assert loaded == {
        "layers.0.self_attn.fused_qkv_a_proj.weight",
        "layers.0.self_attn.g_proj.weight",
        "layers.0.self_attn.fused_qkv_a_proj.weight_scale",
        "layers.0.self_attn.fused_qkv_a_proj.weight_offset",
    }
    torch.testing.assert_close(attention.fused_qkv_a_proj.weight_scale, scale.repeat(2, 1))
    torch.testing.assert_close(attention.fused_qkv_a_proj.weight_offset, scale.repeat(2, 1))
    torch.testing.assert_close(attention.fused_qkv_a_proj.weight, torch.cat((q, kv)))
    torch.testing.assert_close(attention.g_proj.weight, gate_slice)
    # The gate remains nonzero; the old fused INT8 loader truncated it to zero.
    assert attention.g_proj.weight.count_nonzero() > 0
    for num_tokens in (1, 5, 16):
        hidden = torch.ones((num_tokens, 8), dtype=torch.bfloat16)
        actual = torch.nn.functional.linear(hidden, attention.g_proj.weight).sigmoid()
        expected = torch.nn.functional.linear(hidden, gate_slice).sigmoid()
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("quant_type", ["FLOAT", "W8A8_DYNAMIC"])
def test_uniform_mla_keeps_fused_gate(projection_environment, quant_type):
    config = _quant_config(quant_type, quant_type)
    attention = _attention(config)
    assert len(attention.fused_qkv_a_proj.output_sizes) == 3
    assert not hasattr(attention, "g_proj")
    assert attention.impl.qkv_a_gate_fused
    assert config.packed_modules_mapping["fused_qkv_a_proj"] == ["q_a_proj", "kv_a_proj_with_mqa"]
    tp_size, rank = projection_environment
    gate_rows = 128 // tp_size
    dtype = torch.bfloat16 if quant_type == "FLOAT" else torch.int8
    gate = torch.arange(128 * 8).reshape(128, 8).remainder(11).to(dtype)
    loaded = _model(attention).load_weights([("layers.0.self_attn.g_proj.weight", gate)])
    assert loaded == {"layers.0.self_attn.fused_qkv_a_proj.weight"}
    torch.testing.assert_close(attention.fused_qkv_a_proj.weight[8:], gate[rank * gate_rows : (rank + 1) * gate_rows])
    if quant_type != "FLOAT":
        scales = torch.arange(128, dtype=torch.bfloat16).reshape(128, 1) + 1
        latent_scales = torch.arange(4, dtype=torch.bfloat16).reshape(4, 1) + 2
        model = _model(attention)
        model.load_weights(
            [
                ("layers.0.self_attn.q_a_proj.weight_scale", latent_scales),
                ("layers.0.self_attn.kv_a_proj_with_mqa.weight_scale", latent_scales),
                ("layers.0.self_attn.g_proj.weight_scale", scales),
            ]
        )
        torch.testing.assert_close(attention.fused_qkv_a_proj.weight_scale[:8], latent_scales.repeat(2, 1))
        torch.testing.assert_close(
            attention.fused_qkv_a_proj.weight_scale[8:], scales[rank * gate_rows : (rank + 1) * gate_rows]
        )
    # A target and ungated drafter can share this config without leaking the
    # target's third shard into the draft mapping.
    draft = _attention(config, gated=False)
    assert draft.fused_qkv_a_proj.output_sizes == [4, 4]


def test_ungated_draft_keeps_two_shards_without_gate_metadata(projection_environment):
    config = _quant_config()
    del config.quant_description[f"{PREFIX}.g_proj.weight"]
    attention = _attention(config, gated=False)
    assert attention.fused_qkv_a_proj.output_sizes == [4, 4]
    assert not hasattr(attention, "g_proj")
    assert not attention.impl.use_output_gate


def test_fused_mla_validates_gate_without_mutating_two_shard_mapping(projection_environment):
    config = _quant_config()
    with pytest.raises(ValueError, match="Not all shards"):
        KimiK3MergedQKVGateLinear(8, 4, 2, 2, 32, 4, quant_config=config, prefix=f"{PREFIX}.fused_qkv_a_proj")
    del config.quant_description[f"{PREFIX}.g_proj.weight"]
    with pytest.raises(KeyError, match="g_proj"):
        _attention(config)
    assert config.packed_modules_mapping["fused_qkv_a_proj"] == ["q_a_proj", "kv_a_proj_with_mqa"]


def test_quantized_standalone_gate_loads_scales(projection_environment):
    attention = _attention(_quant_config("FLOAT", "W8A8_DYNAMIC"))
    model = _model(attention)
    gate = attention.g_proj
    tp_size, rank = projection_environment
    gate_rows = 128 // tp_size
    gate_slice = slice(rank * gate_rows, (rank + 1) * gate_rows)
    scale = torch.arange(128, dtype=torch.bfloat16).reshape(128, 1)
    loaded = model.load_weights(
        [
            ("layers.0.self_attn.g_proj.weight_scale", scale),
            ("layers.0.self_attn.g_proj.weight_offset", scale + 1),
        ]
    )
    assert loaded == {"layers.0.self_attn.g_proj.weight_scale", "layers.0.self_attn.g_proj.weight_offset"}
    torch.testing.assert_close(gate.weight_scale, scale[gate_slice])
    torch.testing.assert_close(gate.weight_offset, (scale + 1)[gate_slice])


@pytest.mark.parametrize("model_type", ["kimi_k3", "kimi_linear", "kimi_k3_mtp"])
def test_mla_gate_layout_for_each_model_type(projection_environment, model_type):
    runtime = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)))
    with patch("vllm_ascend.quantization.configs.modelslim_config.get_current_vllm_config", return_value=runtime):
        attention = _attention(_quant_config())
        assert attention.g_proj.weight.dtype == torch.bfloat16
        assert attention.fused_qkv_a_proj.weight.dtype == torch.int8
        uniform = _attention(_quant_config("W8A8_DYNAMIC", "W8A8_DYNAMIC"))
        assert uniform.fused_qkv_a_proj.weight.dtype == torch.int8
        config = _quant_config()
        config.quant_description[f"{PREFIX}.kv_a_proj_with_mqa.weight"] = "FLOAT"
        assert not config.uses_kimi_k3_mixed_mla_projection(PREFIX)
        with pytest.raises(ValueError, match="Not all shards"):
            _attention(config)


def test_mixed_mla_accepts_bare_quant_description_keys(projection_environment):
    config = _quant_config()
    config.quant_description = {name.removesuffix(".weight"): value for name, value in config.quant_description.items()}
    assert config.uses_kimi_k3_mixed_mla_projection(PREFIX)
    attention = _attention(config)
    assert attention.g_proj.weight.dtype == torch.bfloat16
    assert attention.fused_qkv_a_proj.weight.dtype == torch.int8


def test_standalone_gate_loader_preserves_kwargs(projection_environment):
    attention = _attention(_quant_config())
    original_loader = attention.g_proj.weight.weight_loader
    calls = []

    def recording_loader(param, value, **kwargs):
        calls.append(kwargs)
        original_loader(param, value)

    attention.g_proj.weight.weight_loader = recording_loader
    gate = torch.ones((128, 8), dtype=torch.bfloat16)
    loaded = _model(attention).load_weights([("layers.0.self_attn.g_proj.weight", gate, {"checkpoint_flag": True})])
    assert calls == [{"checkpoint_flag": True}]
    assert loaded == {"layers.0.self_attn.g_proj.weight"}
