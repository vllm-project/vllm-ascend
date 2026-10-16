# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the offline Flash fixture's architecture and payload contract."""

import json
import math

import pytest
import torch

from tests.e2e.pull_request.rlhf.glm53flash_weight_fixture import ASSETS, initialize_tensor, load_parameter_metadata


def test_reduced_config_keeps_hybrid_attention_and_moe():
    original = json.loads((ASSETS / "config.original.json").read_text(encoding="utf-8"))["text_config"]
    reduced = json.loads((ASSETS / "config.json").read_text(encoding="utf-8"))
    assert reduced["architectures"] == ["Glm5NextForCausalLM"]
    assert reduced["num_hidden_layers"] == 4
    assert reduced["num_nextn_predict_layers"] == 0
    assert reduced["n_routed_experts"] == reduced["num_experts_per_tok"] == 8
    for key in ("layer_types", "mlp_layer_types", "indexer_types"):
        assert reduced[key] == original[key][:4]
    assert reduced["linear_attn_config"]["kda_layers"] == [0, 1, 2]
    assert reduced["linear_attn_config"]["full_attn_layers"] == [3]
    for key in ("hidden_size", "vocab_size", "moe_intermediate_size", "kv_lora_rank", "q_lora_rank", "mhc", "hc_mult"):
        assert reduced[key] == original[key]
    assert "quantization_config" not in reduced


def test_checkpoint_metadata_covers_the_reduced_model():
    parameters = load_parameter_metadata(ASSETS / "parameter_metadata.json")
    shapes = {name: shape for name, shape, _ in parameters}
    assert len(shapes) == 132
    assert shapes["model.embed_tokens.weight"] == shapes["lm_head.weight"] == (154880, 4096)
    assert shapes["model.norm.weight"] == (4096,)
    assert shapes["model.layers.3.mlp.gate.weight"] == (8, 4096)
    assert shapes["model.layers.3.mlp.gate.e_score_correction_bias"] == (8,)
    for layer in range(4):
        assert shapes[f"model.layers.{layer}.hc_attn_fn"] == (24, 16384)
        assert shapes[f"model.layers.{layer}.hc_ffn_fn"] == (24, 16384)
        if layer < 3:
            assert shapes[f"model.layers.{layer}.self_attn.A_log"] == (64,)
            for projection in ("q", "k", "v"):
                assert f"model.layers.{layer}.self_attn.{projection}_proj.weight" in shapes
                assert f"model.layers.{layer}.self_attn.{projection}_conv1d.weight" in shapes
        else:
            assert "model.layers.3.self_attn.indexer.index_kpool_compress_gate" in shapes
            assert "model.layers.3.self_attn.indexer.wk.weight" in shapes
            for expert in range(8):
                for projection in ("gate", "up", "down"):
                    assert f"model.layers.3.mlp.experts.{expert}.{projection}_proj.weight" in shapes
    assert all(dtype == torch.bfloat16 for _, _, dtype in parameters)
    assert not any("weight_scale" in name or "visual" in name or "layers.4." in name for name in shapes)
    assert not any("experts.8." in name or "language_model" in name for name in shapes)
    # Production widths remain, but the single-card test payload is under 5 GiB.
    assert sum(math.prod(shape) * 2 for _, shape, _ in parameters) < 5 * 2**30


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.0.self_attn.A_log", 0.0),
        ("model.layers.0.self_attn.dt_bias", -2.0),
        ("model.layers.0.hc_attn_scale", 1.0),
        ("model.layers.0.hc_ffn_base", 0.0),
        ("model.layers.0.input_layernorm.weight", 1.0),
        ("model.layers.0.self_attn.o_norm.weight", 1.0),
        ("model.layers.3.self_attn.indexer.k_norm.bias", 0.0),
        ("model.layers.0.hc_attn_fn", 0.0005),
        ("model.layers.3.mlp.experts.0.gate_proj.weight", 0.005),
    ],
)
def test_synthetic_initialization_is_bounded(name, expected):
    tensor = torch.full((4,), 0.01, dtype=torch.bfloat16)
    result = initialize_tensor(name, tensor)
    assert result is tensor
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.full_like(result, expected))


@pytest.mark.parametrize("entries", [[], [{"name": "x", "shape": [1]}] * 2, [{"name": "x", "shape": [0]}]])
def test_invalid_metadata_fails_before_transfer(tmp_path, entries):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(ValueError):
        load_parameter_metadata(path)
