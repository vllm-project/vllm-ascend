# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for native GLM-5.3-Flash checkpoint metadata and initialization."""

import math
from unittest.mock import patch

import pytest
import torch
from transformers import Glm5NextConfig

from tests.e2e.pull_request.rlhf.glm53flash_weight_fixture import initialize_tensor
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import MODEL_CASES, FixedRandomWeightSource


def test_checkpoint_metadata_covers_the_reduced_model():
    case = next(case for case in MODEL_CASES if case.id == "glm-5.3-flash-kda-sfa-mhc")
    config = Glm5NextConfig(text_config={"index_kpool": 4, **case.hf_overrides["text_config"]})
    with patch(
        "tests.e2e.pull_request.rlhf.weight_transfer_test_utils.AutoConfig.from_pretrained", return_value=config
    ):
        source = FixedRandomWeightSource(case, torch.device("cpu"))
    parameters = [(meta.name, meta.shape, meta.dtype) for meta in source.metadata()]
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
    # Native persistent correction bias retains FP32, like M3. Parameters use BF16.
    assert {name for name, _, dtype in parameters if dtype == torch.float32} == {
        "model.layers.3.mlp.gate.e_score_correction_bias"
    }
    assert all(dtype in (torch.bfloat16, torch.float32) for _, _, dtype in parameters)
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
