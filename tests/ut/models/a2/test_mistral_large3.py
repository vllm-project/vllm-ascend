# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm_ascend.model_executor.models.mistral_large3 import MistralLarge3ForCausalLM  # type: ignore[import-untyped]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "layers.3.attention.wq_a.weight",
            "model.layers.3.self_attn.q_a_proj.weight",
        ),
        (
            "layers.3.experts.17.w1.weight",
            "model.layers.3.mlp.experts.17.gate_proj.weight",
        ),
        (
            "layers.3.shared_experts.w2.qscale_act",
            "model.layers.3.mlp.shared_experts.down_proj.input_scale",
        ),
        (
            "layers.3.experts.17.w3.qscale_weight_2",
            "model.layers.3.mlp.experts.17.up_proj.weight_scale_2",
        ),
        ("tok_embeddings.weight", "model.embed_tokens.weight"),
        ("output.weight", "lm_head.weight"),
    ],
)
def test_mistral_large3_weight_mapping(source: str, expected: str) -> None:
    assert MistralLarge3ForCausalLM.hf_to_vllm_mapper._map_name(source) == expected
