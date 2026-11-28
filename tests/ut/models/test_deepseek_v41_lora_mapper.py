# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter keys must resolve to the modules the model actually builds."""

import pytest
from vllm.lora.utils import parse_fine_tuned_lora_name

from vllm_ascend.models.deepseek_v41.vl_model import AscendDeepseekV41ForCausalLM


@pytest.mark.parametrize(
    ("raw_key", "expected_module", "expected_is_a"),
    [
        *(
            (
                f"base_model.model.model.layers.3.attn.{p}.lora_{s}.weight",
                f"language_model.model.layers.3.self_attn.{p}",
                s == "A",
            )
            for p, s in (("wq_a", "A"), ("wq_b", "B"), ("wkv", "B"), ("wo_a", "B"), ("wo_b", "B"))
        ),
        (
            "language_model.model.layers.1.self_attn.wkv.lora_A.weight",
            "language_model.model.layers.1.self_attn.wkv",
            True,
        ),
        ("base_model.model.lm_head.lora_A.weight", "language_model.lm_head", True),
        (
            "base_model.model.model.layers.5.ffn.experts.12.w1.lora_A.weight",
            "language_model.model.layers.5.mlp.experts.12.gate_proj",
            True,
        ),
        (
            "base_model.model.model.layers.5.ffn.experts.12.w2.lora_B.weight",
            "language_model.model.layers.5.mlp.experts.12.down_proj",
            False,
        ),
        (
            "base_model.model.model.layers.5.ffn.experts.12.w3.lora_A.weight",
            "language_model.model.layers.5.mlp.experts.12.up_proj",
            True,
        ),
        (
            "base_model.model.model.layers.5.ffn.shared_experts.w1.lora_A.weight",
            "language_model.model.layers.5.mlp.shared_experts.gate_proj",
            True,
        ),
    ],
)
def test_deepseek_v41_lora_name_mapping(raw_key: str, expected_module: str, expected_is_a: bool) -> None:
    mapper = AscendDeepseekV41ForCausalLM.build_lora_weights_mapper().get_rename_mapper()
    module_name, is_lora_a = parse_fine_tuned_lora_name(raw_key, mapper)
    assert module_name == expected_module
    assert is_lora_a is expected_is_a
