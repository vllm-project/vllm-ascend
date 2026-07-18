# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_ascend.quantization.nvfp4 import AscendNvFp4Config

MISTRAL_NVFP4_CONFIG = {
    "quantization_config": {
        "config_groups": {
            "NVFP4": {
                "format": "nvfp4-pack-quantized",
                "weights": {"group_size": 16, "num_bits": 4, "type": "float"},
            }
        },
        "format": "nvfp4-pack-quantized",
        "ignore": ["re:.*attn.*", "lm_head"],
        "quant_method": "compressed-tensors",
    }
}


def test_mistral_native_nvfp4_config_is_auto_detected():
    assert (
        AscendNvFp4Config.override_quantization_method(
            MISTRAL_NVFP4_CONFIG,
            user_quant=None,
        )
        == "nvfp4"
    )


def test_mistral_native_nvfp4_config_and_regex_ignore_are_parsed():
    config = AscendNvFp4Config.from_config(MISTRAL_NVFP4_CONFIG)

    assert config.group_size == 16
    assert config._is_ignored("model.layers.0.self_attn.q_proj")
    assert config._is_ignored("lm_head")
    assert not config._is_ignored("model.layers.3.mlp.experts")
