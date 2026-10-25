# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State sizing must work before the multimodal text tower is constructed."""

from types import SimpleNamespace

import pytest

from vllm_ascend.models.glm5next.model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration


@pytest.mark.parametrize("model_cls", [Glm5NextForCausalLM, Glm5NextForConditionalGeneration])
@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("num_spec", [0, 5])
def test_kda_state_shape_uses_text_config(model_cls, multimodal, tp_size, num_spec):
    text_config = SimpleNamespace(linear_num_heads=64, linear_head_dim=128, linear_conv_kernel_dim=4)
    hf_config = SimpleNamespace(text_config=text_config) if multimodal else text_config
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, hf_text_config=text_config),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        speculative_config=SimpleNamespace(num_speculative_tokens=num_spec) if num_spec else None,
    )

    conv_shape, recurrent_shape = model_cls.get_mamba_state_shape_from_config(config)

    assert conv_shape == (3 + num_spec, 3 * 64 * 128 // tp_size)
    assert recurrent_shape == (64 // tp_size, 128, 128)
