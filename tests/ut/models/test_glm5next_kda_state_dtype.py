# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig

from vllm_ascend.models.glm5next.kda import Glm5NextLinearAttention
from vllm_ascend.models.glm5next.model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration


def make_layer_and_config(ssm_dtype, conv_dtype="auto"):
    cache_config = CacheConfig(
        mamba_cache_dtype=conv_dtype,
        mamba_ssm_cache_dtype=ssm_dtype,
        mamba_block_size=512,
        mamba_cache_mode="align",
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16, hf_text_config=SimpleNamespace(model_type="glm5_next")),
        cache_config=cache_config,
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
        num_speculative_tokens=5,
    )
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.use_recoverssm = False
    layer.model_config, layer.cache_config = config.model_config, cache_config
    layer.tp_size, layer.num_heads, layer.head_dim, layer.conv_size, layer.num_spec = 1, 64, 128, 4, 5
    return layer, config


@pytest.mark.parametrize(
    ("ssm_dtype", "conv_dtype", "expected_conv", "expected"),
    [
        ("auto", "auto", torch.bfloat16, torch.float32),
        ("float32", "auto", torch.bfloat16, torch.float32),
        ("bfloat16", "auto", torch.bfloat16, torch.bfloat16),
        ("bfloat16", "float32", torch.float32, torch.bfloat16),
    ],
)
def test_kda_state_dtype_matches_model_and_cache_spec(ssm_dtype, conv_dtype, expected_conv, expected):
    layer, config = make_layer_and_config(ssm_dtype, conv_dtype)
    want = (expected_conv, expected)
    assert layer.get_state_dtype() == want
    assert Glm5NextForCausalLM.get_mamba_state_dtype_from_config(config) == want
    assert Glm5NextForConditionalGeneration.get_mamba_state_dtype_from_config(config) == want
    spec = layer.get_kv_cache_spec(config)
    assert spec.dtypes == want
    state_bytes = 64 * 128 * 128 * torch.empty((), dtype=expected).element_size()
    conv_bytes = 3 * 64 * 128 * (3 + 5) * torch.empty((), dtype=expected_conv).element_size()
    assert spec.page_size_bytes == state_bytes + conv_bytes


def test_kda_rejects_unsupported_recurrent_state_dtype():
    layer, config = make_layer_and_config("float16")
    with pytest.raises(ValueError, match="KDA recurrent state supports only"):
        layer.get_state_dtype()
    with pytest.raises(ValueError, match="KDA recurrent state supports only"):
        Glm5NextForCausalLM.get_mamba_state_dtype_from_config(config)
