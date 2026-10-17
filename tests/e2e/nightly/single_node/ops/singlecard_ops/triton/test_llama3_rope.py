# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.llama3_rope import Llama3RotaryEmbedding

from vllm_ascend.ops.rotary_embedding import AscendLlama3RotaryEmbedding
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


class ReferenceLlama3RotaryEmbedding(Llama3RotaryEmbedding):
    """Avoid the exact-name OOT registration when constructing the reference."""


@pytest.mark.parametrize("tokens", [1, 4, 32, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_neox_style", [True, False])
def test_llama3_scaled_cache_and_fused_forward(tokens, dtype, is_neox_style):
    init_device_properties_triton()
    torch.manual_seed(0)
    head_size = 128
    max_position = 32768
    with set_current_vllm_config(VllmConfig()):
        args = (head_size, head_size, max_position, 500000, is_neox_style, dtype, 8, 1, 4, 8192)
        reference = ReferenceLlama3RotaryEmbedding(*args).npu()
        candidate = AscendLlama3RotaryEmbedding(*args).npu()
        torch.testing.assert_close(candidate.cos_sin_cache, reference.cos_sin_cache, rtol=0, atol=0)
        # Noncontiguous Q/K views match a merged QKV projection in Llama 70B.
        qkv = torch.randn(tokens, (64 + 2 * 8) * head_size, dtype=dtype, device="npu")
        query, key, _ = qkv.split([64 * head_size, 8 * head_size, 8 * head_size], dim=-1)
        positions = torch.tensor([0, 8191, 8192, max_position - 1], device="npu").repeat((tokens + 3) // 4)[:tokens]
        cos, sin = reference.cos_sin_cache.index_select(0, positions).float().chunk(2, dim=-1)
        expected = [
            ApplyRotaryEmb.forward_static(x.float().view(tokens, -1, head_size), cos, sin, is_neox_style)
            .reshape_as(x)
            .to(dtype)
            for x in (query, key)
        ]
        actual = candidate(positions, query, key)
        for output, target in zip(actual, expected):
            torch.testing.assert_close(output, target, atol=1e-3, rtol=1e-3)
