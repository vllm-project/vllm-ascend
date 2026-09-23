# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.gemma4_rope import Gemma4RotaryEmbedding

from vllm_ascend.ops.rotary_embedding import AscendGemma4RotaryEmbedding
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


class ReferenceGemma4RotaryEmbedding(Gemma4RotaryEmbedding):
    """Avoid the exact-name OOT registration when constructing the reference."""


# Gemma 4 global attention: 512-dim heads, partial_rotary_factor=0.25, theta=1e6.
HEAD_SIZE = 512
ROTARY_DIM = 128
NUM_HEADS, NUM_KV_HEADS = 32, 4
MAX_POSITION = 32768
BASE = 1000000.0


def _expected(x: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor, is_neox_style: bool) -> torch.Tensor:
    tokens = x.shape[0]
    cos, sin = cache.index_select(0, positions).float().chunk(2, dim=-1)
    out = ApplyRotaryEmb.forward_static(x.float().view(tokens, -1, HEAD_SIZE), cos, sin, is_neox_style)
    return out.reshape_as(x).to(x.dtype)


@pytest.mark.parametrize("tokens", [1, 4, 32, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("query_only", [False, True])
def test_gemma4_proportional_cache_and_fused_forward(tokens, dtype, query_only):
    init_device_properties_triton()
    torch.manual_seed(0)
    with set_current_vllm_config(VllmConfig()):
        args = (HEAD_SIZE, ROTARY_DIM, MAX_POSITION, BASE, True, dtype)
        reference = ReferenceGemma4RotaryEmbedding(*args).npu()
        candidate = AscendGemma4RotaryEmbedding(*args).npu()
        assert candidate.rotary_dim == HEAD_SIZE
        torch.testing.assert_close(candidate.cos_sin_cache, reference.cos_sin_cache, rtol=0, atol=0)

        # Noncontiguous Q/K views of a merged QKV projection.
        qkv = torch.randn(tokens, (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_SIZE, dtype=dtype, device="npu")
        query, key, _ = qkv.split([NUM_HEADS * HEAD_SIZE, NUM_KV_HEADS * HEAD_SIZE, NUM_KV_HEADS * HEAD_SIZE], dim=-1)
        positions = torch.tensor([0, 1, 4095, MAX_POSITION - 1], device="npu").repeat((tokens + 3) // 4)[:tokens]
        inputs = (query,) if query_only else (query, key)
        expected = [_expected(x, reference.cos_sin_cache, positions, True) for x in inputs]
        originals = [x.clone() for x in inputs]

        actual_query, actual_key = candidate(positions, query, None if query_only else key)
        if query_only:
            assert actual_key is None
        actual = [actual_query] if query_only else [actual_query, actual_key]
        tolerance = {torch.bfloat16: 1.6e-2, torch.float16: 1e-3}[dtype]
        for output, target, original in zip(actual, expected, originals):
            torch.testing.assert_close(output, target, atol=tolerance, rtol=tolerance)
            # Non-rotated angle pairs have cos=1 and sin=0, so they pass through exactly.
            heads = output.view(tokens, -1, HEAD_SIZE)
            source = original.view(tokens, -1, HEAD_SIZE)
            for start in (ROTARY_DIM // 2, HEAD_SIZE // 2 + ROTARY_DIM // 2):
                stop = start + (HEAD_SIZE - ROTARY_DIM) // 2
                torch.testing.assert_close(heads[..., start:stop], source[..., start:stop], rtol=0, atol=0)
