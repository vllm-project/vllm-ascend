# SPDX-License-Identifier: Apache-2.0
#forked https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_cascade_flash_attn.py
#test https://github.com/vllm-project/vllm/blob/main/vllm/attention/backends/mla/common.py#L1263-L1270

from typing import Optional

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_attn import (cascade_attention,
                                                   merge_attn_states)
from vllm.vllm_flash_attn import (fa_version_unsupported_reason,
                                  flash_attn_varlen_func,
                                  is_fa_version_supported)

NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
HEAD_SIZES = [128, 192, 256]
BLOCK_SIZES = [16]
DTYPES = [torch.float16, torch.bfloat16]


def merge_attn_states_torch(
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
        output_lse: Optional[torch.Tensor] = None,
    ) -> None:
    # Reference implementation.
    dtype = prefix_output.dtype
    max_lse = torch.maximum(prefix_lse, suffix_lse)
    p_lse = torch.exp(prefix_lse - max_lse)
    s_lse = torch.exp(suffix_lse - max_lse)
    p_scale = p_lse / (p_lse + s_lse)
    s_scale = s_lse / (p_lse + s_lse)
    p_scale = p_scale.transpose(0, 1).unsqueeze(2)
    s_scale = s_scale.transpose(0, 1).unsqueeze(2)
    output = p_scale * prefix_output + s_scale * suffix_output
    output = output.to(dtype)
    return output

@pytest.mark.parametrize("num_tokens", [1, 39, 16912])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_merge_kernel(
    num_tokens: int,
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0

    # Prepare inputs.
    prefix_output = torch.randn(num_tokens,
                                num_query_heads,
                                head_size,
                                dtype=dtype)
    suffix_output = torch.randn(num_tokens,
                                num_query_heads,
                                head_size,
                                dtype=dtype)
    prefix_lse = torch.randn(num_query_heads, num_tokens, dtype=torch.float32)
    suffix_lse = torch.randn(num_query_heads, num_tokens, dtype=torch.float32)

    # Run the kernel.
    output = torch.empty(num_tokens, num_query_heads, head_size, dtype=dtype)
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                      suffix_lse)

    '''
    # Reference implementation.
    max_lse = torch.maximum(prefix_lse, suffix_lse)
    p_lse = torch.exp(prefix_lse - max_lse)
    s_lse = torch.exp(suffix_lse - max_lse)
    p_scale = p_lse / (p_lse + s_lse)
    s_scale = s_lse / (p_lse + s_lse)
    p_scale = p_scale.transpose(0, 1).unsqueeze(2)
    s_scale = s_scale.transpose(0, 1).unsqueeze(2)
    ref_output = p_scale * prefix_output + s_scale * suffix_output
    ref_output = ref_output.to(dtype)
    '''

    ref_output = merge_attn_states_torch(prefix_output, prefix_lse, suffix_output, suffix_lse)
   
    # Compare the results.
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)

