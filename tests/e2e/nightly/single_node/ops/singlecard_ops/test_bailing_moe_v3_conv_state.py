# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real FLA regressions for Bailing MRV2 convolution warmup and replay."""

import pytest
import torch
import torch_npu  # noqa: F401
from torch.nn import functional as F
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from vllm_ascend.ops.bailing_moe_v3_kda import AscendBailingMoeV3KimiDeltaAttention


@pytest.mark.parametrize("layout", ["sd", "ds", "page_strided"])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("capture", [False, True])
@torch.inference_mode()
def test_causal_conv_null_warmup_then_active_packed_row_zero(layout, cache_dtype, capture):
    """Null/padded requests must not access state; packed row zero must run."""
    requests, query_len, slots, state_len, dim = 32, 4, 4, 6, 1536
    attention = AscendBailingMoeV3KimiDeltaAttention.__new__(AscendBailingMoeV3KimiDeltaAttention)
    torch.nn.Module.__init__(attention)
    attention._conv_state_dim_first = layout == "ds"
    state_shape = (dim, state_len) if layout == "ds" else (state_len, dim)
    backing = torch.ones((slots * 2, *state_shape), dtype=cache_dtype, device="npu")
    storage = backing[::2] if layout == "page_strided" else backing[:slots]
    original = backing.cpu()
    mixed_qkv = torch.full((requests * query_len, dim), 0.25, dtype=torch.bfloat16, device="npu")
    weights = torch.zeros((4, dim), dtype=mixed_qkv.dtype, device="npu")
    weights[-1].fill_(1)  # Only the current token contributes to the reference.
    starts = torch.arange(0, (requests + 1) * query_len, query_len, dtype=torch.int32, device="npu")
    indices = torch.full((requests, query_len), NULL_BLOCK_ID, dtype=torch.int32, device="npu")
    accepted = torch.full((requests,), query_len, dtype=torch.int32, device="npu")

    def run():
        return attention._run_causal_conv1d(
            mixed_qkv,
            weights,
            storage,
            starts,
            indices,
            None,
            run_mode=1,
            max_query_len=query_len,
            num_accepted_tokens=accepted,
        )

    # Exercise the real gather, FLA update and scatter with MRV2's all-null input.
    output = run()
    torch.testing.assert_close(output.cpu(), torch.zeros_like(output.cpu()), atol=0, rtol=0)
    torch.testing.assert_close(backing.cpu(), original, atol=0, rtol=0)
    if capture:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = run()

    # Change device metadata after capture: request zero becomes packed row zero,
    # while explicit padding and reserved null slots must remain inactive.
    indices[0, 0] = 2
    indices[2, 0] = PAD_SLOT_ID
    indices[3, 0] = 3
    if capture:
        graph.replay()
    else:
        output = run()

    expected_output = torch.zeros_like(mixed_qkv.cpu())
    for request in (0, 3):
        expected_output[request * query_len : (request + 1) * query_len] = F.silu(
            mixed_qkv[request * query_len : (request + 1) * query_len].float().cpu()
        ).to(mixed_qkv.dtype)
    torch.testing.assert_close(output.cpu(), expected_output, atol=2e-3, rtol=2e-2)
    expected_storage = original[::2] if layout == "page_strided" else original[:slots]
    expected_state = expected_storage.transpose(-1, -2) if layout == "ds" else expected_storage
    expected_state[2:4, -query_len:].fill_(0.25)
    # Includes null slot zero, unrelated slots and gaps between strided pages.
    torch.testing.assert_close(backing.cpu(), original, atol=0, rtol=0)
