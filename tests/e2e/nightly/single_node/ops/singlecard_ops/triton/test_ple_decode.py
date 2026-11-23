# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Additional PLE decode shapes and argument categories against upstream."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.models.qwen4_exp.common.ops.ple import ple_conv as upstream_ple_conv
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.ops.triton_ple import ple_conv_decode


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tokens,channels,kernel_size,dilation,mapped,strided_indices,window_contiguous",
    [
        (7, 1031, 4, 3, True, True, False),  # channel tail and mixed-token map
        (5, 256, 2, 1, False, False, True),  # one-element state window
        (6, 257, 3, 2, False, True, True),  # different dilation and state layout
    ],
)
def test_decode_parameter_categories(
    dtype, tokens, channels, kernel_size, dilation, mapped, strided_indices, window_contiguous
):
    torch.manual_seed(1024)
    state_len = (kernel_size - 1) * dilation
    inputs = torch.randn(tokens, channels, device="npu", dtype=dtype)
    weights = torch.randn(channels, kernel_size, device="npu", dtype=dtype)
    # Ascend kernels use the SD layout [slots, window, channels].
    if window_contiguous:
        state = torch.randn(tokens + 1, channels, state_len, device="npu", dtype=dtype).transpose(1, 2)
    else:
        state = torch.randn(tokens + 1, state_len, channels, device="npu", dtype=dtype)
    output = torch.randn_like(inputs)
    outer = torch.randn_like(inputs) if mapped else None
    ids = torch.arange(1, tokens + 1, device="npu", dtype=torch.int32)
    if strided_indices:
        padded_ids = torch.zeros(tokens, 2, device="npu", dtype=torch.int32)
        padded_ids[:, 0] = ids
        ids = padded_ids[:, 0]
    ids[-1] = NULL_BLOCK_ID
    has_initial_states = torch.tensor([i % 2 == 0 for i in range(tokens)], device="npu")
    token_indices = torch.arange(tokens - 1, -1, -1, device="npu", dtype=torch.int32) if mapped else None
    kwargs = dict(
        inputs=inputs,
        conv_weights=weights,
        state_indices=ids,
        outer_residual=outer,
        dilation=dilation,
        has_initial_states=has_initial_states,
        token_indices=token_indices,
    )
    original_inputs, original_weights = inputs.clone(), weights.clone()
    expected, actual = output.clone(), output.clone()
    expected_state, actual_state = state.clone(), state.clone()
    upstream_ple_conv(**kwargs, residual=expected, conv_state=expected_state.transpose(1, 2), mode="decode")
    ple_conv_decode(**kwargs, residual=actual, conv_state=actual_state)
    torch.npu.synchronize()

    error = (actual.float() - expected.float()).abs()
    rtol = 2**-6 if dtype is torch.bfloat16 else 2**-9
    atol = rtol
    max_limit = 1.0 if dtype is torch.bfloat16 else 0.1
    assert (error <= atol + rtol * expected.float().abs()).float().mean().item() >= 0.99
    assert error.max().item() <= max_limit
    assert torch.equal(actual_state, expected_state)
    assert torch.equal(inputs, original_inputs)
    assert torch.equal(weights, original_weights)
