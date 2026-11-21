# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""NPU comparisons for the Ascend PLE speculative (MTP) short convolution."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.models.qwen4_exp.common.ops.ple import ple_conv as upstream_ple_conv
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.ops.triton_ple import ple_conv_spec


def _check_output_accuracy(expected: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    atol = rtol = 2**-9 if expected.dtype == torch.float16 else 2**-6
    max_abs_limit = 0.1 if expected.dtype == torch.float16 else 1.0
    expected_fp32 = expected.float()
    abs_error = (expected_fp32 - actual.float()).abs()
    within_tolerance = abs_error <= atol + rtol * expected_fp32.abs()
    pass_ratio = torch.count_nonzero(within_tolerance).item() / expected.numel()
    max_abs = abs_error.max().item()
    assert pass_ratio >= 0.99, f"only {pass_ratio:.6%} of elements meet the mixed tolerance"
    assert max_abs <= max_abs_limit, f"max absolute error {max_abs} exceeds {max_abs_limit}"
    return pass_ratio, max_abs


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "channels,kernel_size,dilation,spec_query_len",
    [
        (256, 4, 3, 3),
        (1031, 3, 2, 2),  # channel tail
        (257, 4, 1, 4),
    ],
)
@pytest.mark.parametrize("mapped,outer_enabled", [(True, True), (False, False)])
def test_spec_matches_upstream(dtype, channels, kernel_size, dilation, spec_query_len, mapped, outer_enabled):
    torch.manual_seed(2026)
    num_reqs = 5
    tokens = num_reqs * spec_query_len
    state_len = (kernel_size - 1) * dilation
    state_width = state_len + spec_query_len - 1
    inputs = torch.randn(tokens, channels, device="npu", dtype=dtype)
    weights = torch.randn(channels, kernel_size, device="npu", dtype=dtype)
    # Ascend kernels use the SD layout [slots, window, channels].
    state = torch.randn(num_reqs + 1, state_width, channels, device="npu", dtype=dtype)
    residual = torch.randn_like(inputs)
    outer = torch.randn_like(inputs) if outer_enabled else None
    slots = torch.arange(1, num_reqs + 1, device="npu", dtype=torch.int32)
    slots[-1] = NULL_BLOCK_ID
    query_start_loc = torch.arange(0, tokens + 1, spec_query_len, device="npu", dtype=torch.int32)
    num_accepted = (torch.arange(num_reqs, device="npu", dtype=torch.int32) % spec_query_len) + 1
    token_indices = torch.arange(tokens - 1, -1, -1, device="npu", dtype=torch.int32) if mapped else None
    kwargs = dict(
        inputs=inputs,
        conv_weights=weights,
        state_indices=slots,
        outer_residual=outer,
        dilation=dilation,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted,
        spec_query_len=spec_query_len,
        token_indices=token_indices,
    )
    original_inputs, original_weights = inputs.clone(), weights.clone()
    expected, actual = residual.clone(), residual.clone()
    expected_state, actual_state = state.clone(), state.clone()
    upstream_ple_conv(
        **kwargs,
        residual=expected,
        conv_state=expected_state.transpose(1, 2),
        mode="spec",
    )
    ple_conv_spec(**kwargs, residual=actual, conv_state=actual_state)
    torch.npu.synchronize()

    pass_ratio, max_abs = _check_output_accuracy(expected, actual)
    state_exact = torch.equal(expected_state, actual_state)
    print(
        f"SPEC dtype={dtype} C={channels} K={kernel_size} dilation={dilation} nspec={spec_query_len} "
        f"mapped={mapped} outer={outer_enabled} pass_ratio={pass_ratio:.6%} max_abs={max_abs:.6f} "
        f"state_exact={state_exact}",
        flush=True,
    )
    assert state_exact
    assert torch.equal(inputs, original_inputs)
    assert torch.equal(weights, original_weights)
