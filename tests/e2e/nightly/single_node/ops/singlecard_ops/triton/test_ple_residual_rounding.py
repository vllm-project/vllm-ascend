# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""PLE residuals must preserve eager low-precision materialization boundaries."""

from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from vllm_ascend.ops import triton_ple


def _expected_state(initial, inputs, slots, token_rows, lengths, mode, has_initial, accepted):
    """Reference each request's history independently of the fused kernel."""
    result = initial.clone()
    history_length = 9  # (kernel_size - 1) * dilation = 3 * 3.
    start = 0
    for request, (slot, length) in enumerate(zip(slots, lengths)):
        rows = token_rows[start : start + length]
        start += length
        if mode == "spec":
            offset = accepted[request] - 1
            previous = initial[slot, offset : offset + history_length]
            history = torch.cat((previous, inputs[rows]), dim=0)
            # Preserve rollback choices for the next speculative step. The
            # remaining capacity of a short chunk must stay untouched.
            result[slot, : history_length + length - 1] = history[1:]
        else:
            previous = initial[slot, :history_length]
            if not has_initial[request]:
                previous = torch.zeros_like(previous)
            history = torch.cat((previous, inputs[rows]), dim=0)
            result[slot, :history_length] = history[-history_length:]
    return result


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "mode,accepted_count",
    [("decode", None), ("prefill", None), ("spec", 1), ("spec", 2), ("spec", 4)],
)
@pytest.mark.parametrize("mapped", [False, True])
def test_ple_residual_rounding_and_state(dtype, mode, accepted_count, mapped):
    channels, kernel_size, dilation, spec_query_len = 17, 4, 3, 4
    lengths = [1, 1, 1] if mode == "decode" else ([2, 3] if mode == "prefill" else [2, 4])
    logical_tokens = sum(lengths)
    physical_tokens = logical_tokens + 2 if mapped else logical_tokens
    slots = [3, 1, 4] if mode == "decode" else [3, 1]
    token_rows = list(range(1, logical_tokens + 1))[::-1] if mapped else list(range(logical_tokens))
    has_initial = [request % 2 == 0 for request in range(len(slots))]
    accepted = [accepted_count, spec_query_len + 1 - accepted_count] if mode == "spec" else None

    # Channel zero is the exact rounding witness: gated=1, hidden=-1,
    # conv=BF16/FP16(SiLU(1)). Other channels vary by physical row so the
    # token map and request-slot permutation affect the history reference.
    inputs_cpu = (
        torch.arange(physical_tokens * channels, device="cpu").reshape(physical_tokens, channels).remainder(19).float()
        / 8
        - 1
    ).to(dtype)
    inputs_cpu[:, 0] = 1
    weights_cpu = torch.zeros(channels, kernel_size, dtype=dtype, device="cpu")
    weights_cpu[:, -1] = 1
    initial_cpu = (
        torch.arange(5 * 12 * channels, device="cpu").reshape(5, 12, channels).remainder(31).float() / 16 - 1
    ).to(dtype)
    gated_cpu = torch.ones_like(inputs_cpu)
    hidden_cpu = -torch.ones_like(inputs_cpu)

    inputs = inputs_cpu.to("npu")
    weights = weights_cpu.to("npu")
    outer = hidden_cpu.to("npu")
    # Also retain the strided block-table-column state-index contract.
    indices = torch.zeros(len(slots), 2, dtype=torch.int32, device="npu")
    indices[:, 0] = torch.tensor(slots, dtype=torch.int32, device="npu")
    kwargs = dict(inputs=inputs, conv_weights=weights, state_indices=indices[:, 0], dilation=dilation)
    if mapped:
        kwargs["token_indices"] = torch.tensor(token_rows, dtype=torch.int32, device="npu")
    if mode == "spec":
        kwargs.update(
            query_start_loc=torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device="npu"),
            num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32, device="npu"),
            spec_query_len=spec_query_len,
        )
    else:
        kwargs["has_initial_states"] = torch.tensor(has_initial, dtype=torch.bool, device="npu")
        if mode == "prefill":
            kwargs["query_start_loc"] = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device="npu")

    conv_only = torch.zeros_like(inputs)
    actual = gated_cpu.to("npu")
    conv_state = initial_cpu.to("npu")
    fused_state = conv_state.clone()
    launcher = getattr(triton_ple, f"ple_conv_{mode}")
    launcher(**kwargs, residual=conv_only, conv_state=conv_state, outer_residual=None)
    launcher(**kwargs, residual=actual, conv_state=fused_state, outer_residual=outer)
    torch.npu.synchronize()

    # Use separate eager CPU additions as the rounding reference. An upstream
    # fused kernel can share the same missing-rounding defect, so it is not a
    # suitable reference for this regression. Reuse only the materialized
    # conv result to isolate residual rounding from convolution tolerances.
    conv_cpu = conv_only.cpu()
    expected = gated_cpu.clone()
    expected[token_rows] = (gated_cpu[token_rows] + conv_cpu[token_rows]) + hidden_cpu[token_rows]
    witness_conv = F.silu(torch.tensor(1.0, device="cpu")).to(dtype)
    assert torch.equal(conv_cpu[token_rows, 0], witness_conv.expand(logical_tokens))
    reassociated = (gated_cpu[token_rows] + hidden_cpu[token_rows]) + conv_cpu[token_rows]
    assert torch.any(expected[token_rows] != reassociated), "fixture must distinguish the two rounding orders"
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    expected_state = _expected_state(initial_cpu, inputs_cpu, slots, token_rows, lengths, mode, has_initial, accepted)
    torch.testing.assert_close(conv_state.cpu(), expected_state, rtol=0, atol=0)
    torch.testing.assert_close(fused_state.cpu(), expected_state, rtol=0, atol=0)
    torch.testing.assert_close(inputs.cpu(), inputs_cpu, rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), weights_cpu, rtol=0, atol=0)
    torch.testing.assert_close(outer.cpu(), hidden_cpu, rtol=0, atol=0)
