# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU reference checks for narrow output slices and adapter isolation."""

import pytest
import torch

from vllm_ascend.lora.torch_ops import expand_slice


@pytest.mark.parametrize("add_inputs", [False, True])
@pytest.mark.parametrize("tokens", [0, 1, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("four_dims", [False, True])
def test_narrow_slice_preserves_other_columns_and_adapter_isolation(add_inputs, tokens, dtype, four_dims):
    torch.manual_seed(5301)
    inputs = torch.randn(tokens, 8).to(dtype)
    weights = torch.randn(2, 4, 8).to(dtype)
    if four_dims:
        weights = weights[:, None]
    base = torch.randn(tokens, 11).to(dtype)
    for slot in (0, -1, 1, 0):
        output, expected = base.clone(), base.clone()
        if slot >= 0:
            factor = weights[slot].reshape(4, 8)
            delta = inputs.float() @ factor.float().T
            expected[:, 3:7] = expected[:, 3:7].float() + delta if add_inputs else delta
        expand_slice(output, inputs, weights, torch.full((tokens,), slot), 3, 4, add_inputs)
        torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("add_inputs", [False, True])
def test_unselected_adapter_nan_does_not_contaminate_narrow_slice(add_inputs):
    output = torch.arange(5, dtype=torch.float32).view(1, 5)
    expected = output.clone()
    expand_slice(output, torch.ones(1, 8), torch.full((2, 1, 4, 8), float("nan")), torch.tensor([-1]), 1, 4, add_inputs)
    assert torch.equal(output, expected)


@pytest.mark.parametrize("add_inputs", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fp32_shrink_preserves_precision_and_inactive_rows(dtype, add_inputs):
    inputs = torch.tensor([[1.001, -1.0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    weights = torch.ones(2, 1, 4, 8, dtype=dtype)
    weights[1].mul_(2)
    base = torch.arange(6, dtype=dtype).view(1, 6) / 100
    for slot in (0, -1, 1):
        output, expected = base.clone(), base.clone()
        if slot >= 0:
            delta = inputs.double() @ weights[slot, 0].double().T
            expected[:, 1:5] = expected[:, 1:5].double() + delta if add_inputs else delta

        expand_slice(output, inputs, weights, torch.tensor([slot]), 1, 4, add_inputs)

        torch.testing.assert_close(output, expected)
        if slot < 0:
            torch.testing.assert_close(output, base, rtol=0, atol=0)
