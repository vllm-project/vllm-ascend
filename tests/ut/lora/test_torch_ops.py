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
        if not add_inputs:
            expected[:, 3:7].zero_()
        if slot >= 0:
            factor = weights[slot].reshape(4, 8)
            expected[:, 3:7] = expected[:, 3:7].float() + inputs.float() @ factor.float().T
        expand_slice(output, inputs, weights, torch.full((tokens,), slot), 3, 4, add_inputs)
        torch.testing.assert_close(output, expected)


def test_unselected_adapter_nan_does_not_contaminate_narrow_slice():
    output = torch.zeros(1, 5)
    expand_slice(output, torch.ones(1, 8), torch.full((2, 1, 4, 8), float("nan")), torch.tensor([-1]), 1, 4, True)
    assert torch.equal(output, torch.zeros_like(output))
