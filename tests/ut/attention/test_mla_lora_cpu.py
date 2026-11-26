# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU reference checks for absorbed MLA query/value adapter weights."""

import pytest
import torch

from vllm_ascend.attention.mla_projections import add_mla_lora


@pytest.mark.parametrize("is_query", [False, True])
@pytest.mark.parametrize("tokens", [0, 1, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("zero", [False, True])
def test_absorbed_lora_matches_materialized_weight(is_query, tokens, dtype, zero):
    torch.manual_seed(530)
    heads, qdim, vdim, latent, rank = 2, 4, 3, 5, 2
    a = torch.randn(2, 1, rank, latent).to(dtype)
    b = torch.randn(2, 1, heads * (qdim + vdim), rank).to(dtype) * 0.1
    if zero:
        b.zero_()
    x = torch.randn(heads, tokens, qdim if is_query else latent).to(dtype)
    base = torch.randn(tokens, heads, latent if is_query else vdim).to(dtype)
    for slot in (0, -1, 1, 0):
        # One request at a time; switch adapters between requests.
        indices = torch.full((tokens + 3,), slot, dtype=torch.long)
        actual = base.clone()
        add_mla_lora(x, actual, a, b, indices, qdim, is_query=is_query)
        expected = base.double()
        if slot >= 0:
            delta = b[slot, 0].double().view(heads, qdim + vdim, rank) @ a[slot, 0].double()
            if is_query:
                expected += torch.einsum("htq,hqc->thc", x.double(), delta[:, :qdim])
            else:
                expected += torch.einsum("htc,hvc->thv", x.double(), delta[:, qdim:])
        torch.testing.assert_close(actual.double(), expected, atol=0.04 if dtype == torch.bfloat16 else 2e-6, rtol=0.02)
        if slot < 0 or zero:
            torch.testing.assert_close(actual, base, atol=0, rtol=0)


def test_unselected_adapter_nan_does_not_contaminate_absorbed_output():
    inputs = torch.ones(2, 1, 5)
    a = torch.full((2, 1, 2, 5), float("nan"))
    b = torch.ones(2, 1, 14, 2)
    output = torch.zeros(1, 2, 3)
    add_mla_lora(inputs, output, a, b, torch.tensor([-1]), 4, is_query=False)
    assert torch.equal(output, torch.zeros_like(output))
