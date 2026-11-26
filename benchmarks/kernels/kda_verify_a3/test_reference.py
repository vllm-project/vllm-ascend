# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression tests; run with pytest, without torch-npu or TileLang."""

import pytest
import torch
from demo import fixture
from reference import oracle, prepare, prepared_reference


def test_key_major_checkpoint_and_nonzero_correction():
    q = torch.zeros(1, 1, 128)
    q[0, 0, 1] = 128**-0.5
    k = torch.zeros_like(q)
    k[0, 0, 1] = 1
    v = torch.zeros_like(q)
    v[0, 0, 7] = 2
    decay = torch.ones_like(q)
    decay[0, 0, 1] = 0.5
    beta = torch.full((1, 1), 0.25)
    state = torch.zeros(1, 128, 128, dtype=torch.float16)
    state[0, 1, 7] = 3  # key=1, value=7; transposing S would change the result.
    before = state.clone()
    args = (q, k, v, decay, beta, state)
    expected_corr = torch.zeros_like(q, dtype=torch.float64)
    expected_corr[0, 0, 7] = 0.25 * (2 - 0.5 * 3)
    expected_out = torch.zeros_like(expected_corr)
    expected_out[0, 0, 7] = (1.5 + 0.125) * q[0, 0, 1].double()
    for result in (oracle(*args), prepared_reference(*prepare(*args), 1)):
        torch.testing.assert_close(result[0].double(), expected_corr, rtol=0, atol=0)
        torch.testing.assert_close(result[1].double(), expected_out, rtol=0, atol=2e-5)
    torch.testing.assert_close(state, before, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", range(1, 9))
@pytest.mark.parametrize("seed", (1, 731))
def test_zero_decay_and_prefix_consistency(tokens, seed):
    args = list(fixture(2, tokens, seed))
    args[3][:, tokens // 2, ::2] = 0  # Prefix-product division would produce 0/0.
    args[-1] = args[-1].transpose(1, 2)  # Exercise noncontiguous checkpoint input.
    before = args[-1].clone()
    expected = oracle(*args)
    prepared = prepare(*args)
    actual = prepared_reference(*prepared, tokens)
    assert all(tensor.is_contiguous() for tensor in prepared)
    assert expected[0].abs().max() > 0.1
    for got, want in zip(actual, expected, strict=True):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got.double(), want, rtol=2e-3, atol=1e-3)
    for prefix in range(1, tokens + 1):
        short = [tensor[:, :prefix] for tensor in args[:-1]] + [args[-1]]
        short_oracle = oracle(*short)
        short_prepared = prepared_reference(*prepare(*short), prefix)
        for full, shorter in zip(expected, short_oracle, strict=True):
            torch.testing.assert_close(full[:, :prefix], shorter, rtol=0, atol=0)
        for full, shorter in zip(actual, short_prepared, strict=True):
            torch.testing.assert_close(full[:, :prefix], shorter, rtol=0, atol=0)
    torch.testing.assert_close(args[-1], before, rtol=0, atol=0)
