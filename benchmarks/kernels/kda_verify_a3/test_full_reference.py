# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of complete projection, history, recurrence and output semantics."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from full_reference import (
    Config,
    full_reference,
    make_cache,
    make_inputs,
    make_rollback,
    prepare_middle,
    project_inputs,
)


def test_full_path_against_explicit_one_token_calculation():
    c = Config(batch=1, tokens=1, hidden=2, heads=1, dim=2, lower_bound=-2)
    i = make_inputs(c)
    i["x"][:] = torch.tensor([1, 2])
    i["w_in"].zero_()
    i["w_in"][:6] = torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0], [0.5, 0], [0, -1]])
    i["w_in"][6, 0], i["w_in"][7, 0], i["w_in"][10, 1] = 0.25, 1, 1
    i["w_f"][:] = torch.tensor([[1, 0], [-1, 0]])
    i["w_g"][:] = torch.tensor([[0, 0.5], [0, -0.5]])
    i["w_o"][:] = torch.tensor([[1, 2], [-1, 3]])
    i["conv_w"].zero_()
    i["conv_w"][..., 3] = 1
    i["history"].zero_()
    i["a_log"].zero_()
    i["dt_bias"].zero_()
    i["norm_w"][:] = torch.tensor([1, 2])
    i["state"][:] = torch.tensor([[0.3, 0.2], [0.5, -0.4]])  # [value,key]
    before = {name: value.clone() for name, value in i.items()}
    actual = full_reference(c, i, torch.ones(1, dtype=torch.int32))

    raw = torch.tensor([[1, 2], [2, 1], [0.5, -2]], dtype=torch.float32)
    q_raw, k_raw, v = F.silu(raw).bfloat16().double()
    q, k = q_raw / q_raw.norm() / 2**0.5, k_raw / k_raw.norm()
    gate = torch.tensor([1, -1], dtype=torch.float64)
    decayed = i["state"][0, 0].double() * (-2 * gate.sigmoid()).exp()[None, :]
    correction = torch.sigmoid(torch.tensor(0.25, dtype=torch.float64)) * (v - decayed @ k)
    state = decayed + correction[:, None] * k[None, :]
    core = (state @ q).bfloat16().float()
    normalized = core * torch.rsqrt(core.square().mean() + c.eps) * i["norm_w"].float()
    normalized = (normalized * gate.float().sigmoid()).bfloat16()
    expected_output = torch.tensor(
        [
            normalized[0].float() + 2 * normalized[1].float(),
            -normalized[0].float() + 3 * normalized[1].float(),
        ]
    ).bfloat16()
    torch.testing.assert_close(actual.correction[0, 0, 0].double(), correction, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(actual.normalized[0, 0], normalized, rtol=0, atol=0)
    torch.testing.assert_close(actual.output[0], expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual.kg[0, 0, 0], torch.cat((k_raw, gate)).bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(actual.history[0, -1], raw.flatten().bfloat16(), rtol=0, atol=0)
    for name in i:
        torch.testing.assert_close(i[name], before[name], rtol=0, atol=0)


@pytest.mark.parametrize("tokens", (1, 2, 6, 8))
def test_mixed_history_gates_padding_and_prefixes(tokens):
    c = Config(batch=2, tokens=tokens, hidden=16, heads=2, dim=8)
    i = make_inputs(c, seed=731)
    i["rollback"] = make_rollback(c, i)
    cache = make_cache(c, i)
    accepted = torch.tensor([1, tokens], dtype=torch.int32)
    result = full_reference(c, i, accepted, cache=cache)
    qkv, _, _, _ = project_inputs(c, i)
    for b, start in enumerate((0, tokens - 1)):
        selected = cache[b, start : start + 3]
        window = torch.cat((selected, qkv[b].reshape(tokens, -1)), dim=0)
        torch.testing.assert_close(result.history[b], window[1:], rtol=0, atol=0)
        weights = i["conv_w"].reshape(-1, 1, 4).float()
        convolved = F.silu(F.conv1d(window.T[None].float(), weights, groups=3 * c.heads * c.dim))
        k_raw = convolved[0].T.bfloat16().reshape(tokens, 3, c.heads, c.dim)[:, 1]
        torch.testing.assert_close(result.kg[b, :, :, : c.dim], k_raw.transpose(0, 1), rtol=0, atol=0)

    def gate_projector(projected):
        assert projected.shape == (c.batch * tokens, 3 * c.heads * c.dim + c.heads + 2 * c.dim)
        return F.linear(projected[:, -2 * c.dim : -c.dim], i["w_f"]), F.linear(projected[:, -c.dim :], i["w_g"])

    middle = prepare_middle(c, i, accepted, cache=cache)
    padded = prepare_middle(c, i, accepted, c.batch * tokens + 7, gate_projector, cache)
    for actual, expected in zip(padded, middle, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for length in range(1, tokens + 1):
        short = dict(i, x=i["x"][:, :length])
        prefix = full_reference(replace(c, tokens=length), short, accepted, cache=cache)
        torch.testing.assert_close(
            prefix.output.reshape(2, length, -1), result.output.reshape(2, tokens, -1)[:, :length]
        )
        torch.testing.assert_close(prefix.correction, result.correction[:, :, :length], rtol=0, atol=0)


def test_repeated_verify_updates_convolution_history_with_accept_one():
    c = Config(batch=1, tokens=3, hidden=16, heads=2, dim=8)
    i = make_inputs(c)
    cache = make_cache(c, i)
    original = cache.clone()
    accepted = torch.ones(1, dtype=torch.int32)
    first = full_reference(c, i, accepted, cache=cache)
    torch.testing.assert_close(cache, original, rtol=0, atol=0)
    cache.copy_(first.history)
    second = full_reference(c, i, accepted, cache=cache)
    assert not torch.equal(second.output, first.output)
    torch.testing.assert_close(second.history[:, :2], first.history[:, 1:3], rtol=0, atol=0)
