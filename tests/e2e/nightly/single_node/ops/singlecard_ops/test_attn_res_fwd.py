# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu


def cpu_reference(prefix, addend, blocks, projection, gamma, valid, output_gamma, epsilon, mix):
    raw = (prefix.cpu() + addend.cpu()) if addend is not None else prefix.cpu()
    if mix and valid:
        values = torch.cat((blocks[:, :valid].cpu(), raw.unsqueeze(1)), dim=1).float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
        logits = (normalized * gamma.cpu().float() * projection.cpu().float()).sum(-1)
        materialized = (logits.softmax(-1).unsqueeze(-1) * values).sum(1).bfloat16()
    else:
        materialized = raw

    output = materialized
    if output_gamma is not None:
        value = materialized.float()
        output = (
            value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon) * output_gamma.cpu().float()
        ).bfloat16()
    return output, raw, materialized


def assert_precision(actual, expected):
    error = (actual.float().cpu() - expected.float()).abs()
    assert torch.isfinite(actual).all()
    assert (error <= (1 + expected.float().abs()) / 64).float().mean() >= 0.99
    assert error.max() <= 1


@pytest.mark.parametrize(
    "tokens,valid,with_add,with_output_norm,optimize_prefill,mix",
    [
        (1, 0, True, False, False, False),
        (1, 1, False, False, False, True),
        (16, 4, True, True, False, True),
        (128, 8, True, True, False, True),
        (625, 8, True, True, True, True),
        (2048, 8, False, True, True, True),
    ],
)
@torch.inference_mode()
def test_attn_res_fwd_precision(tokens, valid, with_add, with_output_norm, optimize_prefill, mix):
    hidden = 7168
    torch.manual_seed(42 + tokens + valid)
    prefix = torch.randn(tokens, hidden, device="npu", dtype=torch.bfloat16)
    addend = torch.randn_like(prefix) if with_add else None
    bank = torch.randn(tokens, 8, hidden, device="npu", dtype=torch.bfloat16)
    bank[:, valid:] = float("nan")
    projection = (torch.randn(1, hidden, device="npu") / hidden**0.5).bfloat16()
    gamma = torch.randn(hidden, device="npu", dtype=torch.bfloat16)
    output_gamma = torch.randn_like(gamma) if with_output_norm else None
    expected_output, expected_raw, expected_mix = cpu_reference(
        prefix, addend, bank, projection, gamma, valid, output_gamma, 1e-5, mix
    )
    write_idx = valid if valid < 8 else -1
    output, raw, materialized = torch.ops._C_ascend.attn_res_fwd(
        prefix,
        addend,
        bank,
        projection,
        gamma,
        1e-5,
        valid,
        output_gamma,
        1e-5,
        write_idx,
        True,
        mix,
        optimize_prefill=optimize_prefill,
    )
    torch_npu.npu.synchronize()

    torch.testing.assert_close(raw.cpu(), expected_raw, rtol=0, atol=0)
    assert_precision(materialized, expected_mix)
    assert_precision(output, expected_output)
    if write_idx >= 0:
        torch.testing.assert_close(bank[:, write_idx].cpu(), expected_raw, rtol=0, atol=0)
