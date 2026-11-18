# SPDX-License-Identifier: Apache-2.0
"""Public packed-combine coverage with independent CPU packing and FP64 math."""

import pytest
import torch

from vllm_ascend.ops.triton.dcp import dcp_a2a


def make_inputs(shape, edge=False):
    tokens, heads, dim, world, has_local, head_major = shape
    generator = torch.Generator().manual_seed(20878 + tokens)
    values = torch.randn(world, tokens, heads, dim, generator=generator).bfloat16()
    lses = torch.randn(world, tokens, heads, generator=generator) * 4
    local = torch.randn(tokens, heads, dim, generator=generator).bfloat16()
    local_lse = torch.randn(tokens, heads, generator=generator) * 4
    if edge:
        lses[:, 0, 0] = -torch.inf
        local_lse[0, 0] = torch.nan
        values[:, 0, 0] = torch.nan
        local[0, 0] = torch.inf
        lses[0, -1, 0] = torch.nan
        lses[-1, -1, -1] = torch.inf
        values[0, -1, 0] = torch.nan
        lses[:, 2, 0], lses[:, 3, 0] = 200000.0, -200000.0
        local_lse[2, 0], local_lse[3, 0] = 200000.0, -200000.0
    raw = lses.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exponent, fraction = (raw >> 23) & 255, raw & 0x7FFFFF
    finite = exponent < 255
    significand = fraction | torch.where(exponent == 0, 0, 0x800000)
    significand = torch.where(finite, significand, 0)
    code = (torch.maximum(exponent, torch.ones_like(exponent)) + 1).float()
    code = torch.where(raw >> 31 != 0, -code, code)
    code = torch.where((raw & 0x7FFFFFFF) == 0, 128.0, code)
    code = torch.where(finite, code, 0.0)
    fields = torch.stack(
        (
            code,
            ((significand >> 16) & 255).float(),
            ((significand >> 8) & 255).float(),
            (significand & 255).float(),
        ),
        -1,
    ).bfloat16()
    recv = torch.cat((values, fields), -1).permute(0, 2, 1, 3).contiguous().npu()
    all_lses = torch.cat((lses, local_lse[None]), 0) if has_local else lses
    all_values = torch.cat((values, local[None]), 0) if has_local else values
    valid = torch.isfinite(all_lses)
    maximum = torch.where(valid, all_lses.double(), -torch.inf).max(0).values
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    weights = torch.where(valid, torch.exp(all_lses.double() - maximum), 0.0)
    numerator = (torch.where(valid[..., None], all_values.double(), 0.0) * weights[..., None]).sum(0)
    denominator = weights.sum(0)
    expected = numerator / torch.where(denominator > 0, denominator, 1.0)[..., None]
    if head_major:
        local = local.transpose(0, 1).contiguous().npu().transpose(0, 1)
    else:
        local = local.npu()
    return (
        recv,
        local if has_local else None,
        local_lse.unsqueeze(-1).npu() if has_local else None,
        expected,
    )


@pytest.mark.parametrize(
    "shape,edge",
    [
        pytest.param((5, 1, 1, 1, False, False), False, id="single-rank-single-feature"),
        pytest.param((114, 8, 512, 8, False, False), True, id="received-only-invalid-lse"),
        pytest.param((144, 12, 512, 8, True, False), False, id="token-major-local"),
        pytest.param((41, 7, 384, 5, True, True), True, id="head-major-local-rank-and-feature-tails"),
    ],
)
@torch.inference_mode()
def test_public_combine_row_weights(shape, edge, monkeypatch):
    recv, local, local_lse, expected = make_inputs(shape, edge)
    launches = []
    launch = dcp_a2a._launch_kernel

    def record_launch(fn, *args, **kwargs):
        launches.append(fn.__name__)
        return launch(fn, *args, **kwargs)

    monkeypatch.setattr(dcp_a2a, "_launch_kernel", record_launch)
    actual = dcp_a2a.fused_dcp_lse_combine(recv, shape[2], 1, local_output=local, local_lse=local_lse).cpu()
    error = (actual.double() - expected).abs()
    assert torch.isfinite(actual).all()
    assert (error <= 2e-5 + 0.004 * expected.abs()).double().mean() >= 0.99
    assert error.max() <= 0.02
    assert launches == ["_fused_dcp_lse_combine_head_scatter_kernel"]
