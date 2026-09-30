# SPDX-License-Identifier: Apache-2.0
"""Single-card tests for the optional ``group_index`` input of
``dequant_situ_quant`` on the BF16 path.

The BF16 (already-dequantized) routed path used to reject ``group_index``
and therefore scanned the whole worst-case-expanded MoE buffer row by row,
including the dead tail rows that no expert owns. With ``group_index``
accepted, the kernel iterates per-expert row groups and skips the dead tail.

These tests pin the equivalence contract:
  * passing counts whose sum equals the row length is bit-identical to
    passing ``None`` (full scan);
  * counts whose sum is smaller only touch the leading active rows;
  * counts whose sum exceeds the row length (padded tail-chunk buffers)
    clamp safely inside the kernel group loop;
  * outputs match a float32 reference within 1 int8 LSB.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401,E402  (registers torch.ops._C_ascend)

OP = torch.ops._C_ascend.dequant_situ_quant

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")


def _run(x, group_index=None):
    y, scale = OP(
        x=x,
        weight_scale=None,
        activation_scale=None,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=group_index,
        beta=1.0,
        linear_beta=0.0,
        activate_left=True,
        quant_mode="dynamic",
    )
    torch.npu.synchronize()
    return y, scale


def _counts(num_groups, total, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(num_groups, generator=g) + 0.2
    c = (w / w.sum() * total).to(torch.int64)
    c[0] += total - c.sum()
    assert int(c.sum()) == total
    return c.to(device)


def _golden(x_bf16):
    x = x_bf16.to(torch.float32)
    half = x.shape[-1] // 2
    gate, up = x[..., :half], x[..., half:]
    g = torch.tanh(gate) * torch.sigmoid(gate)
    y = g * up
    absmax = y.abs().amax(dim=-1, keepdim=True)
    scale = absmax / 127.0
    inv = torch.where(scale > 0, 1.0 / scale, torch.ones_like(scale))
    y_int8 = (y * inv).round().clamp(-127, 127).to(torch.int8)
    # 1-D, matching the op's per-row scale output; a [rows, 1] return would
    # silently broadcast against the op's [rows] in the assertions below.
    return y_int8, scale.view(-1)


def _bf16_x(rows, cols, seed):
    torch.npu.manual_seed_all(seed)
    return (torch.randn(rows, cols, device="npu:0") * 2.0).to(torch.bfloat16)


@pytest.mark.parametrize("rows,cols,groups", [(896, 6144, 14), (528, 6144, 14), (64, 128, 8)])
def test_group_counts_equal_rows_matches_full_scan(rows, cols, groups):
    """sum(counts) == rows: group path must be bit-identical to None."""
    x = _bf16_x(rows, cols, seed=1)
    counts = _counts(groups, rows, x.device)
    y_none, s_none = _run(x, None)
    y_grp, s_grp = _run(x, counts)
    assert torch.equal(y_none.int(), y_grp.int())
    assert torch.equal(s_none, s_grp)


@pytest.mark.parametrize("rows,active,groups", [(896, 16, 14), (33664, 526, 14), (7296, 456, 14)])
def test_group_counts_smaller_only_touches_active_rows(rows, active, groups):
    """sum(counts) < rows: dead tail rows are skipped; active prefix matches
    the full-scan outputs (the dead rows are never consumed downstream)."""
    x = _bf16_x(rows, 6144, seed=2)
    counts = _counts(groups, active, x.device)
    y_none, s_none = _run(x, None)
    y_grp, s_grp = _run(x, counts)
    assert torch.equal(y_none[:active].int(), y_grp[:active].int())
    assert torch.equal(s_none[:active], s_grp[:active])


def test_group_counts_exceeding_rows_clamps():
    """sum(counts) > rows (padded tail-chunk buffer): must not crash and must
    still produce correct outputs for the rows that exist."""
    rows = 530
    x = _bf16_x(rows, 6144, seed=3)
    counts = torch.full((14,), 2400, dtype=torch.int64, device=x.device)
    counts[0] += rows - 2400 * 14  # sum == 33664 > rows == 530
    y_grp, s_grp = _run(x, counts)
    y_ref, s_ref = _golden(x)
    assert s_grp.shape == s_ref.shape
    assert (y_grp.int() - y_ref.int()).abs().max() <= 1
    assert ((s_grp - s_ref).abs() / s_ref.clamp_min(1e-6)).max() < 1e-5


def test_bf16_output_matches_float32_reference():
    x = _bf16_x(528, 6144, seed=4)
    y, scale = _run(x, None)
    y_ref, scale_ref = _golden(x)
    assert scale.shape == scale_ref.shape
    assert (y.int() - y_ref.int()).abs().max() <= 1
    assert ((scale - scale_ref).abs() / scale_ref.clamp_min(1e-6)).max() < 1e-5
