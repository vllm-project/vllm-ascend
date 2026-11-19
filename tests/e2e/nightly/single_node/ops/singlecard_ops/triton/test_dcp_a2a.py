# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.dcp_test_utils import assert_dcp_close
from vllm_ascend.ops.triton.dcp.dcp_a2a import (
    fused_dcp_lse_combine,
    pack_dcp_output_lse,
)


def _reference_merge(output: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(lse)
    safe_lse = lse.masked_fill(~finite, float("-inf"))
    weights = torch.nan_to_num(torch.softmax(safe_lse, dim=0), nan=0.0)
    safe_output = torch.where(finite.unsqueeze(-1), output.float(), 0.0)
    return (safe_output * weights.unsqueeze(-1)).sum(0).to(output.dtype)


def _simulate_receive(
    sender_outputs: torch.Tensor,
    sender_lses: torch.Tensor,
    destination_rank: int,
    scatter_dim: int,
) -> torch.Tensor:
    dcp_size = sender_outputs.shape[0]
    send_buffers = [
        pack_dcp_output_lse(
            sender_outputs[source_rank],
            sender_lses[source_rank],
            dcp_size,
            scatter_dim,
        )
        for source_rank in range(dcp_size)
    ]
    return torch.stack([send_buffers[source_rank][destination_rank] for source_rank in range(dcp_size)])


@pytest.mark.parametrize("tokens", [128, 129])
@torch.inference_mode()
def test_decode_combine_without_local(tokens):
    torch.manual_seed(35)
    world, heads, dim = 8, 96, 512
    outputs = torch.randn(world, tokens, heads, dim, device="npu", dtype=torch.bfloat16)
    lses = torch.randn(world, tokens, heads, 1, device="npu") * 8 + 70_000
    lses[:, 0, 0] = -torch.inf
    outputs[:, 0, 0] = torch.nan
    lses[0, 1, 0] = torch.nan
    outputs[0, 1, 0] = torch.nan
    recv = _simulate_receive(outputs, lses, 0, 1)
    actual = fused_dcp_lse_combine(recv, dim, 1)
    expected = _reference_merge(outputs[:, :, : heads // world], lses[:, :, : heads // world, 0])
    assert_dcp_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scatter_dim", [0, 1])
@pytest.mark.parametrize("head_dim", [96, 128, 160, 256])
@torch.inference_mode()
def test_pack_and_fused_lse_combine(
    dtype: torch.dtype,
    scatter_dim: int,
    head_dim: int,
) -> None:
    torch.manual_seed(2026)
    dcp_size = 8
    num_tokens, num_heads = (16, 4) if scatter_dim == 0 else (5, 64)
    sender_outputs = torch.randn(
        dcp_size,
        num_tokens,
        num_heads,
        head_dim,
        dtype=dtype,
        device="npu",
    )
    sender_lses = torch.randn(
        dcp_size,
        num_tokens,
        num_heads,
        1,
        dtype=torch.float32,
        device="npu",
    )
    destination_rank = 3
    recv = _simulate_receive(
        sender_outputs,
        sender_lses,
        destination_rank,
        scatter_dim,
    )

    if scatter_dim == 0:
        local_tokens = num_tokens // dcp_size
        token_slice = slice(destination_rank * local_tokens, (destination_rank + 1) * local_tokens)
        expected = _reference_merge(
            sender_outputs[:, token_slice],
            sender_lses[:, token_slice, :, 0],
        )
    else:
        local_heads = num_heads // dcp_size
        head_slice = slice(destination_rank * local_heads, (destination_rank + 1) * local_heads)
        expected = _reference_merge(
            sender_outputs[:, :, head_slice],
            sender_lses[:, :, head_slice, 0],
        )
    actual = fused_dcp_lse_combine(recv, head_dim, scatter_dim)

    assert_dcp_close(actual, expected)


@pytest.mark.parametrize("scatter_dim", [0, 1])
@torch.inference_mode()
def test_stride_aware_pack(scatter_dim: int) -> None:
    torch.manual_seed(2026)
    dcp_size, num_tokens, num_heads, head_dim = 8, 16, 64, 128
    output_storage = torch.randn(
        dcp_size,
        num_tokens,
        num_heads,
        head_dim + 4,
        dtype=torch.bfloat16,
        device="npu",
    )
    lse_storage = torch.randn(
        dcp_size,
        num_tokens,
        num_heads,
        2,
        dtype=torch.float32,
        device="npu",
    )
    sender_outputs = output_storage[..., :head_dim]
    sender_lses = lse_storage[..., :1]
    assert not sender_outputs.is_contiguous()
    assert not sender_lses.is_contiguous()

    destination_rank = 5
    recv = _simulate_receive(
        sender_outputs,
        sender_lses,
        destination_rank,
        scatter_dim,
    )
    if scatter_dim == 0:
        local_tokens = num_tokens // dcp_size
        token_slice = slice(destination_rank * local_tokens, (destination_rank + 1) * local_tokens)
        expected = _reference_merge(
            sender_outputs[:, token_slice],
            sender_lses[:, token_slice, :, 0],
        )
    else:
        local_heads = num_heads // dcp_size
        head_slice = slice(destination_rank * local_heads, (destination_rank + 1) * local_heads)
        expected = _reference_merge(
            sender_outputs[:, :, head_slice],
            sender_lses[:, :, head_slice, 0],
        )
    actual = fused_dcp_lse_combine(recv, head_dim, scatter_dim)

    assert_dcp_close(actual, expected)


@pytest.mark.parametrize("scatter_dim", [0, 1])
@torch.inference_mode()
def test_invalid_lse_and_all_invalid_rows(scatter_dim: int) -> None:
    torch.manual_seed(2026)
    dcp_size, num_tokens, num_heads, head_dim = 8, 16, 64, 256
    sender_outputs = torch.randn(
        dcp_size,
        num_tokens,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="npu",
    )
    sender_lses = torch.randn(
        dcp_size,
        num_tokens,
        num_heads,
        1,
        dtype=torch.float32,
        device="npu",
    )
    sender_lses[0, 0, 0, 0] = float("nan")
    sender_lses[1, 0, 0, 0] = float("inf")
    sender_lses[2, 0, 0, 0] = float("-inf")
    sender_outputs[:3, 0, 0] = float("nan")
    sender_lses[:, 1, 0, 0] = float("-inf")
    sender_outputs[:, 1, 0] = float("nan")

    destination_rank = 0
    recv = _simulate_receive(
        sender_outputs,
        sender_lses,
        destination_rank,
        scatter_dim,
    )
    actual = fused_dcp_lse_combine(recv, head_dim, scatter_dim)

    if scatter_dim == 0:
        expected = _reference_merge(
            sender_outputs[:, : num_tokens // dcp_size],
            sender_lses[:, : num_tokens // dcp_size, :, 0],
        )
    else:
        expected = _reference_merge(
            sender_outputs[:, :, : num_heads // dcp_size],
            sender_lses[:, :, : num_heads // dcp_size, 0],
        )
    assert_dcp_close(actual, expected)
    assert torch.count_nonzero(actual[1, 0]).item() == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scatter_dim", [0, 1])
@torch.inference_mode()
def test_finite_lse_outside_activation_dtype_range(
    dtype: torch.dtype,
    scatter_dim: int,
) -> None:
    torch.manual_seed(2026)
    dcp_size, num_tokens, num_heads, head_dim = 8, 16, 64, 128
    sender_outputs = torch.randn(
        dcp_size,
        num_tokens,
        num_heads,
        head_dim,
        dtype=dtype,
        device="npu",
    )
    sender_lses = torch.full(
        (dcp_size, num_tokens, num_heads, 1),
        70_000.0,
        dtype=torch.float32,
        device="npu",
    )
    sender_lses += torch.arange(dcp_size, dtype=torch.float32, device="npu").view(-1, 1, 1, 1) * 0.25

    destination_rank = 4
    recv = _simulate_receive(
        sender_outputs,
        sender_lses,
        destination_rank,
        scatter_dim,
    )
    actual = fused_dcp_lse_combine(recv, head_dim, scatter_dim)

    if scatter_dim == 0:
        local_tokens = num_tokens // dcp_size
        token_slice = slice(destination_rank * local_tokens, (destination_rank + 1) * local_tokens)
        expected = _reference_merge(
            sender_outputs[:, token_slice],
            sender_lses[:, token_slice, :, 0],
        )
    else:
        local_heads = num_heads // dcp_size
        head_slice = slice(destination_rank * local_heads, (destination_rank + 1) * local_heads)
        expected = _reference_merge(
            sender_outputs[:, :, head_slice],
            sender_lses[:, :, head_slice, 0],
        )

    assert_dcp_close(actual, expected)
    assert torch.count_nonzero(actual).item() > 0


@torch.inference_mode()
def test_mla_history_lse_then_current_merge() -> None:
    num_tokens = 48
    torch.manual_seed(num_tokens)
    outputs = torch.randn(17, num_tokens, 6, 512)
    lse = torch.randn(17, num_tokens, 6) * 5 + 1000
    lse[:16, 0] = float("inf")
    outputs[:16, 0] = float("nan")
    lse[:, 1] = float("inf")
    outputs[:, 1] = float("nan")
    lse[1:16, 2] = float("inf")
    lse[0, 3] = float("nan")
    lse[1, 3] = -float("inf")
    expected = _reference_merge(outputs, lse)
    # Compare direct and staged merges against the same independent reference.
    partials = torch.cat((outputs, lse.unsqueeze(-1)), dim=-1).npu()
    direct = fused_dcp_lse_combine(partials, 512, scatter_dim=0)
    assert direct.dtype == torch.float32
    assert_dcp_close(direct, expected)
    history = torch.cat((outputs[:16], lse[:16].unsqueeze(-1)), dim=-1).npu()
    merged_history = fused_dcp_lse_combine(history, 512, scatter_dim=0, return_lse=True)
    assert merged_history.shape == (num_tokens, 6, 513)
    current = torch.cat((outputs[-1], lse[-1].unsqueeze(-1)), dim=-1).npu()
    actual = fused_dcp_lse_combine(torch.stack((merged_history, current)), 512, scatter_dim=0)
    assert_dcp_close(actual, expected)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize(
    "scatter_dim,dcp_size,head_dim,local_dtype",
    [
        (0, 1, 96, torch.float32),
        (1, 1, 96, torch.float32),
        (0, 1, 96, torch.bfloat16),
        (1, 1, 96, torch.bfloat16),
        (0, 1, 96, torch.float16),
        (1, 1, 96, torch.float16),
        (1, 1, 256, torch.float32),
        (0, 1, 256, torch.bfloat16),
        (1, 1, 256, torch.float16),
        (0, 1, 512, torch.float32),
        (1, 1, 512, torch.bfloat16),
        (0, 1, 512, torch.float16),
        (1, 2, 96, torch.float32),
        (0, 2, 96, torch.bfloat16),
        (1, 2, 96, torch.float16),
        (0, 2, 256, torch.float32),
        (1, 2, 256, torch.bfloat16),
        (0, 2, 256, torch.float16),
        (0, 2, 512, torch.float32),
        (1, 2, 512, torch.float32),
        (0, 2, 512, torch.bfloat16),
        (1, 2, 512, torch.bfloat16),
        (0, 2, 512, torch.float16),
        (1, 2, 512, torch.float16),
        (0, 8, 96, torch.float32),
        (1, 8, 96, torch.bfloat16),
        (0, 8, 96, torch.float16),
        (0, 8, 256, torch.float32),
        (1, 8, 256, torch.float32),
        (0, 8, 256, torch.bfloat16),
        (1, 8, 256, torch.bfloat16),
        (0, 8, 256, torch.float16),
        (1, 8, 256, torch.float16),
        (0, 8, 512, torch.float32),
        (1, 8, 512, torch.bfloat16),
        (0, 8, 512, torch.float16),
    ],
)
@torch.inference_mode()
def test_combine_with_raw_local_fia(scatter_dim, dcp_size, head_dim, local_dtype):
    torch.manual_seed(16362)
    tokens, heads = 3, 2
    history = torch.randn(dcp_size, tokens, heads, head_dim, device="npu")
    history_lse = torch.randn(dcp_size, tokens, heads, 1, device="npu") * 80
    # Slice both tensors to exercise independent non-contiguous FIA strides.
    local = torch.randn(tokens, heads, head_dim * 2, device="npu", dtype=local_dtype)[..., ::2]
    local_lse = (torch.randn(tokens, heads, 2, device="npu") * 80)[..., :1]
    # History empty/current valid, history valid/current empty, both empty.
    history_lse[:, 0, 0] = -torch.inf
    history[:, 0, 0] = torch.nan
    local_lse[0, 1] = torch.inf
    local[0, 1] = torch.nan
    history_lse[:, 1, 0] = torch.nan
    history[:, 1, 0] = torch.nan
    local_lse[1, 0] = -torch.inf
    local[1, 0] = torch.nan
    recv = torch.cat((history, history_lse), dim=-1)
    if scatter_dim == 1:
        recv = recv.transpose(1, 2).contiguous()
    actual = fused_dcp_lse_combine(
        recv, head_dim, scatter_dim, return_lse=True, local_output=local, local_lse=local_lse
    )
    values = torch.cat((history, local.float().unsqueeze(0)))
    lses = torch.cat((history_lse, local_lse.unsqueeze(0)))[..., 0]
    expected = _reference_merge(values, lses)
    expected_lse = torch.logsumexp(lses.masked_fill(~torch.isfinite(lses), -torch.inf), dim=0)
    assert_dcp_close(actual[..., :head_dim], expected)
    # Empty contributions have an exact -inf LSE sentinel, outside finite tolerances.
    actual_lse, golden_lse = actual[..., head_dim].cpu(), expected_lse.cpu()
    finite = golden_lse.isfinite()
    assert torch.equal(actual_lse[~finite], golden_lse[~finite])
    assert_dcp_close(actual_lse[finite], golden_lse[finite])
