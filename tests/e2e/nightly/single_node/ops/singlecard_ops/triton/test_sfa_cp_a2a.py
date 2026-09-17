# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.triton.sfa_cp import (
    fused_sfa_dcp_lse_combine,
    pack_sfa_dcp_output_lse,
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
        pack_sfa_dcp_output_lse(
            sender_outputs[source_rank],
            sender_lses[source_rank],
            dcp_size,
            scatter_dim,
        )
        for source_rank in range(dcp_size)
    ]
    return torch.stack([send_buffers[source_rank][destination_rank] for source_rank in range(dcp_size)])


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
    actual = fused_sfa_dcp_lse_combine(recv, head_dim, scatter_dim)

    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


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
    actual = fused_sfa_dcp_lse_combine(recv, head_dim, scatter_dim)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


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
    actual = fused_sfa_dcp_lse_combine(recv, head_dim, scatter_dim)

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
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
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
    actual = fused_sfa_dcp_lse_combine(recv, head_dim, scatter_dim)

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

    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert torch.count_nonzero(actual).item() > 0


@torch.inference_mode()
def test_mla_history_and_current_merge() -> None:
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
    # Compare packed and local-contribution merges against the same reference.
    partials = torch.cat((outputs, lse.unsqueeze(-1)), dim=-1).npu()
    direct = fused_sfa_dcp_lse_combine(partials, 512, scatter_dim=0)
    assert direct.dtype == torch.float32
    torch.testing.assert_close(direct.cpu(), expected, atol=3e-6, rtol=3e-5)
    history = torch.cat((outputs[:16], lse[:16].unsqueeze(-1)), dim=-1).npu()
    actual = fused_sfa_dcp_lse_combine(
        history, 512, scatter_dim=0, local_output=outputs[-1].npu(), local_lse=lse[-1].unsqueeze(-1).npu()
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-4, rtol=5e-4)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("scatter_dim", [0, 1])
@pytest.mark.parametrize("dcp_size", [1, 2, 8])
@pytest.mark.parametrize("head_dim", [96, 256, 512])
@pytest.mark.parametrize("local_dtype", [torch.float32, torch.bfloat16, torch.float16])
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
    actual = fused_sfa_dcp_lse_combine(recv, head_dim, scatter_dim, local_output=local, local_lse=local_lse)
    values = torch.cat((history, local.float().unsqueeze(0)))
    lses = torch.cat((history_lse, local_lse.unsqueeze(0)))[..., 0]
    expected = _reference_merge(values, lses)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("num_tokens", [8, 16, 31, 32, 63, 64, 65, 128, 256])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "dcp_size,num_heads,head_dim",
    [(2, 6, 64), (4, 28, 128), (8, 96, 512), (16, 80, 384), (4, 36, 129), (2, 2, 1), (4, 20, 511), (4, 20, 513)],
)
@torch.inference_mode()
def test_sfa_batched_local_combine(
    num_tokens: int, strided: bool, dcp_size: int, num_heads: int, head_dim: int
) -> None:
    """Cover row-batching boundaries, rank counts, head counts and column tails."""
    torch.manual_seed(20260917)
    destination_rank = dcp_size - 1
    local_heads = num_heads // dcp_size
    stride = 2 if strided else 1
    values = torch.randn(dcp_size, num_tokens, num_heads, head_dim * stride, device="npu", dtype=torch.bfloat16)[
        ..., ::stride
    ]
    lses = (torch.randn(dcp_size, num_tokens, num_heads, stride, device="npu") * 3)[..., :1]
    local = torch.randn(num_tokens, local_heads, head_dim * stride, device="npu", dtype=torch.bfloat16)[..., ::stride]
    local_lse = (torch.randn(num_tokens, local_heads, stride, device="npu") * 3)[..., :1]
    values[:, 0] = torch.nan
    lses[:, 0] = -torch.inf
    local[0] = torch.nan
    local_lse[0] = -torch.inf
    recv = _simulate_receive(values, lses, destination_rank, scatter_dim=1)
    actual = fused_sfa_dcp_lse_combine(recv, head_dim, scatter_dim=1, local_output=local, local_lse=local_lse)
    head_slice = slice(destination_rank * local_heads, (destination_rank + 1) * local_heads)
    reference_values = torch.cat((values[:, :, head_slice], local.unsqueeze(0)))
    reference_lses = torch.cat((lses[:, :, head_slice], local_lse.unsqueeze(0)))[..., 0]
    expected = _reference_merge(reference_values, reference_lses)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("with_local", [False, True])
@torch.inference_mode()
def test_raw_dcp8_bit_packing_and_merge(num_tokens: int, strided: bool, with_local: bool) -> None:
    from vllm_ascend.ops.triton.sfa_dcp_exchange import pack_raw_dcp_output_lse
    from vllm_ascend.ops.triton.sfa_dcp_merge import merge_raw_dcp_output_lse

    torch.manual_seed(20260917)
    stride = 2 if strided else 1
    values = torch.randn(8, num_tokens, 96, 512 * stride, device="npu", dtype=torch.bfloat16)[..., ::stride]
    lses = (torch.randn(8, num_tokens, 96, stride, device="npu") * 80)[..., :1]
    values[:, 0] = torch.nan
    lses[:, 0] = -torch.inf
    lses[0, 1, 0] = torch.inf
    lses[1, 1, 0] = torch.nan
    values[:2, 1, 0] = torch.nan
    # A large finite FP32 LSE must survive raw-bit transport without narrowing.
    lses[:, 2] = 70000 + torch.arange(8, device="npu")[:, None, None] * 0.25
    recv = []
    for source in range(8):
        packed = pack_raw_dcp_output_lse(values[source], lses[source])
        golden_words = values[source].cpu().contiguous().view(torch.int32).transpose(0, 1)
        golden_lse = lses[source].cpu().contiguous().view(torch.int32).transpose(0, 1)
        golden = torch.cat((golden_words, golden_lse), dim=-1).reshape(8, 12, num_tokens, 257)
        torch.testing.assert_close(packed.cpu(), golden, rtol=0, atol=0)
        recv.append(packed[0])
    recv = torch.stack(recv)
    local = torch.randn(num_tokens, 12, 512 * stride, device="npu", dtype=torch.bfloat16)[..., ::stride]
    local_lse = (torch.randn(num_tokens, 12, stride, device="npu") * 80)[..., :1]
    local[0] = torch.nan
    local_lse[0] = torch.nan
    args = (local, local_lse) if with_local else (None, None)
    actual = merge_raw_dcp_output_lse(recv, 512, 1, *args)
    golden_values, golden_lses = values[:, :, :12].cpu(), lses[:, :, :12, 0].cpu()
    if with_local:
        golden_values = torch.cat((golden_values, local.cpu().unsqueeze(0)))
        golden_lses = torch.cat((golden_lses, local_lse.cpu()[..., 0].unsqueeze(0)))
    expected = _reference_merge(golden_values, golden_lses)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[0]).item() == 0


@pytest.mark.parametrize("num_tokens", [4, 32, 64])
@pytest.mark.parametrize("odd_offset", [False, True])
@torch.inference_mode()
def test_raw_dcp8_pack_leading_strides(num_tokens: int, odd_offset: bool) -> None:
    from vllm_ascend.ops.triton.sfa_dcp_exchange import pack_raw_dcp_output_lse

    if odd_offset:
        output = torch.randn(num_tokens, 96, 514, device="npu", dtype=torch.bfloat16)[..., 1:513]
    else:
        output = torch.randn(num_tokens * 2, 192, 512, device="npu", dtype=torch.bfloat16)[::2, ::2]
    lse = torch.randn(num_tokens * 2, 192, 1, device="npu")[::2, ::2]
    actual = pack_raw_dcp_output_lse(output, lse)
    expected = (
        torch.cat((output.cpu().contiguous().view(torch.int32), lse.cpu().contiguous().view(torch.int32)), dim=-1)
        .transpose(0, 1)
        .reshape(8, 12, num_tokens, 257)
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
