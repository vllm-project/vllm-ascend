# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import Mock

import pytest
import torch

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.dcp_test_utils import assert_dcp_close
from vllm_ascend.device.device_config import AscendDeviceType
from vllm_ascend.ops.triton.dcp import dcp_a2a
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


@pytest.mark.parametrize(
    "head_dim,scatter_dim,dtype",
    [
        # Pair dtype with scatter direction, keeping both at each head dimension.
        (96, 0, torch.float16),
        (96, 1, torch.bfloat16),
        (128, 0, torch.bfloat16),
        (128, 1, torch.float16),
        (160, 0, torch.float16),
        (160, 1, torch.bfloat16),
        (256, 0, torch.bfloat16),
        (256, 1, torch.float16),
    ],
)
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
        # Keep every shape/dtype and shape/scatter pair.
        # Exercise both scatter directions for each local dtype across the shapes.
        (0, 1, 96, torch.float32),
        (1, 1, 96, torch.bfloat16),
        (0, 1, 96, torch.float16),
        (1, 2, 512, torch.float32),
        (0, 2, 512, torch.bfloat16),
        (1, 2, 512, torch.float16),
        (0, 8, 256, torch.float32),
        (1, 8, 256, torch.bfloat16),
        (0, 8, 256, torch.float16),
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


@pytest.mark.parametrize(
    "ranks,head_dim,num_tokens,local_heads,local_stride,local_dtype",
    [(r, d, 16, 4, 1, torch.bfloat16) for r in (4, 8, 16) for d in (96, 256, 512, 1024)]
    + [(16, 512, t, 4, 1, torch.bfloat16) for t in (1, 4, 64)]
    + [pytest.param(16, 512, t, 6, 1, torch.bfloat16, id=f"kimi-heads6-t{t}") for t in (16, 64)]
    + [(12, 160, 16, 4, 1, torch.bfloat16), (16, 130, 16, 4, 1, torch.bfloat16)]
    + [(r, 512, 16, 4, 2, torch.bfloat16) for r in (4, 8, 16)]
    + [(r, d, 16, 4, 1, dtype) for r, d in ((4, 96), (16, 512), (8, 256)) for dtype in (torch.float16, torch.float32)],
)
@torch.inference_mode()
def test_rank_tiled_combine_fp64_reference(
    monkeypatch, ranks, head_dim, num_tokens, local_heads, local_stride, local_dtype
):
    """Check the dispatched kernel against original inputs, not a scalar reduction.

    Parallel FP32 summation may cross BF16 rounding boundaries. Independently
    bound its mathematical error, then check the output's BF16 rounding exactly.
    """
    torch.manual_seed(7201 + num_tokens + local_heads)
    num_heads = local_heads * ranks
    history = torch.randn(ranks, num_tokens, local_heads, head_dim, device="npu", dtype=torch.bfloat16)
    history_lse = torch.randn(ranks, num_tokens, local_heads, 1, device="npu") * 30
    local = torch.randn(num_tokens, local_heads, head_dim * local_stride, device="npu", dtype=local_dtype)[
        ..., ::local_stride
    ]
    local_lse = (torch.randn(num_tokens, local_heads, 2, device="npu") * 30)[..., :1]
    history_lse[:, 0, 0] = -torch.inf
    history[:, 0, 0] = torch.nan
    local_lse[0, 0] = torch.nan
    local[0, 0] = torch.nan
    if num_tokens > 1:
        local_lse[1, 0] = torch.inf
        local[1, 0] = torch.nan
    # Treat head partitions as received rank contributions. The oracle below
    # reads the original values and LSE, independently of the packed wire.
    packed = pack_dcp_output_lse(
        history.transpose(0, 1).reshape(num_tokens, num_heads, head_dim),
        history_lse.transpose(0, 1).reshape(num_tokens, num_heads, 1),
        ranks,
        1,
    )
    rank_tiled_kernel = dcp_a2a._fused_dcp_lse_combine_rank_tiled_kernel
    scalar_kernel = dcp_a2a._fused_dcp_lse_combine_kernel
    batched_kernel = dcp_a2a._fused_dcp_lse_combine_batched_kernel
    # No static twin is warmed here: A3 must fall back, A5 keeps the dynamic
    # kernel because a runtime rank-row stride still vectorizes there.
    monkeypatch.setattr(dcp_a2a, "_A3_WARMING", False)
    monkeypatch.setattr(dcp_a2a, "_A3_STATIC_SIGNATURES", frozenset())
    launches: list[tuple[Any, ...]] = []

    class LaunchSpy:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((self.kernel, grid, args, kwargs))
                return self.kernel[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(dcp_a2a, "_fused_dcp_lse_combine_rank_tiled_kernel", LaunchSpy(rank_tiled_kernel))
    monkeypatch.setattr(dcp_a2a, "_fused_dcp_lse_combine_kernel", LaunchSpy(scalar_kernel))
    monkeypatch.setattr(dcp_a2a, "_fused_dcp_lse_combine_batched_kernel", LaunchSpy(batched_kernel))
    actual = fused_dcp_lse_combine(packed, head_dim, 1, local_output=local, local_lse=local_lse)
    assert len(launches) == 1
    kernel, grid, args, kwargs = launches[0]
    rank_tile = dcp_a2a.triton.next_power_of_2(ranks)
    feature_tile = min(512, dcp_a2a.triton.next_power_of_2(head_dim), 8192 // rank_tile)
    tiles_per_head = dcp_a2a.triton.cdiv(head_dim, feature_tile)
    total_rows = num_tokens * local_heads
    cores = dcp_a2a.get_vectorcore_num()
    min_rows = max(dcp_a2a.triton.cdiv(2 * cores, ranks), ranks)
    # The receive buffer is BF16 (history is BF16), so the rank-tiled gate only
    # depends on the world size, head dim, row stride and the row minimum.
    rank_tile_common = 8 <= ranks <= 16 and head_dim >= 512 and local_stride == 1 and total_rows >= min_rows
    expected_dynamic = dcp_a2a.is_950() and rank_tile_common and ranks >= 2 * tiles_per_head
    expected_batched = not expected_dynamic and head_dim <= 512 and total_rows >= 4 * cores and local_stride == 1
    assert kernel is (rank_tiled_kernel if expected_dynamic else batched_kernel if expected_batched else scalar_kernel)
    # Check the same dispatched reduction with an FP32 output allocation.
    # Output strides are measured in elements, so they are unchanged here.
    fp32 = torch.empty(actual.shape, device="npu", dtype=torch.float32)
    fp32_args = list(args)
    fp32_args[1] = fp32
    kernel[grid](*fp32_args, **kwargs)
    values = torch.cat((history.cpu().double(), local.cpu().double()[None]))
    logs = torch.cat((history_lse.cpu().double(), local_lse.cpu().double()[None]))
    valid = torch.isfinite(logs)
    maximum = torch.where(valid, logs, -torch.inf).amax(0)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    weights = torch.where(valid, torch.exp(logs - maximum), 0.0)
    denominator = weights.sum(0)
    expected = (torch.where(valid, values, 0.0) * weights).sum(0) / torch.where(denominator > 0, denominator, 1.0)
    # The vector unit's FP32 exp/reduction differs from a float64 oracle by up
    # to ~1.5e-5 on outputs that nearly cancel; the scalar kernel shows the
    # same bound, so it is hardware rounding rather than a dispatch difference.
    torch.testing.assert_close(fp32.cpu().double(), expected, rtol=1e-4, atol=5e-5)
    torch.testing.assert_close(actual, fp32.to(torch.bfloat16), rtol=0, atol=0)
    assert torch.isfinite(actual).all().item()
    assert actual[0, 0].count_nonzero().item() == 0
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        replayed = fused_dcp_lse_combine(packed, head_dim, 1, local_output=local, local_lse=local_lse)
    for _ in range(3):
        local_lse.add_(0.1)
        graph.replay()
        eager = fused_dcp_lse_combine(packed, head_dim, 1, local_output=local, local_lse=local_lse)
        torch.testing.assert_close(replayed, eager, rtol=0, atol=0)


class _LaunchSpy:
    """Record the kernel and arguments the dispatcher selected."""

    def __init__(self, kernel, launches):
        self.kernel = kernel
        self.launches = launches

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((self.kernel, grid, args, kwargs))
            return self.kernel[grid](*args, **kwargs)

        return launch


def _rank_tiled_inputs(ranks, head_dim, tokens, heads, layout="token_major"):
    torch.manual_seed(19555 + tokens + heads)
    history = torch.randn(ranks, tokens, heads, head_dim, device="npu", dtype=torch.bfloat16)
    history_lse = torch.randn(ranks, tokens, heads, 1, device="npu") * 8
    packed = pack_dcp_output_lse(
        history.transpose(0, 1).reshape(tokens, ranks * heads, head_dim),
        history_lse.transpose(0, 1).reshape(tokens, ranks * heads, 1),
        ranks,
        1,
    )
    if layout == "token_major":
        local = torch.randn(tokens, heads, head_dim, device="npu", dtype=torch.bfloat16)
        local_lse = torch.randn(tokens, heads, 1, device="npu") * 8
    else:
        local = torch.randn(heads, tokens, head_dim, device="npu", dtype=torch.bfloat16).permute(1, 0, 2)
        local_lse = (torch.randn(heads, tokens, 1, device="npu") * 8).permute(1, 0, 2)
    return packed, local, local_lse, history, history_lse


def _fp64_reference_merge(history, history_lse, local, local_lse):
    values = torch.cat((history.cpu().double(), local.cpu().double()[None]))
    logs = torch.cat((history_lse.cpu().double(), local_lse.cpu().double()[None]))
    valid = torch.isfinite(logs)
    maximum = torch.where(valid, logs, -torch.inf).amax(0)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    weights = torch.where(valid, torch.exp(logs - maximum), 0.0)
    denominator = weights.sum(0)
    return (torch.where(valid, values, 0.0) * weights).sum(0) / torch.where(denominator > 0, denominator, 1.0)


@torch.inference_mode()
def test_dynamic_rank_tiled_reuses_one_compiled_variant(monkeypatch):
    """A5's runtime token/head counts must not create Triton specializations."""
    if not dcp_a2a.is_950():
        pytest.skip("the dynamic rank-tiled kernel is the A5 dispatch path")
    original = dcp_a2a._fused_dcp_lse_combine_rank_tiled_kernel
    kernel = dcp_a2a.triton.jit(original.fn, do_not_specialize=original.do_not_specialize)
    compile_spy = Mock(wraps=kernel._do_compile)
    monkeypatch.setattr(kernel, "_do_compile", compile_spy)
    monkeypatch.setattr(dcp_a2a, "_fused_dcp_lse_combine_rank_tiled_kernel", kernel)

    device = torch.npu.current_device()
    if hasattr(kernel, "device_caches"):
        kernel_cache = kernel.device_caches[device][0]
    else:
        kernel_cache = kernel.cache[device]
    kernel_cache.clear()

    ranks, head_dim = 16, 512
    for tokens, heads in ((4, 4), (8, 4), (16, 4), (32, 4), (16, 6)):
        partial = torch.randn(tokens, ranks * heads, head_dim, device="npu", dtype=torch.bfloat16)
        lse = torch.randn(tokens, ranks * heads, 1, device="npu") * 30
        recv = pack_dcp_output_lse(partial, lse, ranks, 1)
        local = torch.randn(tokens, heads, head_dim, device="npu", dtype=torch.bfloat16)
        local_lse = torch.randn(tokens, heads, 1, device="npu") * 30
        fused_dcp_lse_combine(recv, head_dim, 1, local_output=local, local_lse=local_lse)
        torch.npu.synchronize()
        assert len(kernel_cache) == compile_spy.call_count == 1, (tokens, heads)


@torch.inference_mode()
def test_a3_static_unwarmed_combine_falls_back(monkeypatch):
    """A3 without a warmed full signature must keep the scalar/batched path."""
    if dcp_a2a.get_ascend_device_type() != AscendDeviceType.A3:
        pytest.skip("A3 dispatches the static rank-tiled twin")
    monkeypatch.setattr(dcp_a2a, "is_950", lambda: False)
    monkeypatch.setattr(dcp_a2a, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(dcp_a2a, "_A3_WARMING", False)
    monkeypatch.setattr(dcp_a2a, "_A3_STATIC_SIGNATURES", frozenset())
    monkeypatch.setattr(dcp_a2a, "_UNWARMED_LOGGED", set())
    logger = Mock()
    monkeypatch.setattr(dcp_a2a, "logger", logger)

    ranks, head_dim, heads = 16, 512, 4
    static_kernel = dcp_a2a._fused_dcp_lse_combine_rank_tiled_static_kernel
    dynamic_kernel = dcp_a2a._fused_dcp_lse_combine_rank_tiled_kernel
    scalar_kernel = dcp_a2a._fused_dcp_lse_combine_kernel
    batched_kernel = dcp_a2a._fused_dcp_lse_combine_batched_kernel
    launches: list[tuple[Any, ...]] = []
    for name, kernel in (
        ("_fused_dcp_lse_combine_rank_tiled_static_kernel", static_kernel),
        ("_fused_dcp_lse_combine_rank_tiled_kernel", dynamic_kernel),
        ("_fused_dcp_lse_combine_kernel", scalar_kernel),
        ("_fused_dcp_lse_combine_batched_kernel", batched_kernel),
    ):
        monkeypatch.setattr(dcp_a2a, name, _LaunchSpy(kernel, launches))

    tokens = 16
    packed, local, local_lse, _, _ = _rank_tiled_inputs(ranks, head_dim, tokens, heads)
    for _ in range(2):
        fused_dcp_lse_combine(packed, head_dim, 1, local_output=local, local_lse=local_lse)
        torch.npu.synchronize()
    # 64 rows keep the fused loop below the batched threshold.
    assert [kernel for kernel, *_ in launches] == [scalar_kernel, scalar_kernel]
    # Decode-sized misses warn once; the second call stays quiet.
    assert logger.warning.call_count == 1
    assert logger.debug.call_count == 0

    # Prefill-sized misses log at debug so large counts do not spam warnings.
    logger.reset_mock()
    monkeypatch.setattr(dcp_a2a, "_UNWARMED_LOGGED", set())
    packed, local, local_lse, _, _ = _rank_tiled_inputs(ranks, head_dim, 256, heads)
    fused_dcp_lse_combine(packed, head_dim, 1, local_output=local, local_lse=local_lse)
    torch.npu.synchronize()
    assert launches[-1][0] is batched_kernel
    assert logger.warning.call_count == 0
    assert logger.debug.call_count == 1


@torch.inference_mode()
def test_a3_static_warmed_signature_matches_reference(monkeypatch):
    """A warmed A3 signature selects the static twin and keeps kernel accuracy."""
    if dcp_a2a.get_ascend_device_type() != AscendDeviceType.A3:
        pytest.skip("A3 dispatches the static rank-tiled twin")
    monkeypatch.setattr(dcp_a2a, "is_950", lambda: False)
    monkeypatch.setattr(dcp_a2a, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(dcp_a2a, "_A3_WARMING", False)
    monkeypatch.setattr(dcp_a2a, "_UNWARMED_LOGGED", set())
    logger = Mock()
    monkeypatch.setattr(dcp_a2a, "logger", logger)

    ranks, head_dim, tokens, heads = 16, 512, 16, 4
    packed, local, local_lse, history, history_lse = _rank_tiled_inputs(ranks, head_dim, tokens, heads)
    signature = (heads, tokens, head_dim, ranks, *local.stride(), *local_lse.stride()[:2])
    monkeypatch.setattr(dcp_a2a, "_A3_STATIC_SIGNATURES", frozenset({signature}))

    static_kernel = dcp_a2a._fused_dcp_lse_combine_rank_tiled_static_kernel
    dynamic_kernel = dcp_a2a._fused_dcp_lse_combine_rank_tiled_kernel
    scalar_kernel = dcp_a2a._fused_dcp_lse_combine_kernel
    batched_kernel = dcp_a2a._fused_dcp_lse_combine_batched_kernel
    launches: list[tuple[Any, ...]] = []
    for name, kernel in (
        ("_fused_dcp_lse_combine_rank_tiled_static_kernel", static_kernel),
        ("_fused_dcp_lse_combine_rank_tiled_kernel", dynamic_kernel),
        ("_fused_dcp_lse_combine_kernel", scalar_kernel),
        ("_fused_dcp_lse_combine_batched_kernel", batched_kernel),
    ):
        monkeypatch.setattr(dcp_a2a, name, _LaunchSpy(kernel, launches))

    actual = fused_dcp_lse_combine(packed, head_dim, 1, local_output=local, local_lse=local_lse)
    torch.npu.synchronize()
    assert [kernel for kernel, *_ in launches] == [static_kernel]
    _, grid, args, kwargs = launches[0]
    assert kwargs["NUM_TOKENS"] == tokens
    assert kwargs["NUM_HEADS"] == heads
    assert kwargs["BLOCK_R"] == ranks
    assert kwargs["BLOCK_D"] == 512
    assert grid == (min(tokens * heads, dcp_a2a.get_vectorcore_num()),)
    assert logger.warning.call_count == 0

    # Re-run the selected kernel with an FP32 output: compare the reduction
    # itself against a float64 oracle, then check the BF16 rounding exactly.
    fp32 = torch.empty(actual.shape, device="npu", dtype=torch.float32)
    fp32_args = list(args)
    fp32_args[1] = fp32
    static_kernel[grid](*fp32_args, **kwargs)
    expected = _fp64_reference_merge(history, history_lse, local, local_lse)
    torch.testing.assert_close(fp32.cpu().double(), expected, rtol=1e-4, atol=5e-5)
    torch.testing.assert_close(actual, fp32.to(torch.bfloat16), rtol=0, atol=0)

    # The full signature includes output/LSE strides: the same shape with a
    # head-major layout is not warmed and must fall back.
    launches.clear()
    _, head_major, head_major_lse, _, _ = _rank_tiled_inputs(ranks, head_dim, tokens, heads, layout="head_major")
    fused_dcp_lse_combine(packed, head_dim, 1, local_output=head_major, local_lse=head_major_lse)
    torch.npu.synchronize()
    assert [kernel for kernel, *_ in launches] == [scalar_kernel]
