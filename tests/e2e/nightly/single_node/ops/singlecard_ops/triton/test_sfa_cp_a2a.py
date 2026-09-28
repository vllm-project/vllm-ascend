# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.triton.dcp import sfa_cp
from vllm_ascend.ops.triton.dcp.sfa_cp import (
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


_A5_FIXED_CASES = (
    pytest.param(64, 96, 512, 8, 1, 1, True, id="original-shape"),
    pytest.param(4, 96, 512, 8, 1, 1, True, id="too-few-pack-rows"),
    pytest.param(7, 96, 512, 8, 1, 1, True, id="small-token-count"),
    pytest.param(64, 8, 512, 8, 1, 1, True, id="few-local-heads"),
    pytest.param(64, 32, 128, 4, 1, 1, True, id="dcp4-narrow-head"),
    pytest.param(64, 12, 512, 3, 1, 1, True, id="dcp3-combine-boundary"),
    pytest.param(64, 4, 512, 1, 1, 1, True, id="dcp1-wide-small-rows"),
    pytest.param(128, 4, 512, 1, 1, 1, True, id="dcp1-wide-enough-rows"),
    pytest.param(64, 4, 128, 1, 1, 1, True, id="dcp1-narrow-combine"),
    pytest.param(128, 8, 257, 2, 1, 1, True, id="dcp2-contiguous-wide-local"),
    pytest.param(128, 8, 257, 2, 1, 2, True, id="dcp2-strided-wide-local"),
    pytest.param(64, 96, 1024, 8, 1, 1, True, id="wide-pack-combine-fallback"),
    pytest.param(64, 96, 2048, 8, 1, 1, True, id="pack-dim-limit"),
    pytest.param(64, 96, 512, 8, 2, 1, True, id="strided-pack-fallback"),
    pytest.param(257, 32, 128, 4, 1, 1, True, id="beyond-old-token-limit"),
    pytest.param(64, 36, 128, 9, 1, 1, True, id="dcp9-pack-only"),
    pytest.param(64, 64, 128, 16, 1, 1, False, id="dcp16-pack-only"),
)


@torch.inference_mode()
def _check_a5_batching_case(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    dcp_size: int,
    output_stride: int,
    local_stride: int,
    run_combine: bool,
) -> None:
    """Compare one A5 shape with scalar kernels and check its dispatch."""
    if not sfa_cp.is_950():
        pytest.skip("The batched SFA kernels are enabled only on A5")
    sfa_cp.init_device_properties_triton()
    vector_cores = sfa_cp.get_vectorcore_num()
    torch.manual_seed(2026 + num_tokens + dcp_size)
    output = torch.randn(num_tokens, num_heads, head_dim * output_stride, device="npu", dtype=torch.bfloat16)[
        ..., ::output_stride
    ]
    lse = torch.randn(num_tokens, num_heads, 1, device="npu", dtype=torch.float32)
    local_heads = num_heads // dcp_size
    local_output = torch.randn(num_tokens, local_heads, head_dim * local_stride, device="npu", dtype=torch.bfloat16)[
        ..., ::local_stride
    ]
    local_lse = torch.randn(num_tokens, local_heads, 1, device="npu", dtype=torch.float32)

    monkeypatch.setattr(sfa_cp, "is_950", lambda: False)
    scalar_send = pack_dcp_output_lse(output, lse, dcp_size, 1)
    if run_combine:
        scalar_combined = fused_dcp_lse_combine(
            scalar_send, head_dim, 1, local_output=local_output, local_lse=local_lse
        )

    class LaunchSpy:
        def __init__(self, kernel):
            self.kernel = kernel
            self.calls = 0

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def tracked_launch(*args, **kwargs):
                self.calls += 1
                return launch(*args, **kwargs)

            return tracked_launch

    pack_spy = LaunchSpy(sfa_cp._pack_dcp_output_lse_batched_kernel)
    combine_spy = LaunchSpy(sfa_cp._fused_dcp_lse_combine_batched_kernel)
    monkeypatch.setattr(sfa_cp, "_pack_dcp_output_lse_batched_kernel", pack_spy)
    monkeypatch.setattr(sfa_cp, "_fused_dcp_lse_combine_batched_kernel", combine_spy)
    monkeypatch.setattr(sfa_cp, "is_950", lambda: True)
    batched_send = pack_dcp_output_lse(output, lse, dcp_size, 1)
    if run_combine:
        batched_combined = fused_dcp_lse_combine(
            batched_send, head_dim, 1, local_output=local_output, local_lse=local_lse
        )

    expected_output = output.reshape(num_tokens, dcp_size, local_heads, head_dim).permute(1, 2, 0, 3)
    torch.testing.assert_close(batched_send[..., :head_dim], expected_output, atol=0, rtol=0)
    torch.testing.assert_close(batched_send[..., :head_dim], scalar_send[..., :head_dim], atol=0, rtol=0)
    encoded_lse = batched_send[..., head_dim:].float()
    exponent_code = encoded_lse[..., :1]
    significand = encoded_lse[..., 1:2] * 65536 + encoded_lse[..., 2:3] * 256 + encoded_lse[..., 3:4]
    decoded_lse = torch.where(exponent_code < 0, -1.0, 1.0) * significand * torch.exp2(exponent_code.abs() - 151)
    expected_lse = lse.reshape(num_tokens, dcp_size, local_heads, 1).permute(1, 2, 0, 3)
    torch.testing.assert_close(decoded_lse, expected_lse, atol=1e-2, rtol=1e-2)
    expected_pack_batch = output_stride == 1 and head_dim <= 2048 and num_tokens * num_heads >= 8 * vector_cores
    assert pack_spy.calls == int(expected_pack_batch)
    if run_combine:
        torch.testing.assert_close(batched_combined, scalar_combined, atol=2e-2, rtol=2e-2)
        rank_values = batched_send[..., :head_dim].permute(0, 2, 1, 3)
        rank_lse = decoded_lse.squeeze(-1).permute(0, 2, 1)
        expected_combined = _reference_merge(
            torch.cat((rank_values, local_output.unsqueeze(0))),
            torch.cat((rank_lse, local_lse[..., 0].unsqueeze(0))),
        )
        torch.testing.assert_close(batched_combined, expected_combined, atol=2e-2, rtol=2e-2)
        combine_rows = num_tokens * local_heads
        min_rows_per_core = 8 if dcp_size <= 2 and head_dim > 256 else 4
        expected_combine_batch = (
            dcp_size <= 8
            and head_dim <= 512
            and combine_rows >= min_rows_per_core * vector_cores
            and (dcp_size > 2 or head_dim <= 256 or local_stride == 1)
        )
        assert combine_spy.calls == int(expected_combine_batch)


@pytest.mark.parametrize(
    ("num_tokens", "num_heads", "head_dim", "dcp_size", "output_stride", "local_stride", "run_combine"),
    _A5_FIXED_CASES,
)
def test_a5_generalized_batching_matches_scalar_path(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    dcp_size: int,
    output_stride: int,
    local_stride: int,
    run_combine: bool,
) -> None:
    """Keep business shapes fixed on every A5 variant."""
    _check_a5_batching_case(
        monkeypatch, num_tokens, num_heads, head_dim, dcp_size, output_stride, local_stride, run_combine
    )


@pytest.mark.parametrize(
    "case",
    [
        "pack-below-threshold",
        "pack-at-threshold",
        "combine-below-threshold",
        "combine-at-threshold",
        "partial-row-tile",
    ],
)
@torch.inference_mode()
def test_a5_core_relative_batching_boundaries(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    if not sfa_cp.is_950():
        pytest.skip("The batched SFA kernels are enabled only on A5")
    sfa_cp.init_device_properties_triton()
    vector_cores = sfa_cp.get_vectorcore_num()
    if case.startswith("pack-"):
        num_tokens = (8 * vector_cores + 11) // 12
        if case == "pack-below-threshold":
            num_tokens -= 1
    else:
        num_tokens = vector_cores
        if case == "combine-below-threshold":
            num_tokens -= 1
        elif case == "partial-row-tile":
            num_tokens |= 1
    assert num_tokens > 0
    if case == "partial-row-tile":
        # Twelve pack heads and four post-scatter heads both leave a tail
        # when the token count is odd.
        assert num_tokens * 12 % 8 != 0
        assert num_tokens * 4 % 8 != 0
    _check_a5_batching_case(monkeypatch, num_tokens, 12, 512, 3, 1, 1, True)


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
    actual = fused_dcp_lse_combine(recv, head_dim, scatter_dim)

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

    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
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
    torch.testing.assert_close(direct.cpu(), expected, atol=3e-6, rtol=3e-5)
    history = torch.cat((outputs[:16], lse[:16].unsqueeze(-1)), dim=-1).npu()
    merged_history = fused_dcp_lse_combine(history, 512, scatter_dim=0, return_lse=True)
    assert merged_history.shape == (num_tokens, 6, 513)
    current = torch.cat((outputs[-1], lse[-1].unsqueeze(-1)), dim=-1).npu()
    actual = fused_dcp_lse_combine(torch.stack((merged_history, current)), 512, scatter_dim=0)
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
    actual = fused_dcp_lse_combine(
        recv, head_dim, scatter_dim, return_lse=True, local_output=local, local_lse=local_lse
    )
    values = torch.cat((history, local.float().unsqueeze(0)))
    lses = torch.cat((history_lse, local_lse.unsqueeze(0)))[..., 0]
    expected = _reference_merge(values, lses)
    expected_lse = torch.logsumexp(lses.masked_fill(~torch.isfinite(lses), -torch.inf), dim=0)
    torch.testing.assert_close(actual[..., :head_dim], expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual[..., head_dim], expected_lse, atol=1e-4, rtol=1e-4)
