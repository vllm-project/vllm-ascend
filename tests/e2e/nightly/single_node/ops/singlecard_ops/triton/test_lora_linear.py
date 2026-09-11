# SPDX-License-Identifier: Apache-2.0

import gc
import itertools

import pytest
import torch

from vllm_ascend.lora.lora_ops import (
    _LORA_TRITON_BLOCK_K,
    _LORA_TRITON_K_SPLIT,
    _LORA_TRITON_MAX_TOKENS,
    _lora_expand_kernel,
    _lora_expand_sliced_kernel,
    _lora_shrink_splitk_kernel,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

# TP-local Qwen3.5-27B inference shapes. The single-slice case covers the
# transposed-B path; QKV and QKVZ cover three- and four-slice packed weights.
PROJECTION_SHAPES = {
    "o_proj": (1536, (5120,)),
    "qkv_proj": (5120, (1024, 512, 512)),
    "qkvz_proj": (5120, (1536, 512, 512, 1536)),
}
TOKEN_CASES = [
    pytest.param(1, "o_proj", id="decode-1"),
    pytest.param(31, "qkv_proj", id="mtp-31"),
    pytest.param(128, "qkvz_proj", id="max-tokens-128"),
]


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@pytest.fixture(scope="module", autouse=True)
def init_triton_device():
    torch.npu.set_device(0)
    init_device_properties_triton()
    yield
    gc.collect()
    torch.npu.empty_cache()


def _make_packed_weights(
    slice_rank: int,
    hidden_size: int,
    output_slices: tuple[int, ...],
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, bool]:
    a_slices = [torch.randn(slice_rank, hidden_size, dtype=torch.bfloat16, device="npu") * 0.01 for _ in output_slices]
    b_slices = [
        torch.randn(output_size, slice_rank, dtype=torch.bfloat16, device="npu") * 0.01 for output_size in output_slices
    ]
    packed_a = torch.cat(a_slices)

    if len(output_slices) == 1:
        return packed_a, b_slices, b_slices[0].contiguous(), True

    total_rank = slice_rank * len(output_slices)
    total_output = sum(output_slices)
    packed_b = torch.zeros(total_rank, total_output, dtype=torch.bfloat16, device="npu")
    rank_start = 0
    output_start = 0
    for output_size, b_slice in zip(output_slices, b_slices):
        packed_b[
            rank_start : rank_start + slice_rank,
            output_start : output_start + output_size,
        ].copy_(b_slice.T)
        rank_start += slice_rank
        output_start += output_size
    return packed_a, b_slices, packed_b, False


def _launch_lora_kernels(
    x: torch.Tensor,
    packed_a: torch.Tensor,
    packed_b: torch.Tensor,
    y: torch.Tensor,
    adapter_mask: torch.Tensor,
    output_slices: tuple[int, ...],
    slice_rank: int,
    scale: float,
    b_transposed: bool,
) -> None:
    token_count = x.size(0)
    total_rank = packed_a.size(0)
    output_size = y.size(1)
    block_m = 16 if token_count <= 8 else 32
    block_n = 1024 if token_count <= 8 and total_rank <= 48 else 512
    workspace = torch.empty(
        _LORA_TRITON_K_SPLIT,
        _LORA_TRITON_MAX_TOKENS,
        total_rank,
        dtype=torch.float32,
        device="npu",
    )

    _lora_shrink_splitk_kernel[(_LORA_TRITON_K_SPLIT, _ceil_div(token_count, block_m))](
        x,
        packed_a,
        workspace,
        token_count,
        x.size(1),
        RANK=total_rank,
        WORKSPACE_TOKENS=_LORA_TRITON_MAX_TOKENS,
        BLOCK_M=block_m,
        BLOCK_K=_LORA_TRITON_BLOCK_K,
        K_SPLIT=_LORA_TRITON_K_SPLIT,
    )

    use_sliced_expand = len(output_slices) > 1 and slice_rank >= 32
    if use_sliced_expand:
        tile_counts = [_ceil_div(size, block_n) for size in output_slices]
        tile_ends = list(itertools.accumulate(tile_counts))
        output_starts = [0, *itertools.accumulate(output_slices)]
        while len(tile_ends) < 4:
            tile_ends.append(tile_ends[-1])
        while len(output_starts) < 5:
            output_starts.append(output_size)
        _lora_expand_sliced_kernel[(sum(tile_counts), _ceil_div(token_count, block_m))](
            workspace,
            packed_b,
            y,
            adapter_mask,
            token_count,
            output_size,
            scale,
            TOTAL_RANK=total_rank,
            SLICE_RANK=slice_rank,
            WORKSPACE_TOKENS=_LORA_TRITON_MAX_TOKENS,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            K_SPLIT=_LORA_TRITON_K_SPLIT,
            TILE_END_0=tile_ends[0],
            TILE_END_1=tile_ends[1],
            TILE_END_2=tile_ends[2],
            OUTPUT_START_1=output_starts[1],
            OUTPUT_START_2=output_starts[2],
            OUTPUT_START_3=output_starts[3],
        )
        return

    _lora_expand_kernel[(_ceil_div(output_size, block_n), _ceil_div(token_count, block_m))](
        workspace,
        packed_b,
        y,
        adapter_mask,
        token_count,
        output_size,
        scale,
        RANK=total_rank,
        WORKSPACE_TOKENS=_LORA_TRITON_MAX_TOKENS,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        K_SPLIT=_LORA_TRITON_K_SPLIT,
        B_TRANSPOSED=b_transposed,
    )


@pytest.mark.parametrize("slice_rank", [8, 16, 32, 64])
@pytest.mark.parametrize("token_count,projection_name", TOKEN_CASES)
@torch.inference_mode()
def test_lora_linear_triton_accuracy(
    slice_rank: int,
    token_count: int,
    projection_name: str,
) -> None:
    torch.manual_seed(123)
    hidden_size, output_slices = PROJECTION_SHAPES[projection_name]
    x = torch.randn(token_count, hidden_size, dtype=torch.bfloat16, device="npu")
    packed_a, b_slices, packed_b, b_transposed = _make_packed_weights(
        slice_rank,
        hidden_size,
        output_slices,
    )
    y_initial = torch.randn(
        token_count,
        sum(output_slices),
        dtype=torch.bfloat16,
        device="npu",
    )
    y = y_initial.clone()
    adapter_mask = (torch.arange(_LORA_TRITON_MAX_TOKENS, device="npu") % 3 != 1).to(torch.bfloat16)
    scale = 0.5

    _launch_lora_kernels(
        x,
        packed_a,
        packed_b,
        y,
        adapter_mask,
        output_slices,
        slice_rank,
        scale,
        b_transposed,
    )
    torch.npu.synchronize()

    a_slices = packed_a.split(slice_rank)
    deltas = [(x.float() @ a_slice.float().T) @ b_slice.float().T for a_slice, b_slice in zip(a_slices, b_slices)]
    expected = y_initial.float() + torch.cat(deltas, dim=1) * adapter_mask[:token_count, None].float() * scale
    difference = y.float() - expected
    relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected)
    assert relative_l2.item() < 0.005

    if token_count > 1:
        inactive = ~adapter_mask[:token_count].bool()
        torch.testing.assert_close(y[inactive], y_initial[inactive], rtol=0, atol=0)

    del x, packed_a, packed_b, b_slices, y_initial, y
    gc.collect()
    torch.npu.empty_cache()
