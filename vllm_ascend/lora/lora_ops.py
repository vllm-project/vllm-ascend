#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools

import torch
from vllm.forward_context import get_forward_context, is_forward_context_available

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
):
    return torch.ops._C_ascend.bgmv_shrink(
        inputs,
        lora_a_weights,
        lora_indices_tensor,
        output_tensor,
        scaling,
    )


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
    return torch.ops._C_ascend.bgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        output_tensor,
        0,
        output_tensor.size(1),
    )


def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
    return torch.ops._C_ascend.bgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, output_tensor, slice_offset, slice_size
    )


def sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
):
    return torch.ops._C_ascend.sgmv_shrink(
        inputs, lora_a_weights, lora_indices_tensor, seq_len_tensor, output_tensor, scaling
    )


def sgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    add_inputs: bool = False,
):
    return torch.ops._C_ascend.sgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        seq_len_tensor,
        output_tensor,
        0,
        output_tensor.size(1),
    )


def sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
):
    return torch.ops._C_ascend.sgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, seq_len_tensor, output_tensor, slice_offset, slice_size
    )


_LORA_TRITON_RANKS = (8, 16, 32, 64)
_LORA_TRITON_MAX_TOKENS = 128
_LORA_TRITON_BLOCK_K = 256
_LORA_TRITON_K_SPLIT = 4


if _HAS_TRITON:

    @triton.jit
    def _lora_shrink_splitk_kernel(
        x_ptr,
        a_ptr,
        workspace_ptr,
        token_count,
        hidden_size,
        RANK: tl.constexpr,
        WORKSPACE_TOKENS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        K_SPLIT: tl.constexpr,
    ):
        split_id = tl.program_id(0)
        block_m_id = tl.program_id(1)
        token_offsets = block_m_id * BLOCK_M + tl.arange(0, BLOCK_M)
        token_mask = token_offsets < token_count
        k_per_split = tl.cdiv(hidden_size, K_SPLIT)
        k_start = split_id * k_per_split
        accum = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)
        for block_k_id in range(0, tl.cdiv(k_per_split, BLOCK_K)):
            k_offsets = k_start + block_k_id * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < tl.minimum(k_start + k_per_split, hidden_size)
            x = tl.load(
                x_ptr + token_offsets[:, None] * hidden_size + k_offsets[None, :],
                mask=token_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            a = tl.load(
                a_ptr + tl.arange(0, RANK)[:, None] * hidden_size + k_offsets[None, :],
                mask=k_mask[None, :],
                other=0.0,
            )
            accum += tl.dot(x, tl.trans(a))
        tl.store(
            workspace_ptr
            + split_id * WORKSPACE_TOKENS * RANK
            + token_offsets[:, None] * RANK
            + tl.arange(0, RANK)[None, :],
            accum.to(workspace_ptr.dtype.element_ty),
            mask=token_mask[:, None],
        )

    @triton.jit
    def _lora_expand_kernel(
        workspace_ptr,
        b_ptr,
        y_ptr,
        adapter_mask_ptr,
        token_count,
        output_size,
        scale,
        RANK: tl.constexpr,
        WORKSPACE_TOKENS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        K_SPLIT: tl.constexpr,
        B_TRANSPOSED: tl.constexpr,
    ):
        block_n_id = tl.program_id(0)
        block_m_id = tl.program_id(1)
        token_offsets = block_m_id * BLOCK_M + tl.arange(0, BLOCK_M)
        token_mask = token_offsets < token_count
        shrink = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)
        for split_id in range(0, K_SPLIT):
            shrink += tl.load(
                workspace_ptr
                + split_id * WORKSPACE_TOKENS * RANK
                + token_offsets[:, None] * RANK
                + tl.arange(0, RANK)[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        adapter_mask = tl.load(
            adapter_mask_ptr + token_offsets,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        shrink = shrink * adapter_mask[:, None] * scale
        output_offsets = block_n_id * BLOCK_N + tl.arange(0, BLOCK_N)
        output_mask = output_offsets < output_size
        if B_TRANSPOSED:
            b = tl.load(
                b_ptr + output_offsets[None, :] * RANK + tl.arange(0, RANK)[:, None],
                mask=output_mask[None, :],
                other=0.0,
            )
        else:
            b = tl.load(
                b_ptr + tl.arange(0, RANK)[:, None] * output_size + output_offsets[None, :],
                mask=output_mask[None, :],
                other=0.0,
            )
        delta = tl.dot(shrink.to(b.dtype), b)
        y_ptrs = y_ptr + token_offsets[:, None] * output_size + output_offsets[None, :]
        store_mask = token_mask[:, None] & output_mask[None, :]
        y = tl.load(y_ptrs, mask=store_mask, other=0.0)
        tl.store(y_ptrs, (y.to(tl.float32) + delta).to(y.dtype), mask=store_mask)

    @triton.jit
    def _lora_expand_sliced_kernel(
        workspace_ptr,
        b_ptr,
        y_ptr,
        adapter_mask_ptr,
        token_count,
        output_size,
        scale,
        TOTAL_RANK: tl.constexpr,
        SLICE_RANK: tl.constexpr,
        WORKSPACE_TOKENS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        K_SPLIT: tl.constexpr,
        TILE_END_0: tl.constexpr,
        TILE_END_1: tl.constexpr,
        TILE_END_2: tl.constexpr,
        OUTPUT_START_1: tl.constexpr,
        OUTPUT_START_2: tl.constexpr,
        OUTPUT_START_3: tl.constexpr,
    ):
        """Expand up to four packed slices without multiplying zero blocks."""
        block_n_id = tl.program_id(0)
        block_m_id = tl.program_id(1)
        slice_id = tl.where(
            block_n_id < TILE_END_0,
            0,
            tl.where(block_n_id < TILE_END_1, 1, tl.where(block_n_id < TILE_END_2, 2, 3)),
        )
        tile_start = tl.where(
            slice_id == 0,
            0,
            tl.where(slice_id == 1, TILE_END_0, tl.where(slice_id == 2, TILE_END_1, TILE_END_2)),
        )
        output_start = tl.where(
            slice_id == 0,
            0,
            tl.where(
                slice_id == 1,
                OUTPUT_START_1,
                tl.where(slice_id == 2, OUTPUT_START_2, OUTPUT_START_3),
            ),
        )
        output_end = tl.where(
            slice_id == 0,
            OUTPUT_START_1,
            tl.where(
                slice_id == 1,
                OUTPUT_START_2,
                tl.where(slice_id == 2, OUTPUT_START_3, output_size),
            ),
        )
        token_offsets = block_m_id * BLOCK_M + tl.arange(0, BLOCK_M)
        token_mask = token_offsets < token_count
        rank_offsets = tl.arange(0, SLICE_RANK)
        rank_start = slice_id * SLICE_RANK
        shrink = tl.zeros((BLOCK_M, SLICE_RANK), dtype=tl.float32)
        for split_id in range(0, K_SPLIT):
            shrink += tl.load(
                workspace_ptr
                + split_id * WORKSPACE_TOKENS * TOTAL_RANK
                + token_offsets[:, None] * TOTAL_RANK
                + rank_start
                + rank_offsets[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        adapter_mask = tl.load(adapter_mask_ptr + token_offsets, mask=token_mask, other=0.0)
        shrink = shrink * adapter_mask[:, None].to(tl.float32) * scale
        output_offsets = output_start + (block_n_id - tile_start) * BLOCK_N + tl.arange(0, BLOCK_N)
        output_mask = output_offsets < output_end
        b = tl.load(
            b_ptr + (rank_start + rank_offsets[:, None]) * output_size + output_offsets[None, :],
            mask=output_mask[None, :],
            other=0.0,
        )
        delta = tl.dot(shrink.to(b.dtype), b)
        y_ptrs = y_ptr + token_offsets[:, None] * output_size + output_offsets[None, :]
        store_mask = token_mask[:, None] & output_mask[None, :]
        y = tl.load(y_ptrs, mask=store_mask, other=0.0)
        tl.store(y_ptrs, (y.to(tl.float32) + delta).to(y.dtype), mask=store_mask)


def _lora_is_capturing() -> bool:
    try:
        return is_forward_context_available() and bool(get_forward_context().capturing)
    except (AttributeError, RuntimeError):
        return False


def _try_lora_linear_triton(
    wrapper,
    y: torch.Tensor,
    x: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    scale: float,
    output_slices: list[int],
    packed_lora_a: torch.Tensor | None,
    packed_lora_b: torch.Tensor | None,
) -> bool:
    if not (_HAS_TRITON and _lora_is_capturing()):
        return False
    x = x.view(-1, x.shape[-1])
    y = y.view(-1, y.shape[-1])
    token_count = x.size(0)
    if token_count > _LORA_TRITON_MAX_TOKENS or x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16:
        return False

    if len(lora_a_stacked) > 1:
        if packed_lora_a is None or packed_lora_b is None or sum(output_slices) != y.size(1):
            return False
        a = packed_lora_a[0, 0]
        b = packed_lora_b[0, 0]
        b_transposed = False
    else:
        a = lora_a_stacked[0][0, 0]
        if packed_lora_b is not None:
            b = packed_lora_b[0, 0]
            b_transposed = False
        else:
            b = lora_b_stacked[0][0, 0, : output_slices[0]]
            b_transposed = True

    if not (x.is_contiguous() and y.is_contiguous() and a.is_contiguous() and b.is_contiguous()):
        return False
    rank = a.size(0)
    slice_rank = lora_b_stacked[0].size(-1)
    output_size = y.size(1)
    if slice_rank not in _LORA_TRITON_RANKS or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        return False
    # Large MLP projections cross over to packed matmul sooner than attention
    # projections. Keep Triton through 128 tokens for attention-sized outputs,
    # while avoiding the measured regressions on gate/up and down projections.
    if output_size >= 8192 and token_count > (32 if slice_rank <= 16 else 64):
        return False
    if output_size >= 5120 and slice_rank >= 16 and token_count > 64:
        return False
    assert wrapper._single_lora_mask is not None
    workspaces = wrapper._lora_triton_workspaces
    workspace = workspaces.get(rank)
    if workspace is None:
        workspace = torch.empty(
            (_LORA_TRITON_K_SPLIT, _LORA_TRITON_MAX_TOKENS, rank),
            dtype=torch.float32,
            device=x.device,
        )
        workspaces[rank] = workspace

    block_m = 16 if token_count <= 8 else 32
    _lora_shrink_splitk_kernel[(_LORA_TRITON_K_SPLIT, triton.cdiv(token_count, block_m))](
        x,
        a,
        workspace,
        token_count,
        x.size(1),
        RANK=rank,
        WORKSPACE_TOKENS=_LORA_TRITON_MAX_TOKENS,
        BLOCK_M=block_m,
        BLOCK_K=_LORA_TRITON_BLOCK_K,
        K_SPLIT=_LORA_TRITON_K_SPLIT,
    )
    block_n = 1024 if token_count <= 8 and rank <= 48 else 512
    use_sliced_expand = len(output_slices) > 1 and slice_rank >= 32 and len(output_slices) <= 4
    if use_sliced_expand:
        tile_counts = [triton.cdiv(size, block_n) for size in output_slices]
        tile_ends = list(itertools.accumulate(tile_counts))
        output_starts = [0, *itertools.accumulate(output_slices)]
        while len(tile_ends) < 4:
            tile_ends.append(tile_ends[-1])
        while len(output_starts) < 5:
            output_starts.append(output_size)
        _lora_expand_sliced_kernel[(tile_ends[len(output_slices) - 1], triton.cdiv(token_count, block_m))](
            workspace,
            b,
            y,
            wrapper._single_lora_mask,
            token_count,
            output_size,
            scale,
            TOTAL_RANK=rank,
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
        return True
    _lora_expand_kernel[(triton.cdiv(output_size, block_n), triton.cdiv(token_count, block_m))](
        workspace,
        b,
        y,
        wrapper._single_lora_mask,
        token_count,
        output_size,
        scale,
        RANK=rank,
        WORKSPACE_TOKENS=_LORA_TRITON_MAX_TOKENS,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        K_SPLIT=_LORA_TRITON_K_SPLIT,
        B_TRANSPOSED=b_transposed,
    )
    return True


_LORA_WRAPPERS: dict = {}  # wrapper_id: int -> wrapper: PunicaWrapperNPU
_LORA_WRAPPER_IDS = itertools.count()


@torch.library.custom_op("_vllm_ascend_lora::lora_linear", mutates_args={"y"})
def lora_linear(
    wrapper_id: int,
    y: torch.Tensor,
    x: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    scale: float,
    output_slices: list[int],
    packed_lora_a: torch.Tensor | None,
    packed_lora_b: torch.Tensor | None,
    add_inputs: bool,
) -> None:
    wrapper = _LORA_WRAPPERS[wrapper_id]
    packed = len(lora_a_stacked) == 1 or (packed_lora_a is not None and packed_lora_b is not None)
    use_kernel = not wrapper._single_lora_slot or not packed
    if use_kernel:
        wrapper._lora_linear_kernel(y, x, lora_a_stacked, lora_b_stacked, scale, output_slices)
    else:
        if _try_lora_linear_triton(
            wrapper,
            y,
            x,
            lora_a_stacked,
            lora_b_stacked,
            scale,
            output_slices,
            packed_lora_a,
            packed_lora_b,
        ):
            return
        wrapper._lora_linear_matmul(
            y,
            x,
            lora_a_stacked,
            lora_b_stacked,
            scale,
            output_slices,
            packed_lora_a,
            packed_lora_b,
            add_inputs,
        )


@lora_linear.register_fake
def _lora_linear_fake(
    wrapper_id,
    y,
    x,
    lora_a_stacked,
    lora_b_stacked,
    scale,
    output_slices,
    packed_lora_a,
    packed_lora_b,
    add_inputs,
) -> None:
    pass
