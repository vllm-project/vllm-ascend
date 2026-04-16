# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

import torch
from vllm.triton_utils import tl, triton


def _cdiv(x: int, y: int) -> int:
    triton_cdiv = getattr(triton, "cdiv", None)
    if triton_cdiv is not None:
        return triton_cdiv(x, y)
    return (x + y - 1) // y


@triton.jit
def _build_chunk_counts_kernel(
    cu_seqlens_ptr,
    chunk_counts_ptr,
    num_seqs,
    chunk_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_seqs

    bos = tl.load(cu_seqlens_ptr + offsets, mask=mask, other=0).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + offsets + 1, mask=mask, other=0).to(tl.int32)
    seq_lens = eos - bos
    chunk_counts = (seq_lens + chunk_size - 1) // chunk_size

    tl.store(chunk_counts_ptr + offsets, chunk_counts, mask=mask)


@triton.jit
def _build_chunk_offsets_kernel(
    chunk_counts_ptr,
    out_offsets_ptr,
    num_seqs,
    ADD_ONE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets <= num_seqs
    prefix = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for seq_idx in range(0, num_seqs):
        chunk_count = tl.load(chunk_counts_ptr + seq_idx, mask=seq_idx < num_seqs, other=0).to(tl.int32)
        prefix += tl.where(mask & (offsets > seq_idx), chunk_count + ADD_ONE, 0)

    tl.store(out_offsets_ptr + offsets, prefix.to(out_offsets_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _build_final_chunk_indices_kernel(
    update_chunk_offsets_ptr,
    out_final_chunk_indices_ptr,
    num_seqs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_seqs
    final_indices = tl.load(update_chunk_offsets_ptr + offsets + 1, mask=mask, other=0).to(tl.int32) - 1
    tl.store(
        out_final_chunk_indices_ptr + offsets,
        final_indices.to(out_final_chunk_indices_ptr.dtype.element_ty),
        mask=mask,
    )


def _validate_optional_output(
    name: str,
    tensor: torch.Tensor | None,
    *,
    expected_shape: tuple[int, ...] | None,
    expected_device: torch.device,
) -> None:
    if tensor is None:
        return
    if tensor.device != expected_device:
        raise ValueError(f"{name} must be on device {expected_device}, got {tensor.device}")
    if tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must have int32 or int64 dtype, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")


def _validate_cu_seqlens(cu_seqlens: torch.Tensor, chunk_size: int) -> None:
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("cu_seqlens must be a torch.Tensor")
    if cu_seqlens.device.type != "npu":
        raise ValueError(f"cu_seqlens must be on NPU, got {cu_seqlens.device}")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"cu_seqlens must have int32 or int64 dtype, got {cu_seqlens.dtype}")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be 1D, got shape {tuple(cu_seqlens.shape)}")
    if cu_seqlens.shape[0] < 1:
        raise ValueError("cu_seqlens must contain at least one element")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")


def _build_seq_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def _build_chunk_counts(seq_lens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_counts = torch.empty(
        seq_lens.shape[0],
        dtype=seq_lens.dtype,
        device=seq_lens.device,
    )
    if seq_lens.numel() == 0:
        return chunk_counts
    torch.div(
        seq_lens + chunk_size - 1,
        chunk_size,
        rounding_mode="floor",
        out=chunk_counts,
    )
    return chunk_counts


def _build_chunk_offsets(
    chunk_counts: torch.Tensor,
    out_offsets: torch.Tensor,
    *,
    add_one: int,
) -> None:
    out_offsets[0] = 0
    if chunk_counts.numel() > 0:
        torch.cumsum(chunk_counts + add_one, dim=0, out=out_offsets[1:])


def _build_final_chunk_indices(
    chunk_counts: torch.Tensor,
    update_chunk_offsets: torch.Tensor,
    out_final_chunk_indices: torch.Tensor,
) -> None:
    num_seqs = chunk_counts.shape[0]
    if hasattr(_build_final_chunk_indices_kernel, "__getitem__"):
        block_size = 256
        grid = (_cdiv(num_seqs, block_size),)
        _build_final_chunk_indices_kernel[grid](
            update_chunk_offsets_ptr=update_chunk_offsets,
            out_final_chunk_indices_ptr=out_final_chunk_indices,
            num_seqs=num_seqs,
            BLOCK_SIZE=block_size,
        )
        return

    if num_seqs > 0:
        torch.cumsum(chunk_counts + 1, dim=0, out=out_final_chunk_indices)
        out_final_chunk_indices.sub_(1)


def _build_chunk_meta_device_from_seq_lens(
    seq_lens: torch.Tensor,
    chunk_size: int,
    out_chunk_indices: torch.Tensor | None = None,
    out_chunk_offsets: torch.Tensor | None = None,
    out_update_chunk_offsets: torch.Tensor | None = None,
    out_final_chunk_indices: torch.Tensor | None = None,
) -> None:
    if (
        out_chunk_indices is None
        and out_chunk_offsets is None
        and out_update_chunk_offsets is None
        and out_final_chunk_indices is None
    ):
        return

    num_seqs = seq_lens.shape[0]
    expected_prefix_shape = (num_seqs + 1,)
    expected_final_shape = (num_seqs,)

    _validate_optional_output(
        "out_chunk_indices",
        out_chunk_indices,
        expected_shape=None,
        expected_device=seq_lens.device,
    )
    if out_chunk_indices is not None and (out_chunk_indices.ndim != 2 or out_chunk_indices.shape[1] != 2):
        raise ValueError(f"out_chunk_indices must have shape [num_chunks, 2], got {tuple(out_chunk_indices.shape)}")
    _validate_optional_output(
        "out_chunk_offsets",
        out_chunk_offsets,
        expected_shape=expected_prefix_shape,
        expected_device=seq_lens.device,
    )
    _validate_optional_output(
        "out_update_chunk_offsets",
        out_update_chunk_offsets,
        expected_shape=expected_prefix_shape,
        expected_device=seq_lens.device,
    )
    _validate_optional_output(
        "out_final_chunk_indices",
        out_final_chunk_indices,
        expected_shape=expected_final_shape,
        expected_device=seq_lens.device,
    )

    if num_seqs == 0:
        if out_chunk_offsets is not None:
            out_chunk_offsets.zero_()
        if out_update_chunk_offsets is not None:
            out_update_chunk_offsets.zero_()
        if out_final_chunk_indices is not None:
            out_final_chunk_indices.zero_()
        return

    chunk_counts = _build_chunk_counts(seq_lens, chunk_size)

    chunk_offsets = out_chunk_offsets
    if chunk_offsets is None and out_chunk_indices is not None:
        chunk_offsets = torch.empty(
            expected_prefix_shape,
            dtype=seq_lens.dtype,
            device=seq_lens.device,
        )
    update_chunk_offsets = out_update_chunk_offsets
    if update_chunk_offsets is None and out_final_chunk_indices is not None:
        update_chunk_offsets = torch.empty(
            expected_prefix_shape,
            dtype=seq_lens.dtype,
            device=seq_lens.device,
        )

    if chunk_offsets is not None:
        _build_chunk_offsets(chunk_counts, chunk_offsets, add_one=0)

    if update_chunk_offsets is not None:
        _build_chunk_offsets(chunk_counts, update_chunk_offsets, add_one=1)

    if out_final_chunk_indices is not None:
        _build_final_chunk_indices(
            chunk_counts,
            update_chunk_offsets,
            out_final_chunk_indices,
        )

    if out_chunk_indices is not None:
        total_chunks = out_chunk_indices.shape[0]
        if total_chunks == 0:
            return
        rows = torch.arange(total_chunks, device=seq_lens.device, dtype=chunk_offsets.dtype)
        compact_chunk_offsets = torch.unique_consecutive(chunk_offsets)
        seq_indices = torch.bucketize(rows, compact_chunk_offsets[1:], right=True)
        chunk_starts = compact_chunk_offsets.index_select(0, seq_indices)
        out_chunk_indices[:, 0].copy_(seq_indices.to(dtype=out_chunk_indices.dtype))
        out_chunk_indices[:, 1].copy_((rows - chunk_starts).to(dtype=out_chunk_indices.dtype))


def build_chunk_meta_device(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    out_chunk_indices: torch.Tensor | None = None,
    out_chunk_offsets: torch.Tensor | None = None,
    out_update_chunk_offsets: torch.Tensor | None = None,
    out_final_chunk_indices: torch.Tensor | None = None,
    *,
    seq_lens: torch.Tensor | None = None,
    validate_inputs: bool = True,
) -> None:
    if validate_inputs:
        _validate_cu_seqlens(cu_seqlens, chunk_size)
    elif chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    _build_chunk_meta_device_from_seq_lens(
        _build_seq_lens(cu_seqlens) if seq_lens is None else seq_lens,
        chunk_size,
        out_chunk_indices=out_chunk_indices,
        out_chunk_offsets=out_chunk_offsets,
        out_update_chunk_offsets=out_update_chunk_offsets,
        out_final_chunk_indices=out_final_chunk_indices,
    )
