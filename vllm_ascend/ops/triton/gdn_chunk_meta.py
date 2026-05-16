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

"""Build GDN FLA chunk metadata without round-tripping through CPU helpers.

The GDN chunk kernels consume three kinds of operator metadata:

- ``chunk_indices``: rows of ``[compact_sequence_index, chunk_index]`` used by
  FLA kernels to locate each non-empty sequence chunk.
- ``chunk_offsets``: prefix offsets into ``chunk_indices`` for each sequence.
- ``update_chunk_offsets`` and ``final_chunk_indices``: update-path offsets and
  the final update row per sequence, where the update path reserves one extra
  row per sequence for recurrent state updates.
"""

import torch
from vllm.triton_utils import tl, triton


def _cdiv(x: int, y: int) -> int:
    """Return ceil(x / y), using Triton's helper when the runtime provides it."""
    triton_cdiv = getattr(triton, "cdiv", None)
    if triton_cdiv is not None:
        return triton_cdiv(x, y)
    return (x + y - 1) // y


# Input:
# - cu_seqlens_ptr: cumulative sequence offsets, shape [num_seqs + 1].
# - num_seqs: number of sequences represented by cu_seqlens_ptr.
# - chunk_size: token count per chunk.
# Output:
# - chunk_counts_ptr: per-sequence ceil(seq_len / chunk_size), shape [num_seqs].
# What it does:
# - Converts cumulative offsets into sequence lengths, then stores how many FLA
#   chunks each sequence needs for one chunk size.
@triton.jit
def _build_chunk_counts_kernel(
    cu_seqlens_ptr,
    chunk_counts_ptr,
    num_seqs,
    chunk_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_seqs = seq_offsets < num_seqs

    seq_start = tl.load(cu_seqlens_ptr + seq_offsets, mask=valid_seqs, other=0).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + seq_offsets + 1, mask=valid_seqs, other=0).to(tl.int32)
    seq_lens = seq_end - seq_start
    chunk_counts = (seq_lens + chunk_size - 1) // chunk_size

    tl.store(chunk_counts_ptr + seq_offsets, chunk_counts, mask=valid_seqs)


# Input:
# - chunk_counts_ptr: per-sequence chunk counts, shape [num_seqs].
# - num_seqs: number of sequences.
# - EXTRA_CHUNKS_PER_SEQ: 0 for normal chunk offsets, 1 for update offsets.
# Output:
# - out_offsets_ptr: prefix offsets, shape [num_seqs + 1].
# What it does:
# - Builds an exclusive prefix sum over chunk counts. The update path passes
#   EXTRA_CHUNKS_PER_SEQ=1 because it has one extra state-update row per
#   sequence.
@triton.jit
def _build_chunk_offsets_kernel(
    chunk_counts_ptr,
    out_offsets_ptr,
    num_seqs,
    EXTRA_CHUNKS_PER_SEQ: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    prefix_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_offsets = prefix_offsets <= num_seqs
    prefix_sums = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for seq_idx in range(0, num_seqs):
        chunk_count = tl.load(chunk_counts_ptr + seq_idx, mask=seq_idx < num_seqs, other=0).to(tl.int32)
        prefix_sums += tl.where(valid_offsets & (prefix_offsets > seq_idx), chunk_count + EXTRA_CHUNKS_PER_SEQ, 0)

    tl.store(out_offsets_ptr + prefix_offsets, prefix_sums.to(out_offsets_ptr.dtype.element_ty), mask=valid_offsets)


# Input:
# - update_chunk_offsets_ptr: update-path prefix offsets, shape [num_seqs + 1].
# - num_seqs: number of sequences.
# Output:
# - out_final_chunk_indices_ptr: final update row index per sequence, shape
#   [num_seqs].
# What it does:
# - Converts each sequence's exclusive end offset into the row index of its
#   final update entry.
@triton.jit
def _build_final_chunk_indices_kernel(
    update_chunk_offsets_ptr,
    out_final_chunk_indices_ptr,
    num_seqs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_seqs = seq_offsets < num_seqs
    final_indices = tl.load(update_chunk_offsets_ptr + seq_offsets + 1, mask=valid_seqs, other=0).to(tl.int32) - 1
    tl.store(
        out_final_chunk_indices_ptr + seq_offsets,
        final_indices.to(out_final_chunk_indices_ptr.dtype.element_ty),
        mask=valid_seqs,
    )


def _validate_optional_output(
    name: str,
    tensor: torch.Tensor | None,
    *,
    expected_shape: tuple[int, ...] | None,
    expected_device: torch.device,
) -> None:
    """Validate an optional output tensor before metadata is written.

    Args:
        name: Human-readable tensor name used in error messages.
        tensor: Optional output tensor supplied by the caller.
        expected_shape: Exact expected shape, or ``None`` when the caller-owned
            leading dimension is validated separately.
        expected_device: Device that must match the input sequence lengths.

    Returns:
        None. Raises ``ValueError`` when a supplied tensor cannot be written by
        the metadata builder.
    """
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
    """Validate the public device metadata-builder inputs.

    Args:
        cu_seqlens: Cumulative sequence offsets on NPU, shape ``[num_seqs + 1]``.
        chunk_size: Number of tokens per chunk for the metadata being built.

    Returns:
        None. Raises when the input cannot be consumed by the NPU metadata path.
    """
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
    """Return per-sequence lengths from cumulative sequence offsets.

    Args:
        cu_seqlens: Cumulative sequence offsets, shape ``[num_seqs + 1]``.

    Returns:
        A tensor of shape ``[num_seqs]`` where each element is ``eos - bos``.
    """
    return cu_seqlens[1:] - cu_seqlens[:-1]


def _build_chunk_counts(seq_lens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Return ceil(seq_len / chunk_size) for each sequence.

    Args:
        seq_lens: Per-sequence token lengths, shape ``[num_seqs]``.
        chunk_size: Number of tokens per chunk.

    Returns:
        A tensor of shape ``[num_seqs]`` with the number of chunks needed by
        each sequence.
    """
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
    extra_chunks_per_seq: int,
) -> None:
    """Build prefix offsets from per-sequence chunk counts.

    Args:
        chunk_counts: Per-sequence chunk counts, shape ``[num_seqs]``.
        out_offsets: Output prefix offsets, shape ``[num_seqs + 1]``.
        extra_chunks_per_seq: Extra rows to reserve per sequence.

    Returns:
        None. ``out_offsets`` is filled in place. ``extra_chunks_per_seq`` is 1
        for update offsets because the FLA update path reserves one extra
        state-update row per sequence.
    """
    out_offsets[0] = 0
    if chunk_counts.numel() > 0:
        torch.cumsum(chunk_counts + extra_chunks_per_seq, dim=0, out=out_offsets[1:])


def _build_final_chunk_indices(
    chunk_counts: torch.Tensor,
    update_chunk_offsets: torch.Tensor,
    out_final_chunk_indices: torch.Tensor,
) -> None:
    """Build the final update-row index for each sequence.

    Args:
        chunk_counts: Per-sequence chunk counts, shape ``[num_seqs]``.
        update_chunk_offsets: Prefix offsets for the update path, shape
            ``[num_seqs + 1]``. Each sequence reserves one extra update row.
        out_final_chunk_indices: Output tensor, shape ``[num_seqs]``.

    Returns:
        None. ``out_final_chunk_indices`` is filled in place with
        ``update_chunk_offsets[1:] - 1``.
    """
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
    """Build GDN chunk operator metadata from per-sequence lengths.

    Args:
        seq_lens: Per-sequence token lengths on the target device, shape
            ``[num_seqs]``.
        chunk_size: Number of tokens per chunk.
        out_chunk_indices: Optional output rows of
            ``[compact_sequence_index, chunk_index]``. The row count must match
            the sum of ``ceil(seq_len / chunk_size)`` over all sequences.
        out_chunk_offsets: Optional prefix offsets into ``out_chunk_indices``,
            shape ``[num_seqs + 1]``.
        out_update_chunk_offsets: Optional update-path prefix offsets, shape
            ``[num_seqs + 1]``. Each sequence reserves one extra update row.
        out_final_chunk_indices: Optional final update-row index per sequence,
            shape ``[num_seqs]``.

    Returns:
        None. Supplied output tensors are filled in place. Zero-length
        sequences are compacted out of ``out_chunk_indices`` to match
        ``prepare_chunk_indices``.
    """
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

    chunk_offsets_for_indices = out_chunk_offsets
    if chunk_offsets_for_indices is None and out_chunk_indices is not None:
        chunk_offsets_for_indices = torch.empty(
            expected_prefix_shape,
            dtype=seq_lens.dtype,
            device=seq_lens.device,
        )
    update_offsets_for_final_indices = out_update_chunk_offsets
    if update_offsets_for_final_indices is None and out_final_chunk_indices is not None:
        update_offsets_for_final_indices = torch.empty(
            expected_prefix_shape,
            dtype=seq_lens.dtype,
            device=seq_lens.device,
        )

    if chunk_offsets_for_indices is not None:
        _build_chunk_offsets(
            chunk_counts,
            chunk_offsets_for_indices,
            extra_chunks_per_seq=0,
        )

    if update_offsets_for_final_indices is not None:
        _build_chunk_offsets(
            chunk_counts,
            update_offsets_for_final_indices,
            extra_chunks_per_seq=1,
        )

    if out_final_chunk_indices is not None:
        _build_final_chunk_indices(
            chunk_counts,
            update_offsets_for_final_indices,
            out_final_chunk_indices,
        )

    if out_chunk_indices is not None:
        total_chunks = out_chunk_indices.shape[0]
        if total_chunks == 0:
            return
        # Runtime prepare_chunk_indices compacts away zero-length sequences.
        # Repeated prefix offsets identify those empty sequences on device.
        chunk_rows = torch.arange(
            total_chunks,
            device=seq_lens.device,
            dtype=chunk_offsets_for_indices.dtype,
        )
        compact_chunk_offsets = torch.unique_consecutive(chunk_offsets_for_indices)
        compact_seq_indices = torch.bucketize(
            chunk_rows,
            compact_chunk_offsets[1:],
            right=True,
        )
        chunk_starts = compact_chunk_offsets.index_select(0, compact_seq_indices)
        out_chunk_indices[:, 0].copy_(compact_seq_indices.to(dtype=out_chunk_indices.dtype))
        out_chunk_indices[:, 1].copy_((chunk_rows - chunk_starts).to(dtype=out_chunk_indices.dtype))


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
    """Build GDN chunk operator metadata on device.

    Args:
        cu_seqlens: Cumulative sequence offsets, shape ``[num_seqs + 1]``.
            The public path expects this tensor on NPU.
        chunk_size: Number of tokens per chunk for this metadata set.
        out_chunk_indices: Optional output rows of
            ``[compact_sequence_index, chunk_index]``.
        out_chunk_offsets: Optional prefix offsets into ``out_chunk_indices``.
        out_update_chunk_offsets: Optional update-path prefix offsets.
        out_final_chunk_indices: Optional final update-row indices.
        seq_lens: Optional precomputed sequence lengths. Passing this lets the
            caller validate ``cu_seqlens`` once and reuse the derived lengths
            across several chunk sizes.
        validate_inputs: Whether to run public input validation. Internal
            callers set this to ``False`` after validating shared inputs once.

    Returns:
        None. The provided output tensors are mutated in place and then passed
        to the GDN FLA operators.
    """
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
