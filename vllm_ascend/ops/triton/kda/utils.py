# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
# mypy: ignore-errors

import contextlib
import functools
from collections.abc import Callable
from typing import Any

import torch
from vllm.triton_utils import triton

# Default chunk size used across FLA Triton kernels.
FLA_CHUNK_SIZE = 64


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Cache recent tensor-derived metadata keyed by tensor object identity."""
    cache_entries: tuple[tuple | None, dict | None, Any] = []
    cache_size = 8

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if (
                len(args) == len(last_args)
                and len(kwargs) == len(last_kwargs)
                and all(a is b for a, b in zip(args, last_args))
                and all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items())
            ):
                cache_entries = cache_entries[:i] + cache_entries[i + 1 :] + [(args, kwargs, last_result)]
                return last_result

        result = fn(*args, **kwargs)
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_indices_dev(
    cu_seqlens: torch.LongTensor,
    total_tokens: int,
    chunk_size: int,
) -> torch.LongTensor:
    """Device-side equivalent of :func:`prepare_chunk_indices`.

    ``prepare_chunk_indices`` ends with ``.tolist()``, i.e. a device->host synchronisation. On a
    serving hot path that stalls the NPU pipeline once per kernel launch, and it makes the
    surrounding op impossible to capture into an ACLGraph (host synchronisation is forbidden
    during capture). This variant never copies anything back to the host.

    The row count must be known on the host, so it is the shape-derived upper bound
    ``total_tokens // chunk_size + num_seqs`` instead of the exact
    ``sum(ceil(len_s / chunk_size))``. Rows past the real chunk count are phantom rows: their
    ``seq_id`` clamps to 0 while ``chunk_in_seq`` runs past the end of the first sequence, so
    every lane they address is fully masked and they store nothing (grid-only over-allocation,
    see the per-kernel masking audit in the PR description).

    Zero-length segments are not supported by either helper: ``prepare_chunk_indices`` numbers
    its ``seq_id`` over the non-empty sequences only, which does not match ``cu_seqlens``
    indexing, so callers have to drop empty segments first (the prefill path already does,
    see ``_compact_empty_segments``).

    Args:
        cu_seqlens: ``[num_seqs + 1]`` cumulative sequence lengths, on device.
        total_tokens: number of tokens in the packed batch, i.e. ``cu_seqlens[-1]``.
        chunk_size: chunk size the indices are built for.

    Returns:
        ``[total_chunks, 2]`` int tensor of ``(seq_id, chunk_in_seq)`` with the same dtype as
        ``cu_seqlens``. Its first ``sum(ceil(len_s / chunk_size))`` rows are element-wise
        identical to :func:`prepare_chunk_indices`.
    """
    num_seqs = cu_seqlens.numel() - 1
    if num_seqs <= 0:
        return torch.zeros((0, 2), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    total_chunks = total_tokens // chunk_size + num_seqs
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
    chunks_per_seq = (lens + (chunk_size - 1)) // chunk_size
    chunk_ends = torch.cumsum(chunks_per_seq, 0)
    row_ids = torch.arange(total_chunks, dtype=torch.int32, device=cu_seqlens.device)
    seq_ids = torch.searchsorted(chunk_ends, row_ids, right=True)
    seq_ids = torch.where(seq_ids >= num_seqs, torch.zeros_like(seq_ids), seq_ids)
    chunk_in_seq = row_ids - (chunk_ends - chunks_per_seq)[seq_ids]
    return torch.stack((seq_ids, chunk_in_seq), dim=1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = torch.npu.device(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper
