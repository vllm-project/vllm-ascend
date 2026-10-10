# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-coordinate sparse patches for the HCCL transport."""

import math
from dataclasses import dataclass

import torch

SPARSE_HCCL_VALUE_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@dataclass
class SparseWeightPatch:
    """Replacement values at unique flat indices in a full checkpoint tensor.

    Names and shapes use checkpoint coordinates, not runtime TP/EP shards.
    NaN is reserved by the native patch loader for unchanged elements.
    """

    name: str
    indices: torch.Tensor
    values: torch.Tensor
    full_shape: tuple[int, ...]


def validate_sparse_patch(patch: SparseWeightPatch) -> None:
    """Reject invalid payloads before starting a worker lifecycle or collective."""
    if not isinstance(patch.name, str) or not patch.name:
        raise ValueError("Sparse checkpoint name must be non-empty")
    if patch.full_shape is None or any(type(dim) is not int or dim < 0 for dim in patch.full_shape):
        raise ValueError(f"Invalid checkpoint full_shape for {patch.name}")
    if patch.indices.dtype != torch.int32:
        raise ValueError(f"Sparse HCCL indices must be int32: {patch.name}")
    if patch.indices.ndim != 1 or patch.values.ndim != 1:
        raise ValueError(f"Sparse indices and values must be 1D: {patch.name}")
    if patch.indices.numel() != patch.values.numel():
        raise ValueError(f"Sparse indices and values must have matching lengths: {patch.name}")
    if patch.values.dtype not in SPARSE_HCCL_VALUE_DTYPES:
        raise ValueError(f"Sparse values require an HCCL-supported floating dtype: {patch.name}")
    # Host reads are required before workers enter HCCL.
    # Keep the bounds/duplicate checks together instead of reading each index.
    indices = patch.indices
    if indices.numel():
        ordered = indices.sort().values
        invalid = torch.stack(
            [
                ((indices < 0) | (indices >= math.prod(patch.full_shape))).any(),
                (ordered[1:] == ordered[:-1]).any(),
            ]
        ).any()
        if invalid.item():
            raise ValueError(f"Sparse indices out of range or duplicated: {patch.name}")
    if torch.isnan(patch.values).any().item():
        raise ValueError(f"Sparse values cannot contain NaN: {patch.name}")
