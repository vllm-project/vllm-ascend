# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport-independent sparse runtime weight patch operations."""

from dataclasses import dataclass

import torch


@dataclass
class SparseWeightPatch:
    """A sparse in-place patch for one existing runtime parameter."""

    name: str
    indices: torch.Tensor
    values: torch.Tensor


def validate_sparse_patch(
    model: torch.nn.Module,
    patch: SparseWeightPatch,
    expected_shape: list[int] | None = None,
) -> torch.nn.Parameter:
    """Validate a flat-index patch and return its target parameter."""
    param = model.get_parameter(patch.name)
    if expected_shape is not None and list(param.shape) != expected_shape:
        raise ValueError(
            f"Sparse parameter shape {list(param.shape)} does not match "
            f"declared shape {expected_shape} for {patch.name}"
        )
    if not param.data.is_contiguous():
        raise NotImplementedError(
            "Sparse weight updates currently require contiguous params: "
            f"{patch.name}"
        )
    if patch.indices.dtype != torch.int32:
        raise ValueError(
            "Sparse weight updates currently require int32 indices: "
            f"{patch.name}"
        )
    if patch.indices.ndim != 1 or patch.values.ndim != 1:
        raise ValueError(
            f"Sparse weight patches must be 1D flattened updates: {patch.name}"
        )
    if patch.indices.numel() != patch.values.numel():
        raise ValueError(
            "`indices` and `values` must have matching lengths for "
            f"{patch.name}"
        )
    if patch.values.dtype != param.dtype:
        raise ValueError(
            f"Sparse values dtype {patch.values.dtype} does not match "
            f"parameter dtype {param.dtype} for {patch.name}"
        )
    return param


def apply_sparse_patch(
    model: torch.nn.Module,
    patch: SparseWeightPatch,
    expected_shape: list[int] | None = None,
    *,
    verify: bool = True,
) -> None:
    """Apply a validated patch and optionally require exact selected values."""
    param = validate_sparse_patch(model, patch, expected_shape)
    flat_param = param.data.view(-1)
    indices = patch.indices.to(device=flat_param.device, dtype=torch.long)
    values = patch.values.to(device=flat_param.device)
    flat_param.index_copy_(0, indices, values)
    if verify and not torch.equal(flat_param.index_select(0, indices), values):
        raise RuntimeError(
            f"Sparse weight update verification failed for {patch.name}"
        )
