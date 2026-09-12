# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport-independent sparse runtime weight patch operations."""

from dataclasses import dataclass

import torch


@dataclass
class SparseWeightPatch:
    """A sparse in-place patch for one existing runtime parameter.

    Dense HCCL transfers every complete runtime parameter, whereas sparse HCCL
    transfers only flattened positions whose values need to change::

        Dense HCCL
        parameter name --------> full tensor shaped like the runtime parameter
                                  [v0, v1, v2, v3, v4, v5, ...]

        Sparse HCCL
        parameter name --------> existing runtime parameter
        flat int32 indices ----> [ 1,  4, ...]
                                  |   |
                                  v   v
        replacement values ---> [x1, x4, ...]

        receiver update ------> flat_parameter[indices] = values

    ``indices`` and ``values`` are both one-dimensional and have equal length.
    Indices address the contiguous, flattened runtime parameter rather than its
    checkpoint layout. The full parameter shape is transported separately as
    metadata and validated before the patch is applied.

    Attributes:
        name: vLLM runtime parameter name, not a checkpoint-format alias.
        indices: One-dimensional ``torch.int32`` flattened element indices.
        values: Replacement values with the target runtime parameter's dtype.
    """

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
) -> None:
    """Apply a validated patch to the selected flattened parameter values."""
    param = validate_sparse_patch(model, patch, expected_shape)
    flat_param = param.data.view(-1)
    flat_param.index_copy_(
        0,
        patch.indices.to(dtype=torch.long),
        patch.values,
    )
