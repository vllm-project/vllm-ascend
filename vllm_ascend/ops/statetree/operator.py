# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Graph-visible StateTree operator and load-time preparation helpers."""

from __future__ import annotations

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.statetree import native
from vllm_ascend.ops.statetree.codec import PackedWeight, pack
from vllm_ascend.ops.statetree.config import NATIVE_GEMM_MIN_ROWS


def _statetree_linear(
    x: torch.Tensor,
    payload: torch.Tensor,
    n: int,
    k: int,
    n_tile: int,
    k_tile: int,
    escape_slots: int,
) -> torch.Tensor:
    packed = PackedWeight(payload, n, k, n_tile, k_tile, escape_slots)
    x = x.contiguous()
    if x.numel() // k < NATIVE_GEMM_MIN_ROWS:
        return native.linear(x, packed)
    # Larger batches can select a different native GEMM reduction schedule.
    # Preserve that schedule; decode directly to its layout without a copy.
    return torch.nn.functional.linear(x, native.decode(packed))


def _statetree_linear_fake(
    x: torch.Tensor,
    payload: torch.Tensor,
    n: int,
    k: int,
    n_tile: int,
    k_tile: int,
    escape_slots: int,
) -> torch.Tensor:
    del payload, k, n_tile, k_tile, escape_slots
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    op_name="statetree_linear",
    op_func=_statetree_linear,
    fake_impl=_statetree_linear_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def apply(
    x: torch.Tensor,
    payload: torch.Tensor,
    shape: tuple[int, int],
    tiles: tuple[int, int, int],
) -> torch.Tensor:
    """Apply the opaque operator using immutable per-layer metadata."""
    n, k = shape
    n_tile, k_tile, escape_slots = tiles
    return torch.ops.vllm.statetree_linear(
        x,
        payload,
        n,
        k,
        n_tile,
        k_tile,
        escape_slots,
    )


@torch.no_grad()
def pack_verify_and_prepare(weight: torch.Tensor) -> PackedWeight:
    """Pack one weight, prove device reconstruction, then warm all kernels."""
    packed = pack(weight, device=weight.device)
    restored = native.decode(packed)
    if not torch.equal(
        restored.contiguous().view(torch.int16),
        weight.contiguous().view(torch.int16),
    ):
        raise RuntimeError(f"StateTree reconstruction is not bit-exact for weight shape {tuple(weight.shape)}")
    native.prepare(packed)
    return packed
