# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow GLM5 Next KPool compression/cache-write op."""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import direct_register_custom_op

TRITON_MAX_POOL_SIZE = 64
TRITON_MAX_HEAD_DIM = 1024

if HAS_TRITON:
    from vllm_ascend.ops.triton.glm5_next_kpool_compress import (
        glm5_next_kpool_compress_and_write_cache_triton,
    )
else:
    glm5_next_kpool_compress_and_write_cache_triton = None


def _validate_inputs(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    if slot_k.ndim != 3:
        raise ValueError(f"slot_k must be [N,P,D], got {slot_k.shape}.")
    if slot_k.dtype != torch.bfloat16:
        raise TypeError(f"slot_k must be bfloat16, got {slot_k.dtype}.")
    if slot_k.shape[1] <= 0 or slot_k.shape[2] <= 0:
        raise ValueError(f"slot_k must have positive pool/head dims, got {slot_k.shape}.")
    if slot_score.shape != slot_k.shape:
        raise ValueError(f"slot_score must match slot_k, got {slot_score.shape} and {slot_k.shape}.")
    if slot_score.dtype != torch.bfloat16:
        raise TypeError(f"slot_score must be bfloat16, got {slot_score.dtype}.")
    if ape.shape != slot_k.shape[1:]:
        raise ValueError(f"ape must be [P,D]={slot_k.shape[1:]}, got {ape.shape}.")
    if ape.dtype != torch.float32:
        raise TypeError(f"ape must be float32, got {ape.dtype}.")
    if loc.shape != (slot_k.shape[0],):
        raise ValueError(f"loc must have shape {(slot_k.shape[0],)}, got {loc.shape}.")
    if loc.dtype != torch.int64:
        raise TypeError(f"loc must be int64, got {loc.dtype}.")
    if kv_cache.ndim != 4 or kv_cache.shape[2:] != (1, slot_k.shape[2]):
        raise ValueError(f"kv_cache must be [blocks,block,1,{slot_k.shape[2]}], got {kv_cache.shape}.")
    if kv_cache.dtype != torch.bfloat16:
        raise TypeError(f"kv_cache must be bfloat16, got {kv_cache.dtype}.")
    for name, tensor in (
        ("kv_cache", kv_cache),
        ("slot_score", slot_score),
        ("ape", ape),
        ("loc", loc),
    ):
        if tensor.device != slot_k.device:
            raise ValueError(f"{name} must be on {slot_k.device}, got {tensor.device}.")


def _can_use_triton(slot_k: torch.Tensor) -> bool:
    if not HAS_TRITON or glm5_next_kpool_compress_and_write_cache_triton is None:
        return False
    if slot_k.device.type != "npu":
        return False
    if slot_k.shape[0] == 0:
        return False
    if slot_k.shape[1] > TRITON_MAX_POOL_SIZE:
        return False
    return slot_k.shape[2] <= TRITON_MAX_HEAD_DIM


def _fallback_kpool_compress_and_write_cache(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    if slot_k.shape[0] == 0:
        return

    scores = slot_score.float() + ape.float().unsqueeze(0)
    compressed_k = (torch.softmax(scores, dim=1) * slot_k.float()).sum(dim=1).to(torch.bfloat16)
    cache_block_size = kv_cache.shape[1]
    block_ids = torch.div(loc, cache_block_size, rounding_mode="floor")
    block_offsets = torch.remainder(loc, cache_block_size)
    kv_cache[block_ids, block_offsets, 0, :] = compressed_k


def glm5_next_kpool_compress_and_write_cache(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    """Compress ``slot_k`` with softmax(slot_score + ape) and write paged cache."""
    _validate_inputs(kv_cache, slot_k, slot_score, ape, loc)
    if _can_use_triton(slot_k):
        assert glm5_next_kpool_compress_and_write_cache_triton is not None
        glm5_next_kpool_compress_and_write_cache_triton(
            kv_cache,
            slot_k,
            slot_score,
            ape,
            loc,
        )
        return

    _fallback_kpool_compress_and_write_cache(
        kv_cache,
        slot_k,
        slot_score,
        ape,
        loc,
    )


def glm5_next_kpool_compress_and_write_cache_fake(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="glm5_next_kpool_compress_and_write_cache",
    op_func=glm5_next_kpool_compress_and_write_cache,
    mutates_args=["kv_cache"],
    fake_impl=glm5_next_kpool_compress_and_write_cache_fake,
    dispatch_key="PrivateUse1",
)
