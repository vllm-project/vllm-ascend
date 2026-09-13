# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The vllm-ascend contributors
"""COW adapter for token-fused MLA caches."""

from collections.abc import Iterable, Sequence

import torch
import vllm.v1.worker.gpu_model_runner as gpu_model_runner
import vllm.v1.worker.utils as utils
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy


def _is_fused_mla_pair(kv_cache: object) -> bool:
    """Recognize nope/rope slices from one token-fused MLA cache.

    This is a transitional inference: reshape still returns an ordinary tuple
    because the Ascend MLA implementation accesses ``kv_cache[0]/[1]``, so the
    fused-cache type is no longer available here.  The three invariants below
    identify only token-fused slices; a component-major pair keeps rope after
    the whole nope region and therefore fails the pointer check.
    """
    if not isinstance(kv_cache, tuple) or len(kv_cache) != 2:
        return False
    nope, rope = kv_cache
    if not isinstance(nope, torch.Tensor) or not isinstance(rope, torch.Tensor):
        return False
    return (
        nope.stride(0) == rope.stride(0)
        and nope.untyped_storage().data_ptr() == rope.untyped_storage().data_ptr()
        and rope.data_ptr() == nope.data_ptr() + nope.shape[-1] * nope.element_size()
    )


def _fused_page_view(nope: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Return a byte view covering complete fused MLA kernel slots."""
    if not _is_fused_mla_pair((nope, rope)):
        raise ValueError("Expected a token-fused MLA (nope, rope) cache pair")

    # stride(0) is the complete physical kernel-slot stride, so the byte view
    # covers token-fused nope+rope plus any trailing hybrid padding.  Convert
    # nope's typed storage offset to a byte offset before binding the uint8 view.
    slot_bytes = nope.stride(0) * nope.element_size()
    raw = torch.empty(0, dtype=torch.uint8, device=nope.device)
    raw.set_(nope.untyped_storage())
    return torch.as_strided(
        raw,
        size=(nope.shape[0], slot_bytes),
        stride=(slot_bytes, 1),
        storage_offset=nope.storage_offset() * nope.element_size(),
    )


def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[object],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
) -> None:
    """Copy whole fused slots, then delegate to upstream block copy."""
    # Upstream copy expects one tensor per cache entry.  Convert a fused MLA
    # tuple to its whole-slot byte view; flatten ordinary tuples into their
    # component tensors and pass every other cache protocol through unchanged.
    copy_caches: list[object] = []
    for kv_cache in kv_caches:
        if _is_fused_mla_pair(kv_cache):
            nope, rope = kv_cache
            copy_caches.append(_fused_page_view(nope, rope))
        elif isinstance(kv_cache, tuple):
            copy_caches.extend(kv_cache)
        else:
            copy_caches.append(kv_cache)

    utils._orig_copy_kv_cache_blocks_inplace(
        copy_caches,
        num_blocks,
        kv_cache_block_copies,
    )


# GPUModelRunner imports this symbol into its own module, so replace both
# bindings. Keep the original implementation for ordinary tensor caches.
if not hasattr(utils, "_orig_copy_kv_cache_blocks_inplace"):
    utils._orig_copy_kv_cache_blocks_inplace = utils.copy_kv_cache_blocks_inplace
utils.copy_kv_cache_blocks_inplace = copy_kv_cache_blocks_inplace
gpu_model_runner.copy_kv_cache_blocks_inplace = copy_kv_cache_blocks_inplace
