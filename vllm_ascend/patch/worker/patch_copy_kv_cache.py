# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The vllm-ascend contributors

from collections.abc import Iterable, Sequence

import torch
import vllm.v1.worker.gpu_model_runner as gpu_model_runner
import vllm.v1.worker.utils as utils
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

from vllm_ascend.worker.mla_component_cache_v1 import is_mla_component_pair as _is_component_pair


def _is_legacy_mla_pair(kv_cache: object) -> bool:
    """Recognize the legacy two-tensor contiguous MLA cache protocol."""
    if not isinstance(kv_cache, tuple) or len(kv_cache) != 2:
        return False
    nope, rope = kv_cache
    return (
        isinstance(nope, torch.Tensor)
        and isinstance(rope, torch.Tensor)
        and nope.ndim == 4
        and rope.ndim == 4
        and nope.dtype == rope.dtype
        and nope.device == rope.device
        and nope.shape[:3] == rope.shape[:3]
        and nope.shape[3] > 0
        and rope.shape[3] > 0
        and nope.is_contiguous()
        and rope.is_contiguous()
    )


def _component_page_view(nope: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Return a whole-slot byte view covering both MLA cache components."""
    if not _is_component_pair((nope, rope)):
        raise ValueError("Expected an MLA component-major (nope, rope) cache pair")

    element_size = nope.element_size()
    slot_bytes = nope.stride(0) * element_size
    raw = torch.empty(0, dtype=torch.uint8, device=nope.device)
    raw.set_(nope.untyped_storage())
    return torch.as_strided(
        raw,
        size=(nope.shape[0], slot_bytes),
        stride=(slot_bytes, 1),
        storage_offset=nope.storage_offset() * element_size,
    )


def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[object],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
) -> None:
    """Copy whole kernel slots when the runner holds component-major MLA tuples."""
    copy_caches: list[object] = []
    for kv_cache in kv_caches:
        if _is_component_pair(kv_cache):
            nope, rope = kv_cache
            copy_caches.append(_component_page_view(nope, rope))
        elif _is_legacy_mla_pair(kv_cache):
            copy_caches.extend(kv_cache)
        else:
            copy_caches.append(kv_cache)

    utils._orig_copy_kv_cache_blocks_inplace(
        copy_caches,
        num_blocks,
        kv_cache_block_copies,
    )


# Keep the original implementation available to ordinary tensor caches. The
# inherited GPUModelRunner resolves this symbol in its own module, so patch both
# bindings. Legacy MLA tuples are split into their two contiguous tensors before
# delegation; all other caches keep executing the upstream copy semantics.
if not hasattr(utils, "_orig_copy_kv_cache_blocks_inplace"):
    utils._orig_copy_kv_cache_blocks_inplace = utils.copy_kv_cache_blocks_inplace
utils.copy_kv_cache_blocks_inplace = copy_kv_cache_blocks_inplace
gpu_model_runner.copy_kv_cache_blocks_inplace = copy_kv_cache_blocks_inplace
