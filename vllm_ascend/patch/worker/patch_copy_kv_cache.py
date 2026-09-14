# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The vllm-ascend contributors
"""Adapter for runner caches that expose multiple tensors as a tuple."""

from collections.abc import Iterable, Sequence

import vllm.v1.worker.gpu_model_runner as gpu_model_runner
import vllm.v1.worker.utils as utils
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy


def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[object],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
) -> None:
    """Flatten tuple caches, then delegate to upstream block copy."""
    copy_caches: list[object] = []
    for kv_cache in kv_caches:
        if isinstance(kv_cache, tuple):
            copy_caches.extend(kv_cache)
        else:
            copy_caches.append(kv_cache)

    utils._orig_copy_kv_cache_blocks_inplace(
        copy_caches,
        num_blocks,
        kv_cache_block_copies,
    )


# GPUModelRunner imports this symbol into its own module, so replace both
# bindings. Fused MLA and ordinary tensor caches are passed through unchanged.
if not hasattr(utils, "_orig_copy_kv_cache_blocks_inplace"):
    utils._orig_copy_kv_cache_blocks_inplace = utils.copy_kv_cache_blocks_inplace
utils.copy_kv_cache_blocks_inplace = copy_kv_cache_blocks_inplace
gpu_model_runner.copy_kv_cache_blocks_inplace = copy_kv_cache_blocks_inplace
