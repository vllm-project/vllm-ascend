# SPDX-License-Identifier: Apache-2.0

from math import prod
from typing import Any

import torch

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

# Large prefill writes benefit from grouped copies; small writes are faster
# with scatter. Keep a conservative crossover validated on A3 caches.
BLOCK_CACHE_STORE_MIN_TOKENS = 2048
BLOCK_CACHE_STORE_MAX_BYTES = 180 * 1024


def build_block_cache_groups(slot_mapping: torch.Tensor, block_size: int) -> dict[str, torch.Tensor]:
    """Group this cache's physical slots once for all layers sharing metadata."""
    groups = [torch.empty_like(slot_mapping, dtype=torch.int32) for _ in range(3)]
    torch.ops._C_ascend.store_kv_block_metadata(slot_mapping, *groups, block_size)
    return dict(zip(("group_len", "group_key_idx", "group_key_cache_idx"), groups))


def try_store_kv_blocks(key: torch.Tensor, cache: torch.Tensor, metadata: Any) -> bool:
    """Use the existing block-copy kernel only for supported large writes."""
    if (
        key.shape[0] < BLOCK_CACHE_STORE_MIN_TOKENS
        or metadata is None
        or getattr(metadata, "group_len", None) is None
        or getattr(metadata, "group_key_idx", None) is None
        or getattr(metadata, "group_key_cache_idx", None) is None
        or key.dtype not in (torch.int8, torch.float16, torch.bfloat16)
        or cache.dtype != key.dtype
        or not cache.is_contiguous()
        or not get_current_hardware_profile().supports(HardwareCapability.BLOCK_KV_CACHE_STORE)
    ):
        return False
    block_size = metadata.block_size
    block_bytes = block_size * prod(key.shape[1:]) * key.element_size()
    if block_size <= 0 or block_bytes > BLOCK_CACHE_STORE_MAX_BYTES:
        return False
    torch.ops._C_ascend.store_kv_block(
        key, cache, metadata.group_len, metadata.group_key_idx, metadata.group_key_cache_idx, block_size
    )
    return True
