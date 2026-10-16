# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility shim over ``config_data``.

The v0.30.0 base renamed this module to ``metadata``. The layerwise
host-KV-offload stack keeps the ``config_data`` implementation (layerwise
range sessions, request-generation leases); this module re-exports it under
the upstream name so code aligned with the v0.30.0 module layout can import
``ascend_store.metadata`` unchanged, and carries the small helpers that only
existed on the renamed module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import *  # noqa: F401,F403
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (  # noqa: F401
    ReqMeta,
    get_block_hashes,
)


def block_hash_to_str(block_hash: BlockHash | str) -> str:
    return block_hash if isinstance(block_hash, str) else block_hash.hex()


def block_hash_to_bytes(block_hash: BlockHash | str) -> bytes:
    if isinstance(block_hash, str):
        if len(block_hash) == 64:
            try:
                return bytes.fromhex(block_hash)
            except ValueError:
                pass
        return block_hash.encode("utf-8")
    return bytes(block_hash)


def get_group_block_size(group_block_sizes: Sequence[int], group_id: int) -> int:
    return group_block_sizes[group_id] if group_id < len(group_block_sizes) else group_block_sizes[0]


def get_group_cache_family(group_cache_families: Sequence[str], group_id: int) -> str:
    return group_cache_families[group_id] if group_id < len(group_cache_families) else "default"


def infer_cache_transfer_granularity(
    group_block_sizes: Sequence[int],
    lcm_block_size: int,
    kv_cache_group_ids: Sequence[int],
) -> int:
    granularities = [lcm_block_size]
    for group_id in kv_cache_group_ids:
        granularities.append(get_group_block_size(group_block_sizes, group_id))
    return math.lcm(*granularities)


def infer_cacheable_group_ids(kv_cache_groups) -> list[int]:
    if not kv_cache_groups:
        return [0]
    group_ids = [i for i, group in enumerate(kv_cache_groups) if is_prefix_cacheable(group.kv_cache_spec)]
    assert group_ids, "AscendStore requires at least one prefix-cacheable KV cache group"
    return group_ids
