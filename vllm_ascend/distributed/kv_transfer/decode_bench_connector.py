# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""DecodeBench adaptation for Ascend's independently defined circular caches.

Keep the runtime cache specs and their managers unchanged. Only the benchmark
connector interprets their fixed-block addressing and zero-initialization needs.
Remove these overrides when upstream DecodeBench supports the same capability.
"""

from typing import TYPE_CHECKING

import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector,
    DecodeBenchConnectorScheduler,
    DecodeBenchConnectorWorker,
)
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import CircularBufferSpec, iter_layer_specs

from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


def _get_ascend_circular_group_ids(kv_cache_config: "KVCacheConfig") -> set[int]:
    """Groups with circular semantics that upstream's isinstance check misses."""
    return {
        group_idx
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups)
        if is_circular_kv_cache_spec(group.kv_cache_spec)
        and not all(isinstance(spec, CircularBufferSpec) for spec in iter_layer_specs(group.kv_cache_spec))
    }


class AscendDecodeBenchConnectorScheduler(DecodeBenchConnectorScheduler):
    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig") -> None:
        super().__init__(vllm_config, kv_cache_config)
        self._ascend_circular_group_ids = _get_ascend_circular_group_ids(kv_cache_config)
        self._startup_fill_group_ids.difference_update(self._ascend_circular_group_ids)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int) -> None:
        # Upstream exposes no per-group block-selection hook. Retain its
        # allocation checks and lifecycle here, extending only circular
        # recognition. Configurations without Ascend circular groups delegate.
        if not self._ascend_circular_group_ids:
            return super().update_state_after_alloc(request, blocks, num_external_tokens)
        if num_external_tokens == 0:
            return

        req_id = request.request_id
        total_computed_tokens = request.num_tokens - 1
        num_local_computed_tokens = total_computed_tokens - num_external_tokens
        block_ids_per_group: list[list[int]] = []
        for group_idx, (group, group_blocks, block_size) in enumerate(
            zip(self.kv_cache_groups, blocks.blocks, self.group_block_sizes, strict=True)
        ):
            if group_idx in self._startup_fill_group_ids or not getattr(group, "enable_kv_transfer", True):
                block_ids_per_group.append([])
                continue
            if is_circular_kv_cache_spec(group.kv_cache_spec):
                selected_blocks = group_blocks
            else:
                num_computed_blocks = cdiv(total_computed_tokens, block_size)
                external_block_start = num_local_computed_tokens // block_size
                assert 0 <= external_block_start <= num_computed_blocks <= len(group_blocks), (
                    "DecodeBenchConnector block range exceeds allocated blocks: "
                    f"request={req_id}, group={group_idx}, "
                    f"range=[{external_block_start}, {num_computed_blocks}), allocated={len(group_blocks)}"
                )
                selected_blocks = group_blocks[external_block_start:num_computed_blocks]
            block_ids = [block.block_id for block in selected_blocks if not block.is_null]
            if not block_ids:
                logger.warning(
                    "DecodeBenchConnector: No blocks selected for KV cache group %d "
                    "with %d external tokens for request %s",
                    group_idx,
                    num_external_tokens,
                    req_id,
                )
            block_ids_per_group.append(block_ids)

        self._filled_requests.add(req_id)
        if self.startup_fill and not any(block_ids_per_group):
            return
        self._pending_fills[req_id] = (tuple(block_ids_per_group), num_external_tokens)


class AscendDecodeBenchConnectorWorker(DecodeBenchConnectorWorker):
    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig") -> None:
        super().__init__(vllm_config, kv_cache_config)
        self._ascend_circular_group_ids = _get_ascend_circular_group_ids(kv_cache_config)
        self._startup_fill_group_ids.difference_update(self._ascend_circular_group_ids)
        self._zero_fill_group_ids.update(self._ascend_circular_group_ids)

    def _fill_blocks(self, group_idx: int, block_ids: list[int], num_tokens: int) -> None:
        if group_idx not in self._ascend_circular_group_ids:
            return super()._fill_blocks(group_idx, block_ids, num_tokens)
        if not block_ids:
            return
        assert self.kv_caches is not None
        for layer_name in self.group_to_layers[group_idx]:
            kv_cache = self.kv_caches[layer_name]
            tensors = (kv_cache,) if isinstance(kv_cache, torch.Tensor) else kv_cache
            # Ascend's circular tail views have a leading block dimension,
            # including list/tuple views. Zero only the requesting blocks:
            # upstream's whole-tensor state fill would reset live neighbours.
            for tensor in tensors:
                if min(block_ids) < 0 or max(block_ids) >= tensor.shape[0]:
                    raise ValueError("DecodeBenchConnector circular block IDs exceed the cache pool")
                self._fill_block_tensor(tensor, block_ids, 0.0, 0.0, tensor.dtype)


class AscendDecodeBenchConnector(DecodeBenchConnector):
    """Reuse upstream DecodeBench, adapting only Ascend circular cache groups."""

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig") -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if not _get_ascend_circular_group_ids(kv_cache_config):
            return
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = AscendDecodeBenchConnectorScheduler(vllm_config, kv_cache_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = AscendDecodeBenchConnectorWorker(vllm_config, kv_cache_config)
