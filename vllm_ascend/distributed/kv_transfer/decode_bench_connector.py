# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""DecodeBench adaptations for Ascend circular caches and FP8 cache writes.

Keep the runtime cache specs and their managers unchanged. Only the benchmark
connector interprets their fixed-block addressing and zero-initialization needs.
Remove these overrides when upstream supports the same circular semantics and
NPU-compatible FP8 filling. FP8 encoding/startup fill are upstream features
(vllm-project/vllm#58472); these overrides address unsupported NPU fill ops.
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


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_FP8_FILL_CHUNK_ELEMENTS = 1 << 24


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

    def _make_fill_values(
        self,
        size: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        fill_mean: float,
        fill_std: float,
    ) -> torch.Tensor:
        if dtype not in _FP8_DTYPES:
            return super()._make_fill_values(size, device, dtype, fill_mean, fill_std)
        # Keep upstream's finite-range clamp and encoding. Only CPU operations
        # create FP8 values; the NPU receives bytes, then views the same storage.
        if fill_std > 0:
            values = super()._make_fill_values(size, torch.device("cpu"), dtype, fill_mean, fill_std)
            return values.view(torch.uint8).to(device).view(dtype)
        scalar = super()._make_fill_values((), torch.device("cpu"), dtype, fill_mean, fill_std)
        encoded = scalar.view(torch.uint8).item()  # CPU scalar: no NPU synchronization.
        return torch.full(size, encoded, dtype=torch.uint8, device=device).view(dtype)

    def _fill_block_tensor(
        self,
        kv_cache: torch.Tensor,
        block_ids: list[int],
        fill_mean: float,
        fill_std: float,
        fill_dtype: torch.dtype,
    ) -> None:
        # Both allocation and indexed assignment avoid FP8 NPU kernels. Reuse
        # upstream block filtering and selection with an equal-width byte view.
        if fill_dtype in _FP8_DTYPES:
            kv_cache = kv_cache.view(torch.uint8)
        super()._fill_block_tensor(kv_cache, block_ids, fill_mean, fill_std, fill_dtype)

    def _fill_tensor(
        self,
        kv_cache: torch.Tensor,
        fill_mean: float,
        fill_std: float,
        fill_dtype: torch.dtype,
    ) -> None:
        if fill_dtype not in _FP8_DTYPES:
            return super()._fill_tensor(kv_cache, fill_mean, fill_std, fill_dtype)
        view = kv_cache.view(torch.uint8)
        if view.numel() == 0:
            return
        if fill_std == 0:
            scalar = self._make_fill_values((), torch.device("cpu"), fill_dtype, fill_mean, fill_std)
            view.fill_(scalar.view(torch.uint8).item())  # CPU scalar.
            return
        if view.ndim == 0:
            view.copy_(self._make_fill_values((), view.device, fill_dtype, fill_mean, fill_std).view(torch.uint8))
            return
        # As upstream does for startup fill, bound temporary host/device buffers
        # by chunking along rows; preserve strided views and their page padding.
        rows = max(1, _FP8_FILL_CHUNK_ELEMENTS // max(1, view[0].numel()))
        for chunk in view.split(rows):
            chunk.copy_(
                self._make_fill_values(chunk.shape, chunk.device, fill_dtype, fill_mean, fill_std).view(torch.uint8)
            )


class AscendDecodeBenchConnector(DecodeBenchConnector):
    """Reuse upstream DecodeBench with Ascend circular and FP8 compatibility."""

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig") -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        circular_groups = _get_ascend_circular_group_ids(kv_cache_config)
        if role == KVConnectorRole.SCHEDULER and circular_groups:
            self.connector_scheduler = AscendDecodeBenchConnectorScheduler(vllm_config, kv_cache_config)
        elif role == KVConnectorRole.WORKER:
            assert self.connector_worker is not None
            native_fp8 = any(
                spec.dtype in _FP8_DTYPES
                for group in kv_cache_config.kv_cache_groups
                for spec in iter_layer_specs(group.kv_cache_spec)
            )
            if circular_groups or native_fp8 or self.connector_worker._fp8_dtypes:
                self.connector_worker = AscendDecodeBenchConnectorWorker(vllm_config, kv_cache_config)
