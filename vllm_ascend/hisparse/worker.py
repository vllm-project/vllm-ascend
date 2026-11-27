# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ascend worker ownership, selected by the external connector factory."""

from __future__ import annotations

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import worker as _core

from vllm_ascend.hisparse.backend import initialize_worker_dma
from vllm_ascend.hisparse.host_pool import AscendHiSparseOffloadManager, uses_memfabric_pd


class AscendHiSparseConnectorWorker(_core.HiSparseConnectorWorker):
    def initialize(
        self,
        cache_handles,
        cache_layer_names,
        max_num_reqs,
        host_num_blocks,
        device,
        pinned_host_pools,
        *,
        shared_host_region=None,
        is_host_writer=True,
    ):
        pool = cache_handles[0].runtime.host_pool
        manager = None
        if shared_host_region is not None or uses_memfabric_pd(pool.vllm_config.kv_transfer_config):
            manager = AscendHiSparseOffloadManager.from_host_pool(pool)
            is_host_writer = manager.is_host_writer
        super().initialize(
            cache_handles,
            cache_layer_names,
            max_num_reqs,
            host_num_blocks,
            device,
            pinned_host_pools,
            shared_host_region=None,
            is_host_writer=is_host_writer,
        )
        self.cache_layer_names = tuple(cache_layer_names)
        self.shared_host_region = manager.shared_region if manager is not None else None
        self._host_write_sync = (
            torch.zeros(1, dtype=torch.int32, device=device) if self.shared_host_region is not None else None
        )

    def _init_dma(self):
        super()._init_dma()
        initialize_worker_dma(self)

    def _restore_pages(self, transfers):
        mapping = None
        if self.shared_host_region is not None and not self.is_host_writer:
            mapping = self.shared_host_region.base_tensor._hisparse_registration
        with self.backend.host_to_device_dma(mapping):
            super()._restore_pages(transfers)

    def invalidate_blocks(self, block_ids, request_state_indices):
        owner = getattr(self, "_ascend_recycled_invalidator", None)
        if owner is None or not owner.try_call(self, block_ids, request_state_indices):
            return super().invalidate_blocks(block_ids, request_state_indices)

    def _finish_previous_step(self) -> None:
        super()._finish_previous_step()
        if self.shared_host_region is not None:
            # All ranks must observe TP0's completed local Host writes before reading Source.
            torch.accelerator.current_stream().wait_event(self.host_write_event)
            _core.get_tp_group().broadcast(self._host_write_sync, src=0)
