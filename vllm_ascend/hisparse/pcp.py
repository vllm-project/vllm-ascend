# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Separate PCP query rows from rank-major HiSparse KV writes.

C8 PCP prefill and request-sharded decode, including full graphs.
Core retains residency scheduling, DMA and the Host write completion fence.
"""

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.v1.kv_cache_interface import HiSparseHotSpec, HiSparseResidentSpec
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def prepare_pcp_query_metadata(metadata, states, local_states, world_size):
    """Keep graph inputs at fixed addresses across capture and real steps."""
    states.fill_(-1)
    states[: local_states.numel()].copy_(local_states)
    count = metadata.num_input_tokens * world_size
    # Keep the fixed backing address, but expose only this batch's request rows.
    metadata.hisparse_request_state_indices = states[: local_states.numel()]
    metadata.hisparse_num_write_tokens = count
    # Gathered slots already mark padding with -1; scatter needs no row counter.
    metadata.hisparse_num_valid_write_rows = None
    metadata.hisparse_write_has_prefill = metadata.pcp_has_global_prefill


def get_pcp_write_request_ids(global_batch, gather_indices, write_mask):
    rows = torch.bucketize(gather_indices.contiguous(), global_batch.query_start_loc[1:].contiguous(), right=True).to(
        torch.int32
    )
    rows.masked_fill_(~write_mask, -1)
    return rows


def prepare_pcp_query_states(cache, metadata):
    """Bind local query states before the leader prefetches its followers."""
    states = getattr(metadata, "hisparse_request_state_indices", None)
    if states is not None:
        for member in cache.index_group_caches:
            member.runtime.request_state_indices = states
            member.host_mirror_required = metadata.hisparse_write_has_prefill or member.runtime.eager_host_mirror


def hisparse_write_target(cache, metadata, num_input_rows, num_slot_rows):
    """Gathered writes include padding and must not use Core's decode row cap."""
    count = getattr(metadata, "hisparse_num_write_tokens", None)
    if count is None:
        return cache.write_target(num_input_rows, num_slot_rows)
    assert cache.view is not None and cache.slot_mapping is not None
    count = min(num_input_rows, num_slot_rows, count)
    return cache.view.cache, cache.slot_mapping[:count], count


def prepare_pcp_mirror_staging(worker):
    """Size pinned Host slot staging for gathered rows before graph capture."""
    config = worker.vllm_config
    capacity = (
        config.scheduler_config.max_num_batched_tokens * config.parallel_config.prefill_context_parallel_size
        + config.scheduler_config.max_num_seqs * (config.num_lookahead_tokens + 1)
    )
    staging = worker._slot_mapping_staging
    if staging is not None and staging.slots.numel() < capacity:
        staging.slots = torch.empty(capacity, dtype=staging.slots.dtype, pin_memory=True)


class PCPHiSparseConnector(ActiveKVConnector):
    """Use canonical requests for Core and gathered slots for Resident/Host DMA."""

    def __init__(self, connector, manager, block_tables, host_group_id):
        self.__dict__.update(connector.__dict__)
        self._pcp_manager = manager
        self._pcp_group_metadata = []
        self._pcp_write_request_ids = None
        self._global_block_tables = block_tables
        worker = self.kv_connector.connector_worker
        groups = worker.kv_cache_config.kv_cache_groups
        resident_groups = [i for i, group in enumerate(groups) if isinstance(group.kv_cache_spec, HiSparseResidentSpec)]
        hot_groups = [
            i
            for i, group in enumerate(groups)
            if isinstance(group.kv_cache_spec, HiSparseHotSpec)
            for _ in group.layer_names
        ]
        self._pcp_bindings = []
        for cache, hot_group in zip(worker.cache_handles, hot_groups, strict=True):
            self._pcp_bindings.append((cache, resident_groups[cache.runtime.resident_source_index], hot_group))
        self._pcp_host_group_id = host_group_id
        # Bind before graph capture, which bypasses connector.pre_forward.
        slots = manager._gathered_kv_slot_mappings
        tables = manager._local_block_tables
        assert slots is not None and tables is not None
        for cache, group_id, hot_group_id in self._pcp_bindings:
            self._bind_cache_layout(cache, group_id, hot_group_id, slots, tables)

    def _bind_cache_layout(self, cache, group_id, hot_group_id, slots, tables):
        cache.slot_mapping = slots[group_id]
        cache.mirror_slot_mapping = slots[self._pcp_host_group_id]
        # Resident keeps its persistent state table; Source/Hot use query rows.
        cache.source_block_table = tables[self._pcp_host_group_id]
        cache.runtime.hot_block_table = tables[hot_group_id]

    def prepare_draft_prefill(self):
        slots = self._pcp_manager._gathered_kv_slot_mappings
        tables = self._pcp_manager._local_block_tables
        for cache, group_id, hot_group_id in self._pcp_bindings:
            if cache.draft_layer:
                self._bind_cache_layout(cache, group_id, hot_group_id, slots, tables)

    def prepare_draft_decode(self):
        """Later draft steps are replicated; use global rows without KV gather."""
        states = self.kv_connector.connector_worker.request_state_indices
        block_tables = self._global_block_tables
        for cache, group_id, hot_group_id in self._pcp_bindings:
            if cache.draft_layer:
                self._bind_cache_layout(
                    cache, group_id, hot_group_id, block_tables.slot_mappings, block_tables.input_block_tables
                )
                cache.runtime.request_state_indices = states

    def _build_load_kwargs(self, scheduler_output, input_batch, attn_metadata):
        self._pcp_group_metadata = []
        self._pcp_write_request_ids = None
        if input_batch is None:
            return super()._build_load_kwargs(scheduler_output, input_batch, attn_metadata)
        if attn_metadata is None and is_forward_context_available():
            attn_metadata = get_forward_context().attn_metadata
        manager = self._pcp_manager
        global_batch = manager.global_batch
        write_request_ids = None
        if manager.is_decode_sharded or global_batch.is_prefilling_np.any():
            write_request_ids = get_pcp_write_request_ids(
                global_batch, manager._padded_gather_idx, manager._gathered_kv_write_mask
            )
        count = write_request_ids.numel() if write_request_ids is not None else input_batch.num_tokens_after_padding
        self._pcp_write_request_ids = write_request_ids
        metadata_by_layer = attn_metadata or {}
        self._pcp_group_metadata = [
            (cache, metadata)
            for name, cache in self.kv_connector.connector_worker._group_leaders
            if (metadata := metadata_by_layer.get(name)) is not None
        ]
        # Keep Core's lifecycle canonical and snapshot all gathered slots.
        load_kwargs = super()._build_load_kwargs(scheduler_output, global_batch, attn_metadata)
        load_kwargs["num_tokens"] = count
        return load_kwargs

    def _start_load_kv(self):
        super()._start_load_kv()
        self.prepare_draft_prefill()
        # FULL replay also needs the resolver bound to local query states.
        for cache, metadata in self._pcp_group_metadata:
            prepare_pcp_query_states(cache, metadata)

    def prepare_host_mirror(self):
        for leader, metadata in self._pcp_group_metadata:
            for cache in leader.index_group_caches:
                cache.num_actual_tokens = metadata.hisparse_num_write_tokens
                cache.req_id_per_token = (
                    metadata.req_id_per_token if self._pcp_write_request_ids is None else self._pcp_write_request_ids
                )
                cache.host_mirror_required = metadata.hisparse_write_has_prefill or cache.runtime.eager_host_mirror

    def finish_forward(self):
        if not self._disabled:
            self.prepare_host_mirror()
        return super().finish_forward()


def adapt_pcp_hisparse_connector(runner):
    connector = runner.kv_connector
    if runner.pcp_manager is None:
        return connector

    # Taking over ActiveKVConnector state must not discard a subclass's lifecycle.
    if type(connector) is not ActiveKVConnector or type(connector.kv_connector) is not HiSparseConnector:
        raise ValueError("HiSparse PCP requires the direct HiSparseConnector")
    prepare_pcp_mirror_staging(connector.kv_connector.connector_worker)
    (host_group_id,) = runner.kv_cache_config.host_group_ids
    adapted = PCPHiSparseConnector(connector, runner.pcp_manager, runner.block_tables, host_group_id)
    if runner.speculator is not None:
        runner.speculator.hisparse_pcp_connector = adapted
    return adapted
