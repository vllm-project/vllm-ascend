# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Separate PCP query rows from rank-major HiSparse KV writes.

C8 PCP prefill and replicated decode, including full graphs.
Core retains residency scheduling, DMA and the Host write completion fence.
"""

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def prepare_pcp_query_metadata(metadata, states, valid_rows, local_states, world_size):
    """Keep graph inputs at fixed addresses across capture and real steps."""
    states.fill_(-1)
    states[: local_states.numel()].copy_(local_states)
    count = metadata.num_input_tokens * world_size
    valid_rows.fill_(count)
    metadata.hisparse_request_state_indices = states
    metadata.hisparse_num_write_tokens = count
    metadata.hisparse_num_valid_write_rows = valid_rows
    metadata.hisparse_write_has_prefill = metadata.pcp_has_global_prefill


def get_pcp_write_request_ids(global_batch, gather_indices, write_mask):
    """Map gathered KV writes to canonical requests, shared by all layers."""
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
        self._pcp_attn_metadata = {}
        self._global_block_tables = block_tables
        self._draft_request_states = torch.empty_like(block_tables.input_block_tables[0][:, 0], dtype=torch.int32)
        worker = self.kv_connector.connector_worker
        self._pcp_bindings = []
        for cache in worker.cache_handles:
            matches = [
                group
                for group, slots in enumerate(block_tables.slot_mappings)
                if slots.data_ptr() == cache.slot_mapping.data_ptr()
            ]
            if len(matches) != 1:
                raise RuntimeError("HiSparse Resident mapping must identify one KV group")
            hot_groups = [
                group
                for group, table in enumerate(block_tables.input_block_tables)
                if table.data_ptr() == cache.runtime.hot_block_table.data_ptr()
            ]
            if len(hot_groups) != 1:
                raise RuntimeError("HiSparse Hot table must identify one KV group")
            self._pcp_bindings.append((cache, matches[0], hot_groups[0]))
        self._pcp_host_group_id = host_group_id
        # Graph warmup/capture bypasses connector.pre_forward. Bind the PCP
        # persistent storage now so capture and replay use identical addresses.
        slots = manager._gathered_kv_slot_mappings
        tables = manager._local_block_tables
        assert slots is not None and tables is not None
        for cache, group_id, hot_group_id in self._pcp_bindings:
            self._bind_cache_layout(cache, group_id, hot_group_id, slots, tables)

    def _bind_cache_layout(self, cache, group_id, hot_group_id, slots, tables):
        cache.slot_mapping = slots[group_id]
        cache.mirror_slot_mapping = slots[self._pcp_host_group_id]
        # Source and Hot use local query request rows. Resident residency uses
        # persistent state slots and must retain its original block table.
        cache.source_block_table = tables[self._pcp_host_group_id]
        cache.runtime.hot_block_table = tables[hot_group_id]

    def prepare_draft_prefill(self):
        """Draft prefill reuses the target's PCP query and gathered KV layout."""
        slots = self._pcp_manager._gathered_kv_slot_mappings
        tables = self._pcp_manager._local_block_tables
        for cache, group_id, hot_group_id in self._pcp_bindings:
            if cache.draft_layer:
                self._bind_cache_layout(cache, group_id, hot_group_id, slots, tables)

    def prepare_draft_decode(self, request_states, num_reqs):
        """Later draft steps are replicated; use global rows without KV gather."""
        states = self._draft_request_states
        states.fill_(-1)
        states[:num_reqs].copy_(request_states[:num_reqs])
        block_tables = self._global_block_tables
        for cache, group_id, hot_group_id in self._pcp_bindings:
            if cache.draft_layer:
                self._bind_cache_layout(
                    cache, group_id, hot_group_id, block_tables.slot_mappings, block_tables.input_block_tables
                )
                cache.runtime.request_state_indices = states

    def _build_load_kwargs(self, scheduler_output, input_batch, attn_metadata):
        self._pcp_attn_metadata = {}
        if input_batch is None:
            return super()._build_load_kwargs(scheduler_output, input_batch, attn_metadata)
        # The eager runner passes metadata through the active forward context;
        # only graph capture supplies it explicitly to this connector.
        if attn_metadata is None and is_forward_context_available():
            attn_metadata = get_forward_context().attn_metadata
        manager = self._pcp_manager
        global_batch = manager.global_batch
        indices = manager._padded_gather_idx
        mask = manager._gathered_kv_write_mask
        slots = manager._gathered_kv_slot_mappings
        assert indices is not None and mask is not None and slots is not None
        count = indices.numel()
        if not manager.is_decode_sharded and not global_batch.is_prefilling_np.any():
            # Replicated decode writes once per rank without a KV collective.
            # Rank 0's gathered span has the same rows, including graph padding.
            count = input_batch.num_tokens_after_padding
            indices = torch.arange(count, device=indices.device)
            mask = indices < global_batch.num_tokens
        local_tables = manager._local_block_tables
        assert local_tables is not None
        for cache, group_id, hot_group_id in self._pcp_bindings:
            self._bind_cache_layout(cache, group_id, hot_group_id, slots[:, :count], local_tables)
        write_request_ids = get_pcp_write_request_ids(global_batch, indices, mask)
        for metadata in (attn_metadata or {}).values():
            metadata.hisparse_write_req_id_per_token = write_request_ids
        self._pcp_attn_metadata = attn_metadata or {}
        # Core owns request lifecycle and load scheduling. Only its load kwargs
        # need canonical requests and a gathered Resident slot snapshot length.
        load_kwargs = super()._build_load_kwargs(scheduler_output, global_batch, attn_metadata)
        load_kwargs["num_tokens"] = count
        return load_kwargs

    def _start_load_kv(self):
        super()._start_load_kv()
        # Core prepares handles even for FULL replay, where attention.forward
        # does not run in Python. Keep its request lifecycle canonical while
        # the resolver addresses the local PCP query rows.
        for name, cache in self.kv_connector.connector_worker._group_leaders:
            metadata = self._pcp_attn_metadata.get(name)
            if metadata is not None:
                prepare_pcp_query_states(cache, metadata)

    def prepare_host_mirror(self):
        """Project write rows only after attention has finished using query rows.

        Core's mirror and clear methods then consume their original fields;
        residency, DMA submission and completion fences remain unchanged.
        """
        for name, leader in self.kv_connector.connector_worker._group_leaders:
            metadata = self._pcp_attn_metadata.get(name)
            if metadata is None:
                continue
            for cache in leader.index_group_caches:
                cache.num_actual_tokens = metadata.hisparse_num_write_tokens
                cache.req_id_per_token = metadata.hisparse_write_req_id_per_token
                cache.host_mirror_required = metadata.hisparse_write_has_prefill or cache.runtime.eager_host_mirror

    def finish_forward(self):
        if not self._disabled:
            self.prepare_host_mirror()
        return super().finish_forward()


def adapt_pcp_hisparse_connector(runner):
    connector = runner.kv_connector
    if runner.pcp_manager is None:
        return connector

    if type(connector) is not ActiveKVConnector or type(connector.kv_connector) is not HiSparseConnector:
        raise ValueError("HiSparse PCP requires the direct HiSparseConnector")
    prepare_pcp_mirror_staging(connector.kv_connector.connector_worker)
    (host_group_id,) = runner.kv_cache_config.host_group_ids
    adapted = PCPHiSparseConnector(connector, runner.pcp_manager, runner.block_tables, host_group_id)
    if runner.speculator is not None:
        runner.speculator.hisparse_pcp_connector = adapted
    return adapted
