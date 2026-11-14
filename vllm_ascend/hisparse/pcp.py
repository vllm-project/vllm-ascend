# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Separate PCP query rows from rank-major HiSparse KV writes.

C8 request-sharded target and replicated draft decode, including full graphs.
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


class PCPHiSparseConnector(ActiveKVConnector):
    """Use canonical requests for Core and gathered slots for Resident/Host DMA."""

    def __init__(self, connector, manager, block_tables, host_group_id):
        self.__dict__.update(connector.__dict__)
        self._pcp_manager = manager
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
            cache.slot_mapping = slots[group_id]
            cache.mirror_slot_mapping = slots[host_group_id]
            cache.source_block_table = tables[host_group_id]
            cache.runtime.hot_block_table = tables[hot_group_id]

    def prepare_draft_prefill(self):
        """Draft prefill reuses the target's PCP query and gathered KV layout."""
        slots = self._pcp_manager._gathered_kv_slot_mappings
        tables = self._pcp_manager._local_block_tables
        for cache, group_id, hot_group_id in self._pcp_bindings:
            if cache.draft_layer:
                cache.slot_mapping = slots[group_id]
                cache.mirror_slot_mapping = slots[self._pcp_host_group_id]
                cache.source_block_table = tables[self._pcp_host_group_id]
                cache.runtime.hot_block_table = tables[hot_group_id]

    def prepare_draft_decode(self, request_states, num_reqs):
        """Later draft steps are replicated; use global rows without KV gather."""
        states = self._draft_request_states
        states.fill_(-1)
        states[:num_reqs].copy_(request_states[:num_reqs])
        block_tables = self._global_block_tables
        for cache, group_id, hot_group_id in self._pcp_bindings:
            if cache.draft_layer:
                cache.slot_mapping = block_tables.slot_mappings[group_id]
                cache.mirror_slot_mapping = block_tables.slot_mappings[self._pcp_host_group_id]
                cache.source_block_table = block_tables.input_block_tables[self._pcp_host_group_id]
                cache.runtime.hot_block_table = block_tables.input_block_tables[hot_group_id]
                cache.runtime.request_state_indices = states

    def _build_load_kwargs(self, scheduler_output, input_batch, attn_metadata):
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
        local_tables = manager._local_block_tables
        assert local_tables is not None
        for cache, group_id, hot_group_id in self._pcp_bindings:
            cache.slot_mapping = slots[group_id, :count]
            cache.mirror_slot_mapping = slots[self._pcp_host_group_id, :count]
            # Source and Hot are indexed by local query request row. Resident
            # residency is indexed by persistent state slot and must stay intact.
            cache.source_block_table = local_tables[self._pcp_host_group_id]
            cache.runtime.hot_block_table = local_tables[hot_group_id]
        write_request_ids = get_pcp_write_request_ids(global_batch, indices, mask)
        for metadata in (attn_metadata or {}).values():
            metadata.hisparse_write_req_id_per_token = write_request_ids
        # Core owns request lifecycle and load scheduling. Only its load kwargs
        # need canonical requests and a gathered Resident slot snapshot length.
        load_kwargs = super()._build_load_kwargs(scheduler_output, global_batch, attn_metadata)
        load_kwargs["num_tokens"] = count
        return load_kwargs


def adapt_pcp_hisparse_connector(runner):
    # Worker patch registration imports the runner; defer this to avoid a cycle.
    from vllm_ascend.patch.worker.patch_hisparse import install_pcp_cache_layout, prepare_pcp_mirror_staging

    connector = runner.kv_connector
    if runner.pcp_manager is None:
        return connector

    if type(connector) is not ActiveKVConnector or type(connector.kv_connector) is not HiSparseConnector:
        raise ValueError("HiSparse PCP requires the direct HiSparseConnector")
    install_pcp_cache_layout()
    prepare_pcp_mirror_staging(connector.kv_connector.connector_worker)
    (host_group_id,) = runner.kv_cache_config.host_group_ids
    adapted = PCPHiSparseConnector(connector, runner.pcp_manager, runner.block_tables, host_group_id)
    if runner.speculator is not None:
        runner.speculator.hisparse_pcp_connector = adapted
    return adapted
