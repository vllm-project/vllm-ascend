# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts at the PCP / HiSparse boundary; no device kernels are run."""

from dataclasses import dataclass
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker
from vllm.v1.hisparse.runtime import HiSparseCacheHandle
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

from vllm_ascend.hisparse.pcp import (
    PCPHiSparseConnector,
    get_pcp_write_request_ids,
    hisparse_write_target,
    prepare_pcp_query_metadata,
    prepare_pcp_query_states,
)


@pytest.mark.parametrize("pcp_world_size", [2, 4, 8])
@pytest.mark.parametrize("initial_capacity", [4, 128])
def test_pcp_host_staging_is_sized_before_capture(monkeypatch, initial_capacity, pcp_world_size):
    dtype = torch.int32
    from vllm_ascend.hisparse import pcp

    cache = handle()
    staging = NS(slots=torch.empty(initial_capacity, dtype=dtype))
    original = staging.slots
    worker = NS(
        vllm_config=NS(
            scheduler_config=NS(max_num_batched_tokens=8, max_num_seqs=2),
            parallel_config=NS(prefill_context_parallel_size=pcp_world_size),
            num_lookahead_tokens=3,
        ),
        _slot_mapping_staging=staging,
        cache_handles=[cache],
    )
    allocate = torch.empty
    allocations = []

    def pinned_empty(size, **kwargs):
        allocations.append((size, kwargs.copy()))
        kwargs.pop("pin_memory")
        return allocate(size, **kwargs)

    monkeypatch.setattr(pcp.torch, "empty", pinned_empty)
    pcp.prepare_pcp_mirror_staging(worker)
    capacity = 8 * pcp_world_size + 2 * (3 + 1)
    assert staging.slots.numel() >= capacity
    assert staging.slots.dtype == dtype
    if initial_capacity < capacity:
        assert allocations == [(capacity, {"dtype": dtype, "pin_memory": True})]
    else:
        assert staging.slots is original and not allocations


def test_pcp_connector_keeps_target_host_mirror_overlap(monkeypatch):
    from vllm.distributed import parallel_state
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector

    from vllm_ascend.hisparse import host_mirror
    from vllm_ascend.hisparse.backend import AscendHiSparseBackend

    worker = HiSparseConnectorWorker.__new__(HiSparseConnectorWorker)
    worker._initialized = True
    worker.backend = AscendHiSparseBackend.__new__(AscendHiSparseBackend)
    worker.shared_host_region = None
    worker.is_host_writer = True
    worker.hot_backing = NS(device=torch.device("npu:0"))
    core = HiSparseConnector.__new__(HiSparseConnector)
    core.connector_worker = worker
    connector = PCPHiSparseConnector.__new__(PCPHiSparseConnector)
    connector.kv_connector = core
    runner = NS(
        kv_connector=connector, speculator=object(), output_copy_stream=NS(stream_id=1, device_index=0, device_type=20)
    )
    compute = NS(device=worker.hot_backing.device, stream_id=0)
    output = NS(device=worker.hot_backing.device, stream_id=1)
    sources = object()
    monkeypatch.setattr(host_mirror._DirectMirrorSources, "bind", lambda *a: sources)
    monkeypatch.setattr(host_mirror.torch, "Stream", lambda **kwargs: output)
    monkeypatch.setattr(host_mirror.torch.accelerator, "current_stream", lambda: compute)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    adapted = host_mirror.overlap_target_host_mirror(runner)
    assert isinstance(adapted, host_mirror.HostMirrorConnector)
    assert adapted._connector is connector and adapted._sources is sources


@dataclass
class Batch:
    num_tokens: int
    idx_mapping: torch.Tensor
    req_ids: list
    query_start_loc: torch.Tensor
    is_prefilling_np: np.ndarray


def metadata(num_tokens=3, decode=True):
    return NS(
        num_actual_tokens=num_tokens,
        num_decode_tokens=num_tokens if decode else 0,
        max_query_len=1 if decode else max(num_tokens, 2),
        num_reqs=num_tokens,
        req_id_per_token=torch.arange(num_tokens, dtype=torch.int32),
        query_start_loc=torch.arange(num_tokens + 1, dtype=torch.int32),
    )


def handle():
    runtime = NS(
        begin_forward=Mock(),
        eager_host_mirror=False,
        max_num_reqs=2,
        max_swap_rows=16,
        request_state_indices=torch.tensor([7, 9], dtype=torch.int32),
        is_group_leader=True,
    )
    cache = HiSparseCacheHandle(runtime)
    cache.view = NS(cache=torch.zeros(12, 8, dtype=torch.uint8))
    cache.slot_mapping = torch.tensor([10, 11, -1, 12, 13, -1])
    return cache


def gathered_metadata(local_tokens=3, prefill=False):
    # Request A owns global rows 0..3; B owns 4..6. Two PCP chunks
    # for A deliberately create duplicate local request-state indices.
    global_batch = Batch(
        7,
        torch.tensor([7, 9], dtype=torch.int32),
        ["A", "B"],
        torch.tensor([0, 4, 7], dtype=torch.int32),
        np.array([prefill, False]),
    )
    local = Batch(
        local_tokens,
        torch.tensor([7, 7, 9], dtype=torch.int64),
        ["A", "A", "B"],
        torch.tensor([0, 1, 2, local_tokens]),
        np.array([prefill, prefill, False]),
    )
    indices = torch.tensor([0, 3, 0, 1, 4, 0], dtype=torch.int64)
    mask = torch.tensor([True, True, False, True, True, False])
    meta = metadata(local_tokens, decode=not prefill)
    meta.num_input_tokens = indices.numel() // 2
    meta.pcp_has_global_prefill = prefill
    prepare_pcp_query_metadata(
        meta,
        torch.empty(local.idx_mapping.shape, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        local.idx_mapping,
        2,
    )
    meta.hisparse_write_req_id_per_token = get_pcp_write_request_ids(global_batch, indices, mask)
    return meta, local, global_batch, indices, mask


def test_global_write_ids_and_local_dual_chunk_states_are_distinct():
    meta, local, _, _, _ = gathered_metadata()
    assert meta.hisparse_write_req_id_per_token.tolist() == [0, 0, -1, 0, 1, -1]
    assert meta.hisparse_request_state_indices.dtype == torch.int32
    assert meta.hisparse_request_state_indices.tolist() == local.idx_mapping.tolist()
    assert meta.hisparse_num_valid_write_rows.tolist() == [6]
    cache = handle()
    canonical_states = cache.runtime.request_state_indices
    cache._prepare_for_batch(meta)
    prepare_pcp_query_states(cache, meta)
    assert canonical_states.tolist() == [7, 9]
    assert cache.runtime.request_state_indices.tolist() == [7, 7, 9]


@pytest.mark.parametrize("prefill", [False, True])
def test_write_target_keeps_gathered_tail_and_masks_padding(prefill):
    meta, *_ = gathered_metadata(prefill=prefill)
    cache = handle()
    cache._prepare_for_batch(meta)
    prepare_pcp_query_states(cache, meta)
    _, slots, count = hisparse_write_target(cache, meta, 6, 6)
    assert count == 6  # local queries=3, max requests=2 must not truncate KV.
    assert slots.tolist() == [10, 11, -1, 12, 13, -1]
    assert cache.host_mirror_required == prefill


def test_zero_local_queries_still_publish_global_prefill(monkeypatch):
    from vllm.config.compilation import CUDAGraphMode

    import vllm_ascend.attention.sfa_hisparse as module

    meta, *_ = gathered_metadata(local_tokens=0, prefill=True)
    cache = handle()
    cache.submit_layer_mirror = Mock()
    cache._prepare_for_batch(meta)
    prepare_pcp_query_states(cache, meta)
    monkeypatch.setattr(module, "get_forward_context", lambda: NS(cudagraph_runtime_mode=CUDAGraphMode.NONE))
    assert cache.num_actual_tokens == 0
    assert cache.host_mirror_required
    module.finish_hisparse_kv_update(cache, meta)
    cache.submit_layer_mirror.assert_called_once()


def make_connector():
    meta, local, global_batch, indices, mask = gathered_metadata(prefill=True)
    original_slots = torch.tensor([[50, 51, 52], [100, 101, 102]])
    cache = handle()
    cache.slot_mapping = original_slots[1]
    original_tables = (torch.tensor([[50], [60]]), torch.tensor([[100], [110]]))
    cache.runtime.hot_block_table = original_tables[1]
    cache.block_table = torch.tensor([[901], [902]])
    underlying = NS(
        connector_worker=NS(cache_handles=[cache], _group_leaders=[("layer0", cache)]),
        handle_preemptions=Mock(),
        bind_connector_metadata=Mock(),
    )
    active = ActiveKVConnector.__new__(ActiveKVConnector)
    active.kv_connector = underlying
    active._disabled = False
    active._pending_load_kwargs = None
    active._start_load_kv = Mock()
    gathered_slots = torch.tensor([[50, 53, -1, 51, 54, -1], [100, 103, -1, 101, 104, -1]])
    manager = NS(
        is_decode_sharded=False,
        global_batch=global_batch,
        _padded_gather_idx=indices,
        _gathered_kv_write_mask=mask,
        _gathered_kv_slot_mappings=gathered_slots,
        _local_block_tables=(torch.tensor([[50], [50], [60]]), torch.tensor([[100], [100], [110]])),
    )
    connector = PCPHiSparseConnector(
        active, manager, NS(slot_mappings=original_slots, input_block_tables=original_tables), 0
    )
    return connector, cache, meta, local, global_batch, manager


@pytest.mark.parametrize("local_tokens", [0, 3])
@pytest.mark.parametrize("overlap", [False, True])
def test_connector_reuses_core_mirror_and_clear_after_local_queries(monkeypatch, local_tokens, overlap):
    import vllm.v1.worker.gpu.kv_connector as connector_module

    from vllm_ascend.hisparse.host_mirror import HostMirrorConnector

    connector, cache, _, local, batch, _ = make_connector()
    meta, *_ = gathered_metadata(local_tokens=local_tokens, prefill=True)
    del connector._start_load_kv
    worker = connector.kv_connector.connector_worker
    worker.is_host_writer = True
    worker._draft_layers = set()
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = set()
    worker._enqueue_row_dma = Mock()
    worker._invalidate_written_slots = Mock()

    def start_load(context, **kwargs):
        cache.runtime.request_state_indices = kwargs["request_state_indices"]
        cache.prepare_group_for_batch(kwargs["attn_metadata"]["layer0"])

    def finish():
        HiSparseConnectorWorker._enqueue_host_mirror(worker)
        HiSparseConnectorWorker._clear_forward_mirror_state(worker)

    connector.kv_connector.start_load_kv = Mock(side_effect=start_load)
    connector.kv_connector.finish_forward = Mock(side_effect=finish)
    monkeypatch.setattr(connector_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(connector_module, "get_forward_context", lambda: NS())
    scheduler = NS(kv_connector_metadata=object(), has_sync_kv_loads=True)
    connector.pre_forward(scheduler, local, {"layer0": meta})
    assert connector._pending_load_kwargs is None
    assert batch.idx_mapping.tolist() == [7, 9]
    assert cache.runtime.request_state_indices.tolist() == [7, 7, 9]
    # FULL replay skips Python attention.forward: Core's query classification
    # must still see local rows until the connector's Host mirror boundary.
    assert cache.num_actual_tokens == local_tokens
    assert cache.num_decode_tokens == 0
    if overlap:
        sources = NS(can_defer=Mock(return_value=False))
        wrapper = HostMirrorConnector(connector, worker, None, sources)
        wrapper._eligible_forward = True
        wrapper.finish_forward()
        sources.can_defer.assert_called_once_with(worker)
    else:
        connector.finish_forward()
    worker._enqueue_row_dma.assert_called_once_with((0,), ready_event=None)
    assert worker._invalidate_written_slots.call_args.args[2].tolist() == [0, 0, -1, 0, 1, -1]
    assert cache.mirror_slot_mapping.tolist() == [50, 53, -1, 51, 54, -1]
    assert cache.num_actual_tokens == 0 and cache.req_id_per_token is None
    assert not cache.host_mirror_required


def test_query_binding_preserves_group_counts_and_enables_early_mirror():
    leader, follower = handle(), handle()
    leader.index_group_caches = [leader, follower]
    meta, *_ = gathered_metadata(local_tokens=0, prefill=True)
    leader.prepare_group_for_batch(meta)
    prepare_pcp_query_states(leader, meta)
    for cache in leader.index_group_caches:
        assert cache.num_actual_tokens == 0 and cache.num_decode_tokens == 0
        assert cache.runtime.request_state_indices is meta.hisparse_request_state_indices
        assert cache.host_mirror_required
    # Exercise the original early-DMA gate, rather than substituting its body.
    ready = Mock()
    worker = NS(
        cache_handles=[leader],
        _slot_mapping_staging=None,
        _per_layer_mirrored=set(),
        _submitted_mirror_layers=set(),
        _layer_ready_events=[ready],
        _enqueue_row_dma=Mock(),
    )
    HiSparseConnectorWorker._enqueue_layer_mirror(worker, 0)
    worker._enqueue_row_dma.assert_called_once_with((0,), ready_event=ready)


def test_connector_stages_canonical_requests_and_gathered_slot_rows():
    connector, cache, meta, local, global_batch, manager = make_connector()
    scheduler = NS(kv_connector_metadata=object(), has_sync_kv_loads=True)
    other = NS()
    connector.pre_forward(scheduler, local, {"layer0": meta, "layer1": other, "layer2": meta})
    assert other.hisparse_write_req_id_per_token is meta.hisparse_write_req_id_per_token
    args = connector._pending_load_kwargs
    assert args["request_state_indices"] is global_batch.idx_mapping
    assert args["request_ids"] == ["A", "B"]
    assert args["num_tokens"] == 6
    assert global_batch.num_tokens == 7  # Load kwargs must leave the scheduler batch intact.
    assert meta.hisparse_request_state_indices.tolist() == local.idx_mapping.tolist()
    assert cache.slot_mapping.tolist() == [100, 103, -1, 101, 104, -1]
    assert cache.mirror_slot_mapping.tolist() == [50, 53, -1, 51, 54, -1]
    assert cache.source_block_table.tolist() == [[50], [50], [60]]
    assert cache.runtime.hot_block_table.tolist() == [[100], [100], [110]]
    assert cache.block_table.tolist() == [[901], [902]]
    connector._start_load_kv.assert_called_once()
    # Next step changes buffer, states and rows; no stale view is retained.
    manager._gathered_kv_slot_mappings = manager._gathered_kv_slot_mappings + 20
    manager._local_block_tables = tuple(table + 30 for table in manager._local_block_tables)
    connector.pre_forward(scheduler, local, {"layer0": meta})
    assert cache.slot_mapping[0].item() == 120
    assert cache.source_block_table[2].item() == 90
    assert cache.runtime.hot_block_table[2].item() == 140


@pytest.mark.parametrize("padded_tokens", [3, 8])
def test_replicated_decode_uses_first_rank_slots_without_gather(padded_tokens):
    connector, cache, meta, local, batch, manager = make_connector()
    batch.num_tokens = 3
    batch.query_start_loc = torch.tensor([0, 1, 3], dtype=torch.int32)
    batch.is_prefilling_np[:] = False
    local.num_tokens_after_padding = padded_tokens
    manager._gathered_kv_slot_mappings = torch.full((2, padded_tokens * 2), -1)
    manager._gathered_kv_slot_mappings[:, :3] = torch.tensor([[50, 51, 52], [100, 101, 102]])
    meta.num_input_tokens = padded_tokens
    meta.pcp_has_global_prefill = False
    prepare_pcp_query_metadata(
        meta, meta.hisparse_request_state_indices, meta.hisparse_num_valid_write_rows, local.idx_mapping, 1
    )
    scheduler = NS(kv_connector_metadata=object(), has_sync_kv_loads=True)
    connector.pre_forward(scheduler, local, {"layer0": meta})
    assert connector._pending_load_kwargs["num_tokens"] == padded_tokens
    assert meta.hisparse_write_req_id_per_token.tolist() == [0, 1, 1] + [-1] * (padded_tokens - 3)
    assert cache.slot_mapping.tolist() == [100, 101, 102] + [-1] * (padded_tokens - 3)
    assert cache.slot_mapping.data_ptr() == manager._gathered_kv_slot_mappings[1].data_ptr()
    assert meta.hisparse_num_valid_write_rows.tolist() == [padded_tokens]
    assert batch.num_tokens == 3


@pytest.mark.parametrize("capture", [False, True])
def test_sharded_decode_keeps_gathered_resident_and_host_rows(capture):
    connector, cache, meta, local, batch, manager = make_connector()
    manager.is_decode_sharded = True
    batch.is_prefilling_np[:] = False
    meta.pcp_has_global_prefill = False
    meta.hisparse_write_has_prefill = False
    meta.hisparse_for_capture = capture
    scheduler = NS(kv_connector_metadata=object(), has_sync_kv_loads=True)
    connector.pre_forward(scheduler, local, {"layer0": meta})
    assert connector._pending_load_kwargs["num_tokens"] == manager._padded_gather_idx.numel()
    assert cache.slot_mapping.data_ptr() == manager._gathered_kv_slot_mappings[1].data_ptr()
    assert cache.slot_mapping.tolist() == [100, 103, -1, 101, 104, -1]
    assert meta.hisparse_write_req_id_per_token.tolist() == [0, 0, -1, 0, 1, -1]
    assert connector._pending_load_kwargs["request_state_indices"].tolist() == [7, 9]
    assert meta.hisparse_request_state_indices.tolist() == [7, 7, 9]


def test_draft_decode_uses_global_slots_tables_and_persistent_canonical_states():
    connector, cache, _, _, _, manager = make_connector()
    cache.draft_layer = True
    target = handle()
    target_slots = target.slot_mapping
    connector._pcp_bindings.append((target, 1, 1))
    connector.prepare_draft_decode(torch.tensor([9, 7], dtype=torch.int64), 2)
    tables = connector._global_block_tables
    assert cache.slot_mapping.data_ptr() == tables.slot_mappings[1].data_ptr()
    assert cache.mirror_slot_mapping.data_ptr() == tables.slot_mappings[0].data_ptr()
    assert cache.source_block_table is tables.input_block_tables[0]
    assert cache.runtime.hot_block_table is tables.input_block_tables[1]
    states = cache.runtime.request_state_indices
    assert states.dtype == torch.int32
    assert states.tolist() == [9, 7]
    assert target.slot_mapping is target_slots
    connector.prepare_draft_decode(torch.tensor([7], dtype=torch.int64), 1)
    assert cache.runtime.request_state_indices is states
    assert states.tolist() == [7, -1]
    connector.prepare_draft_prefill()
    assert cache.slot_mapping.data_ptr() == manager._gathered_kv_slot_mappings[1].data_ptr()
    assert cache.mirror_slot_mapping.data_ptr() == manager._gathered_kv_slot_mappings[0].data_ptr()
    assert cache.source_block_table is manager._local_block_tables[0]
    assert cache.runtime.hot_block_table is manager._local_block_tables[1]
    assert target.slot_mapping is target_slots
    connector._start_load_kv.assert_not_called()


def test_capture_and_real_forward_keep_query_and_scatter_input_addresses():
    from vllm_ascend.hisparse.pcp import prepare_pcp_query_metadata

    connector, cache, meta, local, batch, manager = make_connector()
    assert cache.slot_mapping.data_ptr() == manager._gathered_kv_slot_mappings[1].data_ptr()
    assert cache.mirror_slot_mapping.data_ptr() == manager._gathered_kv_slot_mappings[0].data_ptr()
    meta.num_input_tokens = 3
    meta.pcp_has_global_prefill = False
    states = torch.empty(8, dtype=torch.int32)
    valid_rows = torch.empty(1, dtype=torch.int32)
    # Capture and real forwards are both prepared by the metadata builder.
    prepare_pcp_query_metadata(meta, states, valid_rows, torch.zeros(3, dtype=torch.int64), 2)
    cache._prepare_for_batch(meta)
    prepare_pcp_query_states(cache, meta)
    captured_states = cache.runtime.request_state_indices
    captured_slots = cache.slot_mapping
    prepare_pcp_query_metadata(meta, states, valid_rows, local.idx_mapping, 2)
    scheduler = NS(kv_connector_metadata=object(), has_sync_kv_loads=True)
    connector.pre_forward(scheduler, local, {"layer0": meta})
    assert captured_states is meta.hisparse_request_state_indices
    assert captured_states.tolist() == [7, 7, 9, -1, -1, -1, -1, -1]
    assert valid_rows is meta.hisparse_num_valid_write_rows
    assert valid_rows.tolist() == [6]
    assert captured_slots.data_ptr() == cache.slot_mapping.data_ptr()
    # A later step can reorder/reuse scheduler state slots without recapture.
    local.idx_mapping = torch.tensor([11, 11, 4], dtype=torch.int64)
    prepare_pcp_query_metadata(meta, states, valid_rows, local.idx_mapping, 2)
    connector.pre_forward(scheduler, local, {"layer0": meta})
    assert captured_states.tolist() == [11, 11, 4, -1, -1, -1, -1, -1]
    assert connector._pending_load_kwargs["request_state_indices"] is batch.idx_mapping


def test_connector_disabled_and_no_forward_do_not_require_pcp_layout():
    connector, _, _, _, _, _ = make_connector()
    connector._pcp_manager = None
    connector._disabled = True
    connector.pre_forward(NS())
    connector._disabled = False
    connector.pre_forward(NS(kv_connector_metadata=object(), has_sync_kv_loads=True))
    assert connector._pending_load_kwargs == {}


def test_connector_eager_reads_metadata_from_forward_context(monkeypatch):
    import vllm_ascend.hisparse.pcp as pcp_module

    connector, cache, meta, local, _, _ = make_connector()
    del meta.hisparse_write_req_id_per_token
    monkeypatch.setattr(pcp_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(pcp_module, "get_forward_context", lambda: NS(attn_metadata={"layer0": meta}))
    scheduler = NS(kv_connector_metadata=object(), has_sync_kv_loads=True)
    # Match GPUModelRunner's real eager call: no explicit metadata argument.
    connector.pre_forward(scheduler, local)
    assert meta.hisparse_request_state_indices.tolist() == local.idx_mapping.tolist()
    assert meta.hisparse_num_write_tokens == 6
    assert meta.hisparse_num_valid_write_rows.tolist() == [6]
    cache._prepare_for_batch(meta)
    prepare_pcp_query_states(cache, meta)
    assert cache.runtime.request_state_indices.tolist() == [7, 7, 9]
    assert hisparse_write_target(cache, meta, 6, 6)[2] == 6


def test_attention_binds_query_group_once_before_followers(monkeypatch):
    from vllm_ascend.attention import sfa_hisparse
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

    leader, follower = handle(), handle()
    leader.index_group_caches = follower.index_group_caches = [leader, follower]
    meta, *_ = gathered_metadata(local_tokens=0, prefill=True)

    def prepare_for_batch(index, metadata):
        if index == 0:
            leader.prepare_group_for_batch(metadata)

    group = NS(prepare_for_batch=Mock(side_effect=prepare_for_batch))
    bind = Mock(wraps=prepare_pcp_query_states)
    monkeypatch.setattr(sfa_hisparse, "prepare_pcp_query_states", bind)
    forward = Mock(return_value=object())
    monkeypatch.setattr(AscendSFAImpl, "forward", forward)
    for index, cache in enumerate([leader, follower]):
        impl = sfa_hisparse.AscendHiSparseImpl.__new__(sfa_hisparse.AscendHiSparseImpl)
        impl.enable_sparse_sfa_c8 = True
        impl.pcp_enabled = False
        impl.qk_rope_head_dim = 0
        impl._c8_length_workspace = Mock()
        impl.index_group = group
        impl.index_group_index = index
        impl.hisparse_cache = cache
        assert impl.forward(f"layer{index}", None, None, meta) is forward.return_value
        assert cache.runtime.request_state_indices is meta.hisparse_request_state_indices
        assert cache.host_mirror_required

    bind.assert_called_once_with(leader, meta)
    assert group.prepare_for_batch.call_count == 2
    assert forward.call_count == 2


def test_c8_scatter_covers_gathered_tail(monkeypatch):
    import vllm_ascend.hisparse.c8 as c8
    import vllm_ascend.hisparse.cache_ops as cache_ops
    from vllm_ascend.attention.sfa_hisparse import AscendHiSparseImpl
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

    impl = AscendHiSparseImpl.__new__(AscendHiSparseImpl)
    impl.kv_a_layernorm = object()
    impl.enable_sparse_sfa_c8 = True
    impl.kv_lora_rank = 2
    impl.hisparse_cache = handle()
    impl.hisparse_cache.finish_kv_update = Mock()
    meta, *_ = gathered_metadata()
    impl.hisparse_cache._prepare_for_batch(meta)
    packed = torch.arange(48, dtype=torch.uint8).view(6, 8)
    monkeypatch.setattr(
        AscendSFAImpl, "_exec_kv_local", lambda *a: (torch.zeros(6, 1), torch.zeros(6, 2), torch.ones(6, 1))
    )
    monkeypatch.setattr(c8, "pack_c8_components", lambda *a: packed)

    def scatter(latent, rope, destination, mapping, num_valid_rows):
        # Simulate only the device boundary: the production caller must pass
        # all gathered rows; a local-query length here loses rows 3..5.
        for row in range(int(num_valid_rows[0])):
            slot = int(mapping[row])
            if slot >= 0:
                destination[slot - 10] = latent[row]

    monkeypatch.setattr(cache_ops, "scatter_packed_kv", scatter)
    impl._exec_kv_local(None, None, None, None, impl.hisparse_cache.slot_mapping, meta)
    torch.testing.assert_close(impl.hisparse_cache.view.cache[3], packed[4])
    impl.hisparse_cache.finish_kv_update.assert_not_called()


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("pcp_world_size", [2, 4, 8])
@pytest.mark.parametrize("prefill", [False, True])
def test_pcp_prolog_gathers_local_scratch_before_resident_write(monkeypatch, capture, pcp_world_size, prefill):
    import vllm_ascend.attention.sfa_hisparse as module
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
    from vllm_ascend.hisparse import cache_ops

    impl = module.AscendHiSparseImpl.__new__(module.AscendHiSparseImpl)
    impl.pcp_enabled = True
    impl.is_pcp_decode_sharded = not prefill
    impl.hisparse_cache = handle()
    impl.hisparse_cache.finish_kv_update = Mock()
    meta, *_ = gathered_metadata(prefill=prefill)
    meta.uses_pcp_sharded_layout = True
    meta.num_prefills = int(prefill)
    meta.num_input_tokens = 3
    meta.hisparse_for_capture = capture
    num_write_rows = meta.num_input_tokens * pcp_world_size
    meta.hisparse_num_write_tokens = num_write_rows
    meta.hisparse_num_valid_write_rows.fill_(num_write_rows)
    meta.hisparse_write_req_id_per_token = torch.zeros(num_write_rows, dtype=torch.int32)
    impl.hisparse_cache._prepare_for_batch(meta)
    impl.hisparse_cache.submit_layer_mirror = Mock()
    from vllm.config.compilation import CUDAGraphMode

    monkeypatch.setattr(module, "get_forward_context", lambda: NS(cudagraph_runtime_mode=CUDAGraphMode.NONE))
    impl.hisparse_cache.slot_mapping = torch.arange(num_write_rows) + 10
    impl.hisparse_cache.slot_mapping[2::3] = -1
    impl.hisparse_cache.view.cache = torch.zeros(num_write_rows, 656, dtype=torch.uint8)
    scratch = torch.zeros((1, 128, 1, 656), dtype=torch.uint8).view(torch.float8_e4m3fn)
    slots = torch.arange(128, dtype=torch.int64)
    impl._pcp_prolog_workspace = lambda: (scratch, slots)

    def fused(self, hidden, caches, cos, sin, mapping, **kwargs):
        assert caches[0] is scratch
        assert mapping.tolist() == [0, 1, 2]
        assert torch.count_nonzero(impl.hisparse_cache.view.cache) == 0
        scratch.view(torch.uint8).view(-1, 656)[:3] = torch.arange(1, 4)[:, None]
        return "queries-ready"

    def gather(payload, dim):
        assert payload.shape == (3, 656) and payload.dtype == torch.int8 and dim == 0
        return torch.cat([payload + 10 * rank for rank in range(pcp_world_size)], dim=0)

    def scatter(payload, rope, destination, mapping, valid):
        assert payload[:, 0].tolist() == [10 * rank + row for rank in range(pcp_world_size) for row in (1, 2, 3)]
        assert valid.tolist() == [num_write_rows]
        for row, slot in enumerate(mapping.tolist()):
            if slot >= 0:
                destination[slot - 10].fill_(int(payload[row, 0]))

    monkeypatch.setattr(AscendSFAImpl, "_sfa_preprocess_prolog_v3", fused)
    monkeypatch.setattr(module, "get_pcp_group", lambda: NS(all_gather=gather))
    monkeypatch.setattr(cache_ops, "scatter_packed_kv", scatter)
    assert (
        impl._sfa_preprocess_prolog_v3(torch.zeros(3, 8), (), None, None, slots[:3], attn_metadata=meta)
        == "queries-ready"
    )
    impl.hisparse_cache.finish_kv_update.assert_not_called()
    impl._sync_pcp_kv_cache((), meta)
    expected = [10 * rank + row if row else 0 for rank in range(pcp_world_size) for row in (1, 2, 0)]
    assert impl.hisparse_cache.view.cache[:, 0].tolist() == expected
    impl.hisparse_cache.finish_kv_update.assert_not_called()
    assert impl.hisparse_cache.submit_layer_mirror.call_count == int(prefill and not capture)


def test_replicated_decode_prolog_writes_resident_directly(monkeypatch):
    import vllm_ascend.attention.sfa_hisparse as module
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

    impl = module.AscendHiSparseImpl.__new__(module.AscendHiSparseImpl)
    impl.pcp_enabled = True
    impl.is_pcp_decode_sharded = False
    impl.hisparse_cache = handle()
    impl._pcp_prolog_workspace = Mock(side_effect=AssertionError("replicated decode must not use scratch"))
    meta = metadata()
    meta.uses_pcp_sharded_layout = True
    meta.num_prefills = 0
    meta.pcp_has_global_prefill = False
    impl.hisparse_cache._prepare_for_batch(meta)
    mapping = torch.tensor([10, 11, -1], dtype=torch.int64)
    monkeypatch.setattr(module.AscendHiSparseImpl, "_prolog_slot_mapping", lambda *a: mapping)
    monkeypatch.setattr(module, "finish_hisparse_kv_update", Mock())
    prolog = Mock(return_value="queries-ready")
    monkeypatch.setattr(AscendSFAImpl, "_sfa_preprocess_prolog_v3", prolog)
    assert (
        impl._sfa_preprocess_prolog_v3(torch.zeros(3, 8), (), None, None, mapping, attn_metadata=meta)
        == "queries-ready"
    )
    assert prolog.call_args.args[1][0].data_ptr() == impl.hisparse_cache.view.cache.data_ptr()
    assert prolog.call_args.args[4] is mapping
    impl._pcp_prolog_workspace.assert_not_called()


@pytest.mark.parametrize(
    "lengths,prefills",
    [
        ([1, 1], [False, False]),
        ([1], [False]),
        ([9, 5], [True, True]),
        ([2, 3], [True, True]),
        ([0, 1, 3], [False, False, True]),
        ([64, 32], [True, True]),
        ([1, 1, 9], [False, False, True]),
    ],
)
@pytest.mark.parametrize("pcp_world_size,pcp_rank", [(2, 0), (2, 1), (4, 3), (8, 7)])
@pytest.mark.parametrize("shard_decode", [False, True])
def test_real_pcp_layout_maps_each_global_write_once(
    monkeypatch, lengths, prefills, pcp_world_size, pcp_rank, shard_decode
):
    import vllm_ascend.worker.v2.pcp_manager as pcp_module

    monkeypatch.setattr(pcp_module, "async_copy_to_gpu", lambda values, device: torch.as_tensor(values).clone())
    manager = pcp_module.AscendPCPManager(pcp_world_size, pcp_rank, torch.device("cpu"), shard_decode)
    manager.bind_config(
        NS(
            attention_config=NS(hisparse_config={}),
            model_config=None,
            parallel_config=NS(prefill_context_parallel_size=pcp_world_size, decode_context_parallel_size=1),
        )
    )
    lens = np.array(lengths, dtype=np.int32)
    flags = np.array(prefills, dtype=np.bool_)
    starts = np.concatenate(([0], np.cumsum(lens))).astype(np.int32)
    segments, counts = manager._build_batch_layout(lens, np.zeros_like(lens), flags, starts)
    batch = Batch(
        int(sum(lens)), torch.arange(len(lens), dtype=torch.int32), list(range(len(lens))), torch.tensor(starts), flags
    )
    meta = metadata(counts[pcp_rank], decode=not any(prefills))
    local = NS(idx_mapping=torch.tensor([s.global_batch_req_idx for s in segments[pcp_rank]], dtype=torch.int32))
    meta.hisparse_write_req_id_per_token = get_pcp_write_request_ids(
        batch, manager._padded_gather_idx, manager._gathered_kv_write_mask
    )
    mask = manager._gathered_kv_write_mask
    indices = manager._padded_gather_idx
    assert sorted(indices[mask].tolist()) == list(range(int(sum(lens))))
    actual_ids = meta.hisparse_write_req_id_per_token[mask]
    expected_ids = np.repeat(np.arange(len(lens)), lens)[indices[mask].numpy()]
    assert actual_ids.tolist() == expected_ids.tolist()
    assert torch.all(meta.hisparse_write_req_id_per_token[~mask] == -1)
    if lengths == [64, 32] and pcp_world_size > 1:
        assert len(local.idx_mapping) > len(batch.idx_mapping)  # genuine DualChunk duplication.
    global_slots = torch.arange(int(sum(lens)), dtype=torch.int64)[None, :] + 100
    gathered = manager._convert_to_gathered_slot_mappings(global_slots)
    assert gathered[0, mask].tolist() == (indices[mask] + 100).tolist()
    assert torch.all(gathered[0, ~mask] == -1)


@pytest.mark.parametrize("defer_wait", [False, True])
@pytest.mark.parametrize("pcp_world_size", [2, 4, 8])
def test_native_mixed_kv_retains_resident_and_host_row_order(monkeypatch, defer_wait, pcp_world_size):
    import vllm_ascend.attention.sfa_v1 as sfa
    from vllm_ascend.attention.sfa_hisparse import AscendHiSparseImpl

    impl = AscendHiSparseImpl.__new__(AscendHiSparseImpl)
    impl.pcp_enabled = True
    impl.is_pcp_decode_sharded = False
    impl.qk_rope_head_dim = 0
    meta = NS(
        uses_pcp_sharded_layout=True,
        num_decode_tokens=1,
        num_prefills=1,
        pcp_has_global_prefill=True,
        num_input_tokens=3,
    )
    local = torch.tensor([[10.0], [20.0], [21.0]])
    # One replicated decode row, then two rank-local prefill rows per rank.
    gathered = torch.cat([local + rank * 100 for rank in range(pcp_world_size)])
    slots = torch.arange(3 * pcp_world_size, dtype=torch.int64) + 10
    slots[3::3] = -1
    captured = {}

    def gather(send, group):
        assert torch.equal(send, local)
        return gathered, None

    def commit(values, cos, sin, kv_cache, mapping, metadata):
        captured["values"] = values
        captured["slots"] = mapping
        return values

    monkeypatch.setattr(sfa, "get_pcp_group", lambda: NS(world_size=pcp_world_size))
    monkeypatch.setattr(sfa, "all_gather_async", gather)
    impl._exec_kv_local = commit
    result = impl.exec_kv(local, None, None, (), slots, meta, defer_wait=defer_wait)
    if defer_wait:
        assert not captured
        result = result()
    assert torch.equal(result, gathered)
    assert torch.equal(captured["slots"], slots)
    # Resident scatter and Host mirror consume this same masked rank-major order.
    assert captured["values"].shape[0] == captured["slots"].numel()
    assert captured["values"][3, 0] == 110 and captured["slots"][3] == -1
    assert captured["values"][4, 0] == 120 and captured["slots"][4] == 14
