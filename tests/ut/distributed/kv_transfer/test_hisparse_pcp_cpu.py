# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts at the PCP / HiSparse boundary; no device kernels are run."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker
from vllm.v1.hisparse.runtime import HiSparseCacheHandle
from vllm.v1.kv_cache_interface import HiSparseHotSpec, HiSparseResidentSpec
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

from vllm_ascend.hisparse.pcp import (
    PCPHiSparseConnector,
    get_pcp_write_request_ids,
    prepare_pcp_query_metadata,
    prepare_pcp_query_states,
)


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
        resident_source_index=0,
        max_num_reqs=2,
        max_swap_rows=16,
        request_state_indices=torch.tensor([7, 9], dtype=torch.int32),
        is_group_leader=True,
    )
    cache = HiSparseCacheHandle(runtime)
    cache.view = NS(cache=torch.zeros(12, 8, dtype=torch.uint8))
    cache.slot_mapping = torch.tensor([10, 11, -1, 12, 13, -1])
    return cache


def gathered_metadata(local_tokens=3):
    # A has two local chunks but one canonical request state.
    global_batch = NS(
        num_tokens=7,
        idx_mapping=torch.tensor([7, 9], dtype=torch.int32),
        req_ids=["A", "B"],
        query_start_loc=torch.tensor([0, 4, 7], dtype=torch.int32),
        is_prefilling_np=np.array([True, False]),
    )
    local = NS(
        num_tokens=local_tokens,
        idx_mapping=torch.tensor([7, 7, 9], dtype=torch.int64),
    )
    indices = torch.tensor([0, 3, 0, 1, 4, 0], dtype=torch.int64)
    mask = torch.tensor([True, True, False, True, True, False])
    meta = metadata(local_tokens, decode=False)
    meta.num_input_tokens = indices.numel() // 2
    meta.pcp_has_global_prefill = True
    prepare_pcp_query_metadata(
        meta,
        torch.empty(local.idx_mapping.shape, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        local.idx_mapping,
        2,
    )
    meta.hisparse_write_req_id_per_token = get_pcp_write_request_ids(global_batch, indices, mask)
    return meta, local, global_batch, indices, mask


def make_connector(padded_tokens=3):
    meta, local, global_batch, indices, mask = gathered_metadata()
    original_slots = torch.tensor([[50, 51, 52], [100, 101, 102], [-1, -1, -1]])
    cache = handle()
    cache.slot_mapping = original_slots[1]
    original_tables = (torch.tensor([[50], [60]]), torch.tensor([[901], [902]]), torch.tensor([[100], [110]]))
    cache.runtime.hot_block_table = original_tables[2]
    cache.block_table = torch.tensor([[901], [902]])
    underlying = NS(
        connector_worker=NS(
            cache_handles=[cache],
            _group_leaders=[("layer0", cache)],
            request_state_indices=cache.runtime.request_state_indices,
            kv_cache_config=NS(
                kv_cache_groups=[
                    NS(kv_cache_spec=None, layer_names=[]),
                    NS(kv_cache_spec=HiSparseResidentSpec.__new__(HiSparseResidentSpec), layer_names=["layer0"]),
                    NS(kv_cache_spec=HiSparseHotSpec.__new__(HiSparseHotSpec), layer_names=["layer0"]),
                ]
            ),
        ),
        handle_preemptions=Mock(),
        bind_connector_metadata=Mock(),
    )
    active = ActiveKVConnector.__new__(ActiveKVConnector)
    active.kv_connector = underlying
    active._disabled = False
    active._pending_load_kwargs = None
    active._start_load_kv = Mock()
    gathered_slots = torch.full((3, padded_tokens * 2), -1)
    gathered_slots[:2, :6] = torch.tensor([[50, 53, -1, 51, 54, -1], [100, 103, -1, 101, 104, -1]])
    manager = NS(
        global_batch=global_batch,
        _padded_gather_idx=indices,
        _gathered_kv_write_mask=mask,
        _gathered_kv_slot_mappings=gathered_slots,
        _local_block_tables=(torch.tensor([[50], [50], [60]]), original_tables[1], torch.tensor([[100], [100], [110]])),
    )
    connector = PCPHiSparseConnector(
        active, manager, NS(slot_mappings=original_slots, input_block_tables=original_tables), 0
    )
    return connector, cache, meta, local, global_batch, manager


@pytest.mark.parametrize("local_tokens,mirror_path", [(0, "direct"), (3, "fallback"), (0, "deferred")])
def test_connector_reuses_core_mirror_and_clear_after_local_queries(monkeypatch, local_tokens, mirror_path):
    import vllm.v1.worker.gpu.kv_connector as connector_module

    import vllm_ascend.hisparse.pcp as pcp
    from vllm_ascend.hisparse.host_mirror import HostMirrorConnector

    connector, cache, _, local, batch, _ = make_connector()
    meta, *_ = gathered_metadata(local_tokens=local_tokens)
    del connector._start_load_kv
    worker = connector.kv_connector.connector_worker
    worker.is_host_writer = True
    worker._draft_layers = set()
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = set()
    worker._enqueue_row_dma = Mock()
    worker._invalidate_written_slots = Mock()

    def start_load(context, **kwargs):
        assert kwargs["request_ids"] == ["A", "B"]
        assert kwargs["num_tokens"] == 6
        cache.runtime.request_state_indices = kwargs["request_state_indices"]
        cache.prepare_group_for_batch(kwargs["attn_metadata"]["layer0"])

    def finish(ready_event=None):
        HiSparseConnectorWorker._enqueue_host_mirror(worker, ready_event)
        HiSparseConnectorWorker._clear_forward_mirror_state(worker)

    connector.kv_connector.start_load_kv = Mock(side_effect=start_load)
    connector.kv_connector.finish_forward = Mock(side_effect=finish)
    monkeypatch.setattr(connector_module, "is_forward_context_available", lambda: True)
    context = NS(attn_metadata={"layer0": meta})
    monkeypatch.setattr(connector_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(pcp, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(pcp, "get_forward_context", lambda: context)
    scheduler = NS(kv_connector_metadata=object(), has_sync_kv_loads=True)
    connector.pre_forward(scheduler, local, None if mirror_path == "fallback" else context.attn_metadata)
    assert connector._pending_load_kwargs is None
    assert batch.idx_mapping.tolist() == [7, 9]
    assert cache.runtime.request_state_indices.tolist() == [7, 7, 9]
    # Local queries remain bound until the Host publication boundary.
    assert cache.num_actual_tokens == local_tokens
    assert cache.num_decode_tokens == 0
    connector.prepare_host_mirror = Mock(wraps=connector.prepare_host_mirror)
    if mirror_path != "direct":
        worker._forward_ready_event = worker.host_write_event = Mock()
        worker._pending_dma_descriptors = []
        worker._dma_submitted = False
        worker._finish_mirror_phase = finish
        worker._release_completed_dma_descriptors = Mock()
        monkeypatch.setattr(torch.accelerator, "current_stream", Mock())
        sources = NS(can_defer=Mock(return_value=mirror_path == "deferred"))
        wrapper = HostMirrorConnector(connector, worker, None, sources)
        wrapper._eligible_forward = True
        wrapper.finish_forward()
        sources.can_defer.assert_called_once_with(worker)
    else:
        connector.finish_forward()
    connector.prepare_host_mirror.assert_called_once_with()
    ready_event = worker._forward_ready_event if mirror_path == "deferred" else None
    worker._enqueue_row_dma.assert_called_once_with((0,), ready_event=ready_event)
    assert worker._invalidate_written_slots.call_args.args[2].tolist() == [0, 0, -1, 0, 1, -1]
    assert cache.mirror_slot_mapping.tolist() == [50, 53, -1, 51, 54, -1]
    assert cache.num_actual_tokens == 0 and cache.req_id_per_token is None
    assert not cache.host_mirror_required


def test_draft_switches_between_global_decode_and_local_prefill_layouts():
    connector, cache, _, _, _, manager = make_connector()
    cache.draft_layer = True
    connector.prepare_draft_decode()
    tables = connector._global_block_tables
    assert cache.slot_mapping.data_ptr() == tables.slot_mappings[1].data_ptr()
    assert cache.mirror_slot_mapping.data_ptr() == tables.slot_mappings[0].data_ptr()
    assert cache.runtime.request_state_indices is connector.kv_connector.connector_worker.request_state_indices
    connector.prepare_draft_prefill()
    assert cache.slot_mapping.data_ptr() == manager._gathered_kv_slot_mappings[1].data_ptr()
    assert cache.source_block_table is manager._local_block_tables[0]
    assert cache.runtime.hot_block_table is manager._local_block_tables[2]


def test_capture_and_real_forward_keep_query_and_scatter_input_addresses():
    from vllm_ascend.hisparse.pcp import prepare_pcp_query_metadata

    connector, cache, meta, local, batch, manager = make_connector(padded_tokens=8)
    manager._gathered_kv_slot_mappings.fill_(-1)
    manager._gathered_kv_slot_mappings[1, :3] = torch.tensor([100, 101, 102])
    assert cache.slot_mapping.data_ptr() == manager._gathered_kv_slot_mappings[1].data_ptr()
    assert cache.mirror_slot_mapping.data_ptr() == manager._gathered_kv_slot_mappings[0].data_ptr()
    meta.num_input_tokens = 8
    # The decode graph has replicated queries, not gathered prefill rows.
    meta.pcp_has_global_prefill = False
    batch.is_prefilling_np[:] = False
    local.idx_mapping = batch.idx_mapping
    local.num_tokens_after_padding = 8
    meta.req_id_per_token = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0], dtype=torch.int32)
    states = torch.empty(8, dtype=torch.int32)
    valid_rows = torch.empty(1, dtype=torch.int32)
    # Capture and real forwards are both prepared by the metadata builder.
    prepare_pcp_query_metadata(meta, states, valid_rows, torch.zeros(2, dtype=torch.int64), 1)
    cache._prepare_for_batch(meta)
    prepare_pcp_query_states(cache, meta)
    captured_states = cache.runtime.request_state_indices
    captured_slots = cache.slot_mapping
    prepare_pcp_query_metadata(meta, states, valid_rows, local.idx_mapping, 1)
    scheduler = NS(kv_connector_metadata=object(), has_sync_kv_loads=True)
    connector.pre_forward(scheduler, local, {"layer0": meta})
    assert captured_states is meta.hisparse_request_state_indices
    assert captured_states.tolist() == [7, 9, -1, -1, -1, -1, -1, -1]
    assert valid_rows is meta.hisparse_num_valid_write_rows
    assert valid_rows.tolist() == [8]
    assert meta.hisparse_write_req_id_per_token is meta.req_id_per_token
    assert cache.slot_mapping[:8].tolist() == [100, 101, 102, -1, -1, -1, -1, -1]
    assert connector._pending_load_kwargs["num_tokens"] == 8
    assert captured_slots.data_ptr() == cache.slot_mapping.data_ptr()
    # A later step can reorder/reuse scheduler state slots without recapture.
    local.idx_mapping = torch.tensor([11, 4], dtype=torch.int64)
    prepare_pcp_query_metadata(meta, states, valid_rows, local.idx_mapping, 1)
    connector.pre_forward(scheduler, local, {"layer0": meta})
    assert captured_states.tolist() == [11, 4, -1, -1, -1, -1, -1, -1]
    assert connector._pending_load_kwargs["request_state_indices"] is batch.idx_mapping


@pytest.mark.parametrize("pcp_world_size,capture", [(2, False), (4, True), (8, False)])
def test_pcp_prolog_gathers_local_scratch_before_resident_write(monkeypatch, capture, pcp_world_size):
    import vllm_ascend.attention.sfa_hisparse as module
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
    from vllm_ascend.hisparse import cache_ops

    impl = module.AscendHiSparseImpl.__new__(module.AscendHiSparseImpl)
    impl.pcp_enabled = True
    impl.is_pcp_decode_sharded = False
    impl.hisparse_cache = handle()
    impl.hisparse_cache.finish_kv_update = Mock()
    meta, *_ = gathered_metadata()
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
    monkeypatch.setattr(module, "needs_pcp_kv_gather", lambda *a: True)

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
    impl._sync_pcp_kv_cache((), meta)
    expected = [10 * rank + row if row else 0 for rank in range(pcp_world_size) for row in (1, 2, 0)]
    assert impl.hisparse_cache.view.cache[:, 0].tolist() == expected
    impl.hisparse_cache.finish_kv_update.assert_not_called()
    assert impl.hisparse_cache.submit_layer_mirror.call_count == (0 if capture else 1)


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


def test_native_mixed_kv_retains_resident_and_host_row_order(monkeypatch):
    pcp_world_size = 4
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
    gather = Mock(return_value=(gathered, None))
    monkeypatch.setattr(sfa, "get_pcp_group", lambda: NS(world_size=pcp_world_size))
    monkeypatch.setattr(sfa, "all_gather_async", gather)
    impl._exec_kv_local = Mock(return_value=gathered)
    pending = impl.exec_kv(local, None, None, (), slots, meta, defer_wait=True)
    impl._exec_kv_local.assert_not_called()
    result = pending()
    assert result is gathered
    torch.testing.assert_close(gather.call_args.args[0], local)
    torch.testing.assert_close(impl._exec_kv_local.call_args.args[0], gathered)
    torch.testing.assert_close(impl._exec_kv_local.call_args.args[4], slots)
