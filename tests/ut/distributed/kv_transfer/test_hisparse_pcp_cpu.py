# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts at the PCP / HiSparse boundary; no device kernels are run."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytest.importorskip("vllm.v1.hisparse.backend")
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker
from vllm.v1.hisparse.runtime import HiSparseCacheHandle
from vllm.v1.kv_cache_interface import HiSparseHotSpec, HiSparseResidentSpec
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

from vllm_ascend.hisparse.pcp import (
    PCPHiSparseConnector,
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


def make_connector(padded_tokens=3):
    meta = metadata()
    local = NS(num_tokens=3, idx_mapping=torch.tensor([7, 7, 9], dtype=torch.int64))
    global_batch = NS(
        num_tokens=7,
        idx_mapping=torch.tensor([7, 9], dtype=torch.int32),
        req_ids=["A", "B"],
        is_prefilling_np=np.array([False, False]),
    )
    original_slots = torch.tensor([[50, 51, 52], [100, 101, 102], [-1, -1, -1]])
    cache = handle()
    original_tables = tuple(torch.zeros(2, 1, dtype=torch.int64) for _ in range(3))
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
    manager = NS(
        global_batch=global_batch,
        _gathered_kv_slot_mappings=gathered_slots,
        _local_block_tables=original_tables,
    )
    connector = PCPHiSparseConnector(
        active, manager, NS(slot_mappings=original_slots, input_block_tables=original_tables), 0
    )
    return connector, cache, meta, local, global_batch, manager


def test_host_mirror_mapping_shared_across_groups_and_scheduling_steps(monkeypatch):
    connector, cache, meta, local, batch, manager = make_connector()
    follower, draft = handle(), handle()
    follower.runtime.is_group_leader = False
    draft.draft_layer = True
    cache.index_group_caches.append(follower)
    follower.mirror_slot_mapping = cache.mirror_slot_mapping
    worker = connector.kv_connector.connector_worker
    worker.cache_handles.extend([follower, draft])
    worker._group_leaders.append(("draft", draft))
    connector._pcp_bindings.append((draft, 1, 2))
    del connector._start_load_kv
    monkeypatch.setattr(ActiveKVConnector, "_start_load_kv", Mock())
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = set()
    batch.is_prefilling_np[:] = True
    manager._gathered_kv_write_mask = torch.tensor([True, True, False, True, True, False])
    scheduler = NS(kv_connector_metadata=object(), has_sync_kv_loads=True)
    draft_meta = metadata(decode=False)
    for md in (meta, draft_meta):
        md.hisparse_num_write_tokens = 6
        md.hisparse_write_has_prefill = True
    # Remove/reorder requests and reuse state slots, including the draft transition.
    for requests, states, starts, indices, expected in [
        (["A", "B"], [7, 9], [0, 4, 7], [0, 3, 0, 1, 4, 0], [0, 0, -1, 0, 1, -1]),
        (["B", "C"], [9, 7], [0, 3, 7], [4, 0, 0, 3, 1, 0], [1, 0, -1, 1, 0, -1]),
    ]:
        batch.req_ids = requests
        batch.idx_mapping.copy_(torch.tensor(states))
        batch.query_start_loc = torch.tensor(starts, dtype=torch.int32)
        manager._padded_gather_idx = torch.tensor(indices)
        connector.prepare_draft_decode()
        connector._build_load_kwargs(scheduler, local, {"layer0": meta, "draft": draft_meta})
        connector._start_load_kv()
        connector.prepare_host_mirror()
        assert draft.source_block_table is manager._local_block_tables[0]
        assert draft.runtime.hot_block_table is manager._local_block_tables[2]
        for member in worker.cache_handles:
            assert member.mirror_slot_mapping.data_ptr() == cache.mirror_slot_mapping.data_ptr()
            assert member.req_id_per_token is meta.hisparse_write_req_id_per_token
            assert member.req_id_per_token.tolist() == expected
            assert member.num_actual_tokens == 6 and member.host_mirror_required
        HiSparseConnectorWorker._clear_forward_mirror_state(worker)
        assert all(member.req_id_per_token is None for member in worker.cache_handles)


def test_capture_and_real_forward_keep_query_and_scatter_input_addresses():
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
    prepare_pcp_query_metadata(meta, states, valid_rows, torch.zeros(2, dtype=torch.int64), 1)
    cache._prepare_for_batch(meta)
    prepare_pcp_query_states(cache, meta)
    captured_states = cache.runtime.request_state_indices
    captured_slots = cache.slot_mapping
    prepare_pcp_query_metadata(meta, states, valid_rows, local.idx_mapping, 1)
    scheduler = NS(kv_connector_metadata=object(), has_sync_kv_loads=True)
    connector.pre_forward(scheduler, local, {"layer0": meta, "indexer": NS()})
    assert captured_states is meta.hisparse_request_state_indices
    assert captured_states.tolist() == [7, 9, -1, -1, -1, -1, -1, -1]
    assert valid_rows is meta.hisparse_num_valid_write_rows
    assert valid_rows.tolist() == [8]
    assert meta.hisparse_write_req_id_per_token is meta.req_id_per_token
    assert cache.slot_mapping[:8].tolist() == [100, 101, 102, -1, -1, -1, -1, -1]
    assert connector._pending_load_kwargs["num_tokens"] == 8
    assert captured_slots.data_ptr() == cache.slot_mapping.data_ptr()


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
    meta = metadata(decode=False)
    meta.num_input_tokens = 3
    meta.pcp_has_global_prefill = True
    prepare_pcp_query_metadata(
        meta,
        torch.empty(2, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        impl.hisparse_cache.runtime.request_state_indices,
        pcp_world_size,
    )
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
