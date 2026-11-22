# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts at the PCP / HiSparse boundary; no device kernels are run."""

from types import SimpleNamespace as NS
from typing import Any
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


@pytest.mark.parametrize(
    "world_size,dtype,rope_dim,decode",
    [
        (2, torch.bfloat16, 0, False),
        (2, torch.bfloat16, 64, True),
        (4, torch.float16, 64, False),
    ],
)
def test_unquantized_pcp_writes_all_gathered_rows_without_query_truncation(
    monkeypatch, world_size, dtype, rope_dim, decode
):
    from vllm_ascend.attention import sfa_hisparse
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
    from vllm_ascend.hisparse import cache_ops

    impl = sfa_hisparse.AscendHiSparseImpl.__new__(sfa_hisparse.AscendHiSparseImpl)
    impl.pcp_enabled, impl.is_pcp_decode_sharded, impl.enable_sparse_sfa_c8 = True, decode, False
    impl.kv_lora_rank, impl.qk_rope_head_dim = 4, rope_dim
    impl._kv_staging_capacity = 128
    impl.vllm_config = NS(parallel_config=NS(prefill_context_parallel_size=world_size))
    impl.index_group = NS(logical_topk_indices=torch.empty(1))
    impl.kv_a_layernorm = torch.nn.Identity()
    impl.hisparse_cache = handle()
    monkeypatch.setattr(
        sfa_hisparse,
        "current_workspace_manager",
        lambda: NS(get_persistent_resource=lambda key, allocate: allocate()),
    )
    cache = impl.hisparse_cache
    width, rows = 4 + rope_dim, 3 * world_size
    cache.view.cache = torch.zeros(rows, width, dtype=dtype)
    cache.slot_mapping = torch.arange(rows, dtype=torch.int64)
    cache.slot_mapping[2::3] = -1
    meta = metadata(decode=decode)
    meta.num_input_tokens = 3
    meta.hisparse_num_write_tokens = rows
    meta.hisparse_num_valid_write_rows = None
    cache._prepare_for_batch(meta)
    assert cache.num_actual_tokens == 3
    # NoPE has two projected rows and one padded row in each rank stride.
    local_rows = 3 if rope_dim else 2
    local_kv = torch.arange(local_rows * width, dtype=dtype).reshape(local_rows, width)
    cos = sin = torch.ones(3, rope_dim, dtype=dtype) if rope_dim else None

    def preprocess(self, kv, cos, sin, caches, slots, metadata):
        assert caches[0].shape == (world_size, 128, 1, 4)
        assert caches[1].shape == (world_size, 128, 1, rope_dim)
        assert slots.numel() == rows
        caches[0].view(-1, 4)[:rows].copy_(kv[:, :4])
        caches[1].view(-1, rope_dim)[:rows].copy_(kv[:, 4:])

    def scatter(latent, rope, destination, mapping, valid_rows):
        assert latent.shape[0] == mapping.numel() == rows
        assert valid_rows is None
        valid = mapping >= 0
        destination[mapping[valid]] = torch.cat((latent, rope), dim=-1)[valid]

    monkeypatch.setattr(AscendSFAImpl, "_exec_kv_local", preprocess)
    monkeypatch.setattr(sfa_hisparse, "get_pcp_group", lambda: NS(world_size=world_size))
    monkeypatch.setattr(sfa_hisparse, "needs_pcp_kv_gather", lambda *args: True)
    monkeypatch.setattr(sfa_hisparse, "all_gather_async", lambda value, group: (torch.cat([value] * world_size), None))
    monkeypatch.setattr(cache_ops, "scatter_packed_kv", scatter)
    monkeypatch.setattr(sfa_hisparse, "finish_hisparse_kv_update", Mock())
    finish = impl.exec_kv(local_kv, cos, sin, (), cache.slot_mapping, meta, defer_wait=True)
    assert not cache.view.cache.any()
    finish()
    for rank in range(world_size):
        torch.testing.assert_close(cache.view.cache[3 * rank : 3 * rank + 2], local_kv[:2])
        assert not cache.view.cache[3 * rank + 2].any()


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
        is_decode_sharded=False,
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
    manager._gathered_kv_write_mask = torch.tensor([True, True, False, True, True, False])
    scheduler = NS(kv_connector_metadata=object(), has_sync_kv_loads=True)
    draft_meta = metadata(decode=False)
    for md in (meta, draft_meta):
        md.hisparse_num_write_tokens = 6
        md.hisparse_write_has_prefill = True
    # Step 1 uses global prefill; step 2 uses sharded decode with no prefill.
    for is_prefill, requests, states, starts, indices, expected in [
        (True, ["A", "B"], [7, 9], [0, 4, 7], [0, 3, 0, 1, 4, 0], [0, 0, -1, 0, 1, -1]),
        (False, ["B", "C"], [9, 7], [0, 3, 7], [4, 0, 0, 3, 1, 0], [1, 0, -1, 1, 0, -1]),
    ]:
        batch.is_prefilling_np[:] = is_prefill
        manager.is_decode_sharded = not is_prefill
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
            assert member.req_id_per_token is cache.req_id_per_token
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
    prepare_pcp_query_metadata(meta, states, torch.zeros(2, dtype=torch.int64), 1)
    cache._prepare_for_batch(meta)
    prepare_pcp_query_states(cache, meta)
    captured_states = cache.runtime.request_state_indices
    # Workspace capacity can exceed this batch's Hot/Source request rows.
    assert captured_states.shape == local.idx_mapping.shape
    captured_slots = cache.slot_mapping
    prepare_pcp_query_metadata(meta, states, local.idx_mapping, 1)
    scheduler = NS(kv_connector_metadata=object(), has_sync_kv_loads=True)
    connector.pre_forward(scheduler, local, {"layer0": meta, "indexer": NS()})
    assert captured_states.data_ptr() == meta.hisparse_request_state_indices.data_ptr()
    assert captured_states.tolist() == [7, 9]
    assert meta.hisparse_num_valid_write_rows is None
    connector.prepare_host_mirror()
    assert cache.req_id_per_token is meta.req_id_per_token
    assert cache.slot_mapping[:8].tolist() == [100, 101, 102, -1, -1, -1, -1, -1]
    assert connector._pending_load_kwargs["num_tokens"] == 8
    assert captured_slots.data_ptr() == cache.slot_mapping.data_ptr()


def test_pcp_builder_reuses_state_input_without_per_draft_buffers(monkeypatch):
    from vllm_ascend.attention import sfa_hisparse
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder

    def init(builder, *args, **kwargs):
        builder.device = torch.device("cpu")
        builder.use_pcp = True
        builder.is_pcp_decode_sharded = False

    def build(builder, common, draft_index, context):
        return NS(
            query_start_loc=common.query_start_loc_cpu,
            num_input_tokens=4,
            num_prefills=0,
            pcp_has_global_prefill=context is not None,
        )

    monkeypatch.setattr(AscendSFAMetadataBuilder, "__init__", init)
    monkeypatch.setattr(AscendSFAMetadataBuilder, "_build", build)
    monkeypatch.setattr(sfa_hisparse, "get_pcp_group", lambda: NS(world_size=2))
    monkeypatch.setattr(sfa_hisparse, "needs_pcp_kv_gather", lambda *args: True)
    monkeypatch.setattr(sfa_hisparse, "compute_token_to_req_indices", lambda q, out, n, size: out[:size].zero_())
    config = NS(
        scheduler_config=NS(max_num_seqs=2, max_num_batched_tokens=8),
        compilation_config=NS(max_cudagraph_capture_size=8),
        parallel_config=NS(prefill_context_parallel_size=2),
    )
    builder = sfa_hisparse.AscendHiSparseMetadataBuilder(None, [], config, torch.device("cpu"))
    common = NS(num_reqs=3, query_start_loc_cpu=torch.tensor([0, 2, 3, 4], dtype=torch.int32))
    capture = builder._build(common, pcp_context=NS(local_request_state_indices=torch.zeros(3, dtype=torch.int64)))
    states = capture.hisparse_request_state_indices
    real = builder._build(common, pcp_context=NS(local_request_state_indices=torch.tensor([7, 7, 9])))
    assert real.hisparse_request_state_indices.data_ptr() == states.data_ptr()
    assert states.dtype == torch.int32 and states.tolist() == [7, 7, 9]
    assert real.hisparse_num_write_tokens == 8 and real.hisparse_num_valid_write_rows is None
    draft = builder._build(common, draft_index=1)
    assert not hasattr(draft, "hisparse_request_state_indices")
    assert states.tolist() == [7, 7, 9]
    builder._build(common, pcp_context=NS(local_request_state_indices=torch.tensor([9])))
    assert states.tolist() == [9, -1, -1]


@pytest.mark.parametrize("use_pcp", [False, True])
@pytest.mark.parametrize(
    "num_decodes,prefill_lens,padding",
    [(3, (4, 4), 1), (0, (4, 4), 1), (3, (), 1)],
)
def test_builder_padding_uses_safe_prefill_request_without_forward_ops(
    monkeypatch, use_pcp, num_decodes, prefill_lens, padding
):
    from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup

    from vllm_ascend.attention import sfa_hisparse
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder

    query_lens = torch.tensor([1] * num_decodes + list(prefill_lens), dtype=torch.int32)
    starts = torch.cat((torch.zeros(1, dtype=torch.int32), query_lens.cumsum(0).to(torch.int32)))
    actual = int(starts[-1])
    padded = actual + padding
    num_reqs = query_lens.numel()
    meta = NS(
        query_start_loc=starts,
        num_actual_tokens=actual,
        num_input_tokens=padded,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_prefills=len(prefill_lens),
        pcp_has_global_prefill=bool(prefill_lens),
        uses_pcp_sharded_layout=use_pcp,
        seq_lens_cpu=query_lens,
        seq_lens=query_lens,
        block_table=torch.zeros(num_reqs, 1, dtype=torch.int32),
        block_size=128,
    )
    builder = sfa_hisparse.AscendHiSparseMetadataBuilder.__new__(sfa_hisparse.AscendHiSparseMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.is_pcp_decode_sharded = True
    builder._row_capacity = padded
    builder._request_row_buffers = {}
    builder._pcp_request_state_indices = torch.empty(num_reqs, dtype=torch.int32)
    common = NS(num_reqs=num_reqs, query_start_loc_cpu=starts)
    context = NS(local_request_state_indices=torch.arange(num_reqs)) if use_pcp else None
    calls = []

    def map_requests(offsets, out, num_mapped, num_tokens):
        calls.append((num_mapped, num_tokens))
        tokens = torch.arange(num_tokens, dtype=offsets.dtype)
        rows = torch.searchsorted(offsets[1:], tokens, right=True).clamp_max(num_reqs - 1)
        out[:num_tokens].copy_(torch.where(tokens < num_mapped, rows, 0))
        return out[:num_tokens]

    monkeypatch.setattr(AscendSFAMetadataBuilder, "_build", lambda *args: meta)
    monkeypatch.setattr(sfa_hisparse, "compute_token_to_req_indices", map_requests)
    monkeypatch.setattr(sfa_hisparse, "get_pcp_group", lambda: NS(world_size=2))
    monkeypatch.setattr(
        sfa_hisparse,
        "build_hisparse_prefill_staging_plan",
        lambda table, *args: NS(block_table=table, block_size=128),
    )
    built = builder._build(common, pcp_context=context)
    assert calls == [(padded, padded)]
    expected_real = torch.repeat_interleave(torch.arange(num_reqs), query_lens.long()).to(torch.int32)
    torch.testing.assert_close(built.req_id_per_token[:actual], expected_real)
    if padding:
        assert built.req_id_per_token[-1].item() == num_reqs - 1
    if prefill_lens:
        source = torch.zeros(1, 128, 2)
        cache = NS(view=None, block_table=None, runtime=NS(gather_prefill_cache=lambda *args, **kwargs: source))
        group = NS(cache=lambda index: cache)
        _, table, request_rows = HiSparseMLAIndexGroup.stage_prefill_rows(group, 0, source, built)
        # This lookup used to fail for PCP's [3 decode, 2 prefill, 1 pad] layout.
        selected_tables = table[request_rows.long()]
        assert selected_tables.shape[0] == padded - num_decodes


def test_empty_query_rank_still_prepares_host_mirror(monkeypatch):
    from vllm_ascend.hisparse.host_mirror import HostMirrorConnector

    connector, cache, meta, _, _, _ = make_connector()
    cache.num_actual_tokens = 0  # Other PCP ranks still produce gathered KV.
    meta.hisparse_num_write_tokens = 6
    write_request_ids = torch.tensor([0, -1, -1, 1, -1, -1], dtype=torch.int32)
    meta.hisparse_write_has_prefill = False
    connector._pcp_group_metadata = [(cache, meta)]
    connector._pcp_write_request_ids = write_request_ids
    cache.runtime.eager_host_mirror = True
    finish = Mock()
    monkeypatch.setattr(ActiveKVConnector, "finish_forward", finish)
    wrapper = HostMirrorConnector(connector, connector.kv_connector.connector_worker, None, None)
    wrapper.finish_forward()
    finish.assert_called_once()
    assert cache.num_actual_tokens == 6
    assert cache.req_id_per_token is write_request_ids
    assert cache.host_mirror_required


@pytest.mark.parametrize(
    "prefill,pcp_world_size,capture,dtype",
    [
        (False, 2, False, torch.float8_e4m3fn),
        (True, 2, False, torch.int8),
        (True, 4, True, torch.float8_e4m3fn),
    ],
)
def test_pcp_prolog_gathers_local_scratch_before_resident_write(monkeypatch, capture, pcp_world_size, prefill, dtype):
    from vllm.config.compilation import CUDAGraphMode

    import vllm_ascend.attention.sfa_hisparse as module
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
    from vllm_ascend.hisparse import cache_ops

    impl = module.AscendHiSparseImpl.__new__(module.AscendHiSparseImpl)
    impl.pcp_enabled = True
    impl.is_pcp_decode_sharded = not prefill
    impl.c8_cache_dtype = dtype
    impl._kv_staging_capacity = 256
    impl.index_group = NS(logical_topk_indices=torch.empty(1))
    impl.hisparse_cache = handle()
    impl.hisparse_cache.finish_kv_update = Mock()
    meta = metadata(decode=not prefill)
    meta.uses_pcp_sharded_layout = True
    meta.num_prefills = int(prefill)
    meta.num_input_tokens = 3
    meta.pcp_has_global_prefill = prefill
    prepare_pcp_query_metadata(
        meta,
        torch.empty(2, dtype=torch.int32),
        impl.hisparse_cache.runtime.request_state_indices,
        pcp_world_size,
    )
    meta.hisparse_for_capture = capture
    num_write_rows = meta.num_input_tokens * pcp_world_size
    meta.hisparse_num_write_tokens = num_write_rows
    impl.hisparse_cache._prepare_for_batch(meta)
    impl.hisparse_cache.submit_layer_mirror = Mock()
    monkeypatch.setattr(module, "get_forward_context", lambda: NS(cudagraph_runtime_mode=CUDAGraphMode.NONE))
    resources: dict[Any, Any] = {}
    monkeypatch.setattr(
        module,
        "current_workspace_manager",
        lambda: NS(get_persistent_resource=lambda key, allocate: resources.setdefault(key, allocate())),
    )
    impl.hisparse_cache.slot_mapping = torch.arange(num_write_rows) + 10
    impl.hisparse_cache.slot_mapping[2::3] = -1
    impl.hisparse_cache.view.cache = torch.zeros(num_write_rows, 656, dtype=torch.uint8)

    def fused(self, hidden, caches, cos, sin, mapping, **kwargs):
        assert caches[0].shape == (2, 128, 1, module.C8_ROW_BYTES)
        assert caches[0].dtype == dtype
        assert mapping.tolist() == [0, 1, 2]
        assert torch.count_nonzero(impl.hisparse_cache.view.cache) == 0
        caches[0].view(torch.uint8).view(-1, 656)[:3] = torch.arange(1, 4)[:, None]
        return "queries-ready"

    def gather(payload, dim):
        assert payload.shape == (3, 656) and payload.dtype == torch.int8 and dim == 0
        return torch.cat([payload + 10 * rank for rank in range(pcp_world_size)], dim=0)

    def scatter(payload, rope, destination, mapping, valid):
        assert payload[:, 0].tolist() == [10 * rank + row for rank in range(pcp_world_size) for row in (1, 2, 3)]
        assert valid is None
        for row, slot in enumerate(mapping.tolist()):
            if slot >= 0:
                destination[slot - 10].fill_(int(payload[row, 0]))

    monkeypatch.setattr(AscendSFAImpl, "_sfa_preprocess_prolog_v3", fused)
    monkeypatch.setattr(module, "get_pcp_group", lambda: NS(all_gather=gather))
    monkeypatch.setattr(cache_ops, "scatter_packed_kv", scatter)
    assert (
        impl._sfa_preprocess_prolog_v3(
            torch.zeros(3, 8), (), None, None, torch.arange(3, dtype=torch.int64), attn_metadata=meta
        )
        == "queries-ready"
    )
    impl._sync_pcp_kv_cache((), meta)
    expected = [10 * rank + row if row else 0 for rank in range(pcp_world_size) for row in (1, 2, 0)]
    assert impl.hisparse_cache.view.cache[:, 0].tolist() == expected
    impl.hisparse_cache.finish_kv_update.assert_not_called()
    assert impl.hisparse_cache.submit_layer_mirror.call_count == int(prefill and not capture)
