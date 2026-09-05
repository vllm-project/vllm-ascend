# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.core.recompute_scheduler import RecomputeScheduler  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.manager import (  # noqa: E402
    PreemptedRequestState,
    RecomputeCPUOffloadScheduler,
    TransferMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.metadata import RecomputeCPUOffloadWorkerMetadata


@pytest.fixture
def real_scheduler():
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=2, dtype=torch.int8)
    cache = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[KVCacheTensor(size=16 * spec.page_size_bytes, shared_by=["a"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["a"], kv_cache_spec=spec)],
    )
    config = SimpleNamespace(
        speculative_config=None,
        kv_events_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1, world_size=2),
        model_config=SimpleNamespace(max_model_len=64),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        cache_config=SimpleNamespace(block_size=4),
    )
    scheduler = RecomputeCPUOffloadScheduler(config, cache, 16 * spec.page_size_bytes)
    scheduler.bind_gpu_block_pool(BlockPool(16, True, 4))
    return scheduler


@pytest.mark.parametrize("capacity", [1, 128])
def test_cpu_config_filters_unowned_descriptors_and_preserves_input(capacity):
    tensors = [KVCacheTensor(size=64, shared_by=["a", "b"]), KVCacheTensor(size=256, shared_by=[])]
    groups = []
    gpu = KVCacheConfig(num_blocks=4, kv_cache_tensors=tensors, kv_cache_groups=groups)
    cpu = RecomputeCPUOffloadScheduler._derive_cpu_config(gpu, capacity)
    assert cpu.num_blocks == max(1, capacity // 16)
    assert cpu.kv_cache_tensors == [KVCacheTensor(size=16 * cpu.num_blocks, shared_by=["a", "b"])]
    assert cpu.kv_cache_groups is groups
    assert gpu.num_blocks == 4
    assert gpu.kv_cache_tensors is tensors
    assert cpu.kv_cache_tensors[0].shared_by is not tensors[0].shared_by


def test_group_classification_handles_uniform_and_direct_specs():
    full = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=2, dtype=torch.int8)
    sliding = SlidingWindowSpec(block_size=4, num_kv_heads=1, head_size=2, dtype=torch.int8, sliding_window=8)
    mamba = MambaSpec(block_size=4, shapes=((2,),), dtypes=(torch.int8,))
    specs = [
        full,
        sliding,
        mamba,
        UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs={"a": full}),
        UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs={"a": sliding}),
        UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs={"a": mamba}),
    ]
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs])
    assert RecomputeCPUOffloadScheduler._get_group_is_sliding_window(config) == [False, True, False, False, True, False]
    assert RecomputeCPUOffloadScheduler._get_group_is_mamba(config) == [False, False, True, False, False, True]


@pytest.mark.parametrize("prefix", [False, True])
def test_preempt_restore_round_trip_balances_block_references(real_scheduler, prefix):
    scheduler = real_scheduler
    scheduler.enable_offload_prefix_caching = prefix
    blocks = scheduler._gpu_block_pool.get_new_blocks(2)
    blocks[0].set_block_hash(make_block_hash_with_group_id(b"hash-a", 0))
    request = SimpleNamespace(request_id="a", num_tokens=7)
    ids = ([b.block_id for b in blocks],)
    assert scheduler.update_state_before_preempt(request, ids, 7) is True
    assert scheduler.update_state_before_preempt(request, ids, 7) is True
    assert scheduler.has_preempted_request("a")
    assert scheduler.cpu_block_pool.get_num_free_blocks() == 13
    metadata = scheduler.build_connector_meta(SimpleNamespace(preempted_req_ids={"a"}))
    assert metadata.preempt_store_gpu_blocks == ids[0]
    assert metadata.preempt_store_event == 0
    output = KVConnectorOutput(
        kv_connector_worker_meta=RecomputeCPUOffloadWorkerMetadata(completed_store_events={0: 1})
    )
    scheduler.update_connector_output(output)
    assert scheduler._store_event_pending_counts == {0: 1}
    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
    scheduler.update_connector_output(output)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (7, True)
    destinations = scheduler._gpu_block_pool.get_new_blocks(2)
    mapping = SimpleNamespace(get_block_ids=lambda: ([b.block_id for b in destinations],))
    scheduler.update_state_after_alloc(request, mapping, 7)
    load = scheduler.build_connector_meta(SimpleNamespace(preempted_req_ids=set()))
    assert load.preempt_load_gpu_blocks == [b.block_id for b in destinations]
    assert load.preempt_load_cpu_blocks == metadata.preempt_store_cpu_blocks
    assert all(b.ref_cnt == 2 for b in destinations)
    scheduler.update_connector_output(KVConnectorOutput(finished_recving={"a", "unknown"}))
    assert all(b.ref_cnt == 1 for b in destinations)
    assert scheduler.cpu_block_pool.get_num_free_blocks() == 15
    assert not scheduler.has_preempted_request("a")
    assert scheduler._preempt_load_event_to_reqs == {}
    assert scheduler.has_pending_transfers() is False
    assert scheduler.take_events() == []
    assert scheduler.reset_cache() is True


def test_prefix_blocks_share_pending_and_completed_store_without_duplicate_copy(real_scheduler):
    scheduler = real_scheduler
    gpu = scheduler._gpu_block_pool.get_new_blocks(1)[0]
    gpu.set_block_hash(make_block_hash_with_group_id(b"shared", 0))
    for req_id in ("a", "b"):
        assert scheduler._create_preempt_state(req_id, ([gpu.block_id],), 4)
    first, second = (scheduler._preempted_req_states[k] for k in ("a", "b"))
    assert first.cpu_block_ids == second.cpu_block_ids
    assert second.store_transfer_meta == TransferMeta([], [])
    assert not second.ready
    cpu = scheduler.cpu_block_pool.blocks[first.cpu_block_ids[0][0]]
    assert cpu.ref_cnt == 2
    metadata = scheduler.build_connector_meta(SimpleNamespace(preempted_req_ids={"a", "b"}))
    assert metadata.preempt_store_gpu_blocks == [gpu.block_id]
    scheduler._process_preempt_store_event(metadata.preempt_store_event)
    assert first.ready and second.ready
    assert scheduler._create_preempt_state("c", ([gpu.block_id],), 4)
    third = scheduler._preempted_req_states["c"]
    assert third.ready
    assert third.store_transfer_meta == TransferMeta([], [])
    assert cpu.ref_cnt == 3
    for req_id in ("a", "b", "c"):
        assert scheduler.request_finished(SimpleNamespace(request_id=req_id), []) == (False, None)
    assert cpu.ref_cnt == 0
    assert scheduler.cpu_block_pool.get_num_free_blocks() == 15


@pytest.mark.parametrize("case", ["no_tokens", "unbound", "null", "capacity"])
def test_store_rejection_keeps_allocation_and_state_unchanged(real_scheduler, case):
    scheduler = real_scheduler
    ids, tokens = ([1],), 4
    if case == "no_tokens":
        tokens = 0
    elif case == "unbound":
        scheduler._gpu_block_pool = None
    elif case == "null":
        ids = ([0],)
    else:
        scheduler.cpu_block_pool.get_new_blocks(15)
    free = scheduler.cpu_block_pool.get_num_free_blocks()
    assert scheduler._create_preempt_state("a", ids, tokens) is False
    assert scheduler.cpu_block_pool.get_num_free_blocks() == free
    assert scheduler._preempted_req_states == {}


def test_mamba_offloads_only_speculative_tail_and_restores_accepted_state(real_scheduler):
    scheduler = real_scheduler
    scheduler._group_is_mamba = [True]
    scheduler.num_spec_tokens = 1
    blocks = scheduler._gpu_block_pool.get_new_blocks(4)
    ids = [b.block_id for b in blocks]
    assert scheduler._create_preempt_state("a", (ids,), 8)
    state = scheduler._preempted_req_states["a"]
    assert state.store_transfer_meta.gpu_block_ids == ids[-2:]
    assert len(state.cpu_block_ids[0]) == 2
    state.ready = True
    request = SimpleNamespace(request_id="a", num_tokens=8)
    assert scheduler._prepare_preempt_load_after_alloc(request, ([ids[0]],), 8)
    assert state.load_transfer_meta == TransferMeta([ids[0]], [state.cpu_block_ids[0][0]])


@pytest.mark.parametrize("case", ["absent", "pending", "no_tokens", "groups", "empty", "backwards", "null"])
def test_invalid_restore_mapping_preserves_block_references(real_scheduler, case):
    scheduler = real_scheduler
    state = PreemptedRequestState("r", ([1],), 8, TransferMeta([], []), ready=True)
    scheduler._preempted_req_states["r"] = state
    blocks, count = ([1],), 8
    if case == "absent":
        scheduler._preempted_req_states.clear()
    elif case == "pending":
        state.ready = False
    elif case == "no_tokens":
        count = 0
    elif case == "groups":
        blocks = ([1], [2])
    elif case == "empty":
        blocks = ([],)
    elif case == "backwards":
        state.load_start_tokens = 4
        blocks = ([],)
    elif case == "null":
        blocks = ([0],)
    if case in {"absent", "pending", "no_tokens"}:
        assert scheduler._prepare_preempt_load_after_alloc(SimpleNamespace(request_id="r"), blocks, count) is False
    else:
        with pytest.raises(RuntimeError, match="Recompute H2D"):
            scheduler._prepare_preempt_load_after_alloc(SimpleNamespace(request_id="r"), blocks, count)
    assert state.load_transfer_meta is None
    assert scheduler._gpu_block_pool.get_num_free_blocks() == 15


@pytest.mark.parametrize("same_block", [False, True])
def test_late_duplicate_hash_publication_keeps_existing_cached_block(real_scheduler, same_block):
    scheduler = real_scheduler
    first, second = scheduler.cpu_block_pool.get_new_blocks(2)
    block_hash = make_block_hash_with_group_id(b"hash", 0)
    first.set_block_hash(block_hash)
    second.set_block_hash(block_hash)
    cached = second if same_block else first
    scheduler.cpu_block_pool.cached_block_hash_to_block.insert(block_hash, cached)
    scheduler._preempt_store_event_to_blocks[4] = TransferMeta([1], [second.block_id])
    scheduler._preempt_store_event_to_reqs[4] = ["already-removed"]
    scheduler._process_preempt_store_event(4)
    assert scheduler.cpu_block_pool.cached_block_hash_to_block.get_one_block(block_hash) is cached
    assert second.block_hash == (block_hash if same_block else None)
    assert scheduler._preempt_store_event_to_blocks == {}


def test_reset_refuses_pending_transfer_then_releases_ready_request(real_scheduler):
    scheduler = real_scheduler
    block = scheduler.cpu_block_pool.get_new_blocks(1)[0]
    state = PreemptedRequestState("r", ([block.block_id],), 4, TransferMeta([], []), ready=False)
    scheduler._preempted_req_states["r"] = state
    assert scheduler.reset_cache() is False
    assert block.ref_cnt == 1
    state.ready = True
    assert scheduler.reset_cache() is True
    assert block.ref_cnt == 0
    assert scheduler._preempted_req_states == {}


@pytest.mark.parametrize("event_reqs", [None, [], ["other"], ["r", "other"]])
def test_cleanup_tolerates_missing_or_shared_load_event_entry(real_scheduler, event_reqs):
    scheduler = real_scheduler
    block = scheduler.cpu_block_pool.get_new_blocks(1)[0]
    state = PreemptedRequestState("r", ([block.block_id],), 4, TransferMeta([], []), load_event=2, ready=True)
    scheduler._preempted_req_states["r"] = state
    if event_reqs is not None:
        scheduler._preempt_load_event_to_reqs[2] = list(event_reqs)
    scheduler._cleanup_preempt_load_request("r")
    assert block.ref_cnt == 0
    assert scheduler._preempted_req_states == {}
    assert scheduler._preempt_load_event_to_reqs == ({2: ["other"]} if event_reqs and "other" in event_reqs else {})
    scheduler._cleanup_preempt_load_request("r")
    scheduler._cleanup_preempt_cache_request("r")
    assert block.ref_cnt == 0


def test_cleanup_without_load_event_releases_ready_cache(real_scheduler):
    scheduler = real_scheduler
    block = scheduler.cpu_block_pool.get_new_blocks(1)[0]
    scheduler._preempted_req_states["r"] = PreemptedRequestState(
        "r", ([block.block_id],), 4, TransferMeta([], []), ready=True
    )
    scheduler._cleanup_preempt_load_request("r")
    assert block.ref_cnt == 0
    assert scheduler.request_finished_all_groups(SimpleNamespace(request_id="r"), ([1], [2])) == (False, None)


def test_recompute_cpu_offload_scheduler_get_num_new_matched_tokens_states():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._preempted_req_states = {}
    scheduler._cleanup_preempt_cache_request = MagicMock()
    request = SimpleNamespace(request_id="req-1", num_tokens=10)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)

    scheduler._preempted_req_states["req-1"] = PreemptedRequestState(
        req_id="req-1",
        cpu_block_ids=([1],),
        num_computed_tokens=8,
        store_transfer_meta=TransferMeta([11], [1]),
        ready=False,
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)

    scheduler._preempted_req_states["req-1"].ready = True
    assert scheduler.get_num_new_matched_tokens(request, 3) == (5, True)
    assert scheduler._preempted_req_states["req-1"].load_start_tokens == 3

    assert scheduler.get_num_new_matched_tokens(request, 8) == (0, False)
    scheduler._cleanup_preempt_cache_request.assert_called_once_with("req-1")


def test_recompute_cpu_offload_scheduler_update_state_after_alloc_errors():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._prepare_preempt_load_after_alloc = MagicMock(return_value=False)
    request = SimpleNamespace(request_id="req-1")
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([1, 2],)

    scheduler.update_state_after_alloc(request, blocks, 0)
    scheduler._prepare_preempt_load_after_alloc.assert_not_called()

    try:
        scheduler.update_state_after_alloc(request, blocks, 2)
    except RuntimeError as exc:
        assert "Failed to prepare recompute H2D load" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when load mapping fails")

    scheduler._prepare_preempt_load_after_alloc.assert_called_once_with(request, ([1, 2],), 2)


def test_recompute_cpu_offload_scheduler_aligns_sliding_window_blocks():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._group_is_sliding_window = [True, False]

    assert scheduler._align_group_block_ids(0, [7, 8], 4) == [0, 0, 7, 8]
    assert scheduler._align_group_block_ids(0, [5, 6, 7, 8, 9], 4) == [
        5,
        6,
        7,
        8,
    ]
    assert scheduler._align_group_block_ids(1, [7, 8], 4) == [7, 8]
    assert scheduler._align_group_block_ids(0, [7, 8], 0) == []


def test_recompute_cpu_offload_scheduler_d2h_keeps_sliding_window_offsets():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._group_is_sliding_window = [True]
    scheduler._group_is_mamba = [False]
    scheduler.cpu_kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))]
    )
    scheduler.enable_offload_prefix_caching = False
    scheduler._pending_hash_blocks = {}
    scheduler._gpu_block_pool = SimpleNamespace(
        blocks={
            20: SimpleNamespace(block_id=20, block_hash=None),
            21: SimpleNamespace(block_id=21, block_hash=None),
        },
        _maybe_evict_cached_block=MagicMock(),
    )
    cpu_blocks = [
        SimpleNamespace(block_id=101, _block_hash=None),
        SimpleNamespace(block_id=102, _block_hash=None),
    ]
    scheduler.cpu_block_pool = SimpleNamespace(
        get_num_free_blocks=MagicMock(return_value=8),
        get_new_blocks=MagicMock(return_value=cpu_blocks),
        cached_block_hash_to_block=SimpleNamespace(get_one_block=MagicMock(return_value=None)),
    )
    scheduler._preempted_req_states = {}

    assert scheduler._create_preempt_state("req-1", ([20, 21],), 64) is True

    state = scheduler._preempted_req_states["req-1"]
    assert state.cpu_block_ids == ([0, 0, 101, 102],)
    assert state.store_transfer_meta == TransferMeta([20, 21], [101, 102])
    assert state.ready is False


def test_recompute_cpu_offload_scheduler_h2d_skips_sliding_window_null_blocks():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._group_is_sliding_window = [True]
    scheduler._group_is_mamba = [False]
    scheduler.cpu_kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))]
    )
    scheduler._gpu_block_pool = SimpleNamespace(blocks={30: "gpu30", 31: "gpu31"}, touch=MagicMock())
    scheduler._preempted_req_states = {
        "req-1": PreemptedRequestState(
            req_id="req-1",
            cpu_block_ids=([0, 0, 4, 5],),
            num_computed_tokens=64,
            store_transfer_meta=TransferMeta([20, 21], [4, 5]),
            load_start_tokens=0,
            ready=True,
        )
    }

    prepared = scheduler._prepare_preempt_load_after_alloc(
        SimpleNamespace(request_id="req-1"),
        ([30, 31],),
        num_external_tokens=64,
    )

    assert prepared is True
    state = scheduler._preempted_req_states["req-1"]
    assert state.load_transfer_meta == TransferMeta([30, 31], [4, 5])
    touched = list(scheduler._gpu_block_pool.touch.call_args.args[0])
    assert touched == ["gpu30", "gpu31"]


def test_recompute_cpu_offload_scheduler_h2d_clips_mtp_tail_blocks():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._group_is_sliding_window = [False]
    scheduler._group_is_mamba = [False]
    scheduler.cpu_kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))]
    )
    scheduler._gpu_block_pool = SimpleNamespace(
        blocks={10: "gpu10", 11: "gpu11", 12: "gpu12"},
        touch=MagicMock(),
    )
    scheduler._preempted_req_states = {
        "req-1": PreemptedRequestState(
            req_id="req-1",
            cpu_block_ids=([1, 2, 3, 4],),
            num_computed_tokens=64,
            store_transfer_meta=TransferMeta([20, 21, 22, 23], [1, 2, 3, 4]),
            load_start_tokens=0,
            ready=True,
        )
    }

    prepared = scheduler._prepare_preempt_load_after_alloc(
        SimpleNamespace(request_id="req-1"),
        ([10, 11, 12],),
        num_external_tokens=64,
    )

    assert prepared is True
    state = scheduler._preempted_req_states["req-1"]
    assert state.load_transfer_meta == TransferMeta([10, 11, 12], [1, 2, 3])


def test_recompute_cpu_offload_scheduler_request_finished_ready_and_pending():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._preempted_req_states = {
        "ready": PreemptedRequestState(
            req_id="ready",
            cpu_block_ids=([1],),
            num_computed_tokens=8,
            store_transfer_meta=TransferMeta([11], [1]),
            ready=True,
        ),
        "pending": PreemptedRequestState(
            req_id="pending",
            cpu_block_ids=([2],),
            num_computed_tokens=8,
            store_transfer_meta=TransferMeta([12], [2]),
            ready=False,
        ),
        "loading": PreemptedRequestState(
            req_id="loading",
            cpu_block_ids=([3],),
            num_computed_tokens=8,
            store_transfer_meta=TransferMeta([13], [3]),
            load_event=5,
            ready=True,
        ),
    }
    scheduler._cleanup_preempt_cache_request = MagicMock()

    assert scheduler.request_finished(SimpleNamespace(request_id="ready"), []) == (
        False,
        None,
    )
    assert scheduler.request_finished(SimpleNamespace(request_id="pending"), []) == (False, None)
    assert scheduler.request_finished(SimpleNamespace(request_id="loading"), []) == (False, None)

    scheduler._cleanup_preempt_cache_request.assert_called_once_with("ready")
    assert scheduler._preempted_req_states["pending"].finished is True
    assert scheduler._preempted_req_states["loading"].finished is False


def test_recompute_cpu_offload_scheduler_process_store_event_finishes_pending_req():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    cpu_block = MagicMock()
    cpu_block.block_hash = None
    scheduler.cpu_block_pool = SimpleNamespace(blocks={4: cpu_block})
    scheduler._preempt_store_event_to_blocks = {7: TransferMeta([1], [4])}
    scheduler._preempt_store_event_to_reqs = {7: ["req-1"]}
    scheduler._preempted_req_states = {
        "req-1": PreemptedRequestState(
            req_id="req-1",
            cpu_block_ids=([4],),
            num_computed_tokens=8,
            store_transfer_meta=TransferMeta([1], [4]),
            ready=False,
            finished=True,
        )
    }
    scheduler._cleanup_preempt_cache_request = MagicMock()

    scheduler._process_preempt_store_event(7)

    assert scheduler._preempted_req_states["req-1"].ready is True
    scheduler._cleanup_preempt_cache_request.assert_called_once_with("req-1")
    assert scheduler._preempt_store_event_to_blocks == {}
    assert scheduler._preempt_store_event_to_reqs == {}


def test_recompute_cpu_offload_scheduler_pending_and_reset_cache_paths():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._store_event_pending_counts = {}
    scheduler._preempt_store_event_to_blocks = {}
    scheduler._preempted_req_states = {}

    assert scheduler.has_pending_transfers() is False

    scheduler._preempted_req_states["not-ready"] = PreemptedRequestState(
        req_id="not-ready",
        cpu_block_ids=([1],),
        num_computed_tokens=8,
        store_transfer_meta=TransferMeta([11], [1]),
        ready=False,
    )
    assert scheduler.has_pending_transfers() is True

    scheduler._preempted_req_states.clear()
    scheduler._preempt_store_event_to_reqs = {"unused": []}
    scheduler._preempt_load_event_to_reqs = {1: ["req-1"]}
    scheduler._pending_hash_blocks = {"hash": MagicMock()}
    scheduler.cpu_block_pool = MagicMock()
    scheduler.cpu_block_pool.reset_prefix_cache.return_value = True
    scheduler._cleanup_preempt_cache_request = MagicMock()

    assert scheduler.reset_cache() is True
    scheduler.cpu_block_pool.reset_prefix_cache.assert_called_once_with()
    assert scheduler._preempt_store_event_to_reqs == {}
    assert scheduler._preempt_load_event_to_reqs == {}
    assert scheduler._pending_hash_blocks == {}


def test_recompute_cpu_offload_scheduler_cleanup_preempt_load_request():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._preempt_load_event_to_reqs = {2: ["req-1"]}
    scheduler._preempted_req_states = {
        "req-1": PreemptedRequestState(
            req_id="req-1",
            cpu_block_ids=([4],),
            num_computed_tokens=8,
            store_transfer_meta=TransferMeta([1], [4]),
            load_event=2,
            load_transfer_meta=TransferMeta([10, 11], [4, 5]),
            ready=True,
        )
    }
    scheduler._gpu_block_pool = SimpleNamespace(
        blocks={10: "gpu10", 11: "gpu11"},
        free_blocks=MagicMock(),
    )
    scheduler._cleanup_preempt_cache_request = MagicMock()

    scheduler._cleanup_preempt_load_request("req-1")

    assert scheduler._preempt_load_event_to_reqs == {}
    freed = list(scheduler._gpu_block_pool.free_blocks.call_args.args[0])
    assert freed == ["gpu10", "gpu11"]
    scheduler._cleanup_preempt_cache_request.assert_called_once_with("req-1")


def test_recompute_cpu_offload_scheduler_cleanup_skips_null_cpu_blocks():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._preempted_req_states = {
        "req-1": PreemptedRequestState(
            req_id="req-1",
            cpu_block_ids=([0, 4], [0]),
            num_computed_tokens=32,
            store_transfer_meta=TransferMeta([10], [4]),
            ready=True,
        )
    }
    scheduler.cpu_block_pool = SimpleNamespace(blocks={4: "cpu4"}, free_blocks=MagicMock())

    scheduler._cleanup_preempt_cache_request("req-1")

    freed = list(scheduler.cpu_block_pool.free_blocks.call_args.args[0])
    assert freed == ["cpu4"]


def test_recompute_scheduler_remote_kv_restore_keeps_exact_token_position():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.connector = MagicMock()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_recving_kv_req_ids = {"req-1"}
    scheduler.kv_cache_manager = MagicMock()

    request = SimpleNamespace(
        request_id="req-1",
        num_computed_tokens=9,
        num_tokens=9,
        num_preemptions=1,
        spec_token_ids=[],
    )

    scheduler._update_waiting_for_remote_kv(request)

    scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(request, 8)
    assert request.num_computed_tokens == 8
    assert request.spec_token_ids == []
    assert scheduler.finished_recving_kv_req_ids == set()


def test_recompute_scheduler_remote_kv_restore_frees_failed_empty_load():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.connector = MagicMock()
    scheduler.failed_recving_kv_req_ids = {"req-1"}
    scheduler.finished_recving_kv_req_ids = {"req-1"}
    scheduler.kv_cache_manager = MagicMock()

    request = SimpleNamespace(
        request_id="req-1",
        num_computed_tokens=0,
    )

    scheduler._update_waiting_for_remote_kv(request)

    scheduler.kv_cache_manager.free.assert_called_once_with(request)
    scheduler.kv_cache_manager.cache_blocks.assert_not_called()
    assert scheduler.failed_recving_kv_req_ids == set()
    assert scheduler.finished_recving_kv_req_ids == set()


def test_recompute_cpu_offload_scheduler_build_connector_meta_assigns_events():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._store_event_counter = 4
    scheduler._load_event_counter = 7
    scheduler._preempt_store_event_to_blocks = {}
    scheduler._preempt_store_event_to_reqs = {}
    scheduler._preempt_load_event_to_reqs = {}
    scheduler._pending_hash_blocks = {"hash": MagicMock()}
    scheduler._preempted_req_states = {
        "store-req": PreemptedRequestState(
            req_id="store-req",
            cpu_block_ids=([2],),
            num_computed_tokens=8,
            store_transfer_meta=TransferMeta([10], [2]),
            ready=False,
        ),
        "load-req": PreemptedRequestState(
            req_id="load-req",
            cpu_block_ids=([3],),
            num_computed_tokens=8,
            store_transfer_meta=TransferMeta([], []),
            load_transfer_meta=TransferMeta([11], [3]),
            ready=True,
        ),
    }
    scheduler_output = SimpleNamespace(preempted_req_ids={"store-req"})

    metadata = scheduler.build_connector_meta(scheduler_output)

    assert metadata.need_flush is True
    assert metadata.preempt_store_event == 4
    assert metadata.preempt_store_gpu_blocks == [10]
    assert metadata.preempt_store_cpu_blocks == [2]
    assert metadata.preempt_load_event == 7
    assert metadata.preempt_load_gpu_blocks == [11]
    assert metadata.preempt_load_cpu_blocks == [3]
    assert metadata.preempt_load_event_to_reqs == {7: ["load-req"]}
    assert scheduler._preempted_req_states["store-req"].store_event == 4
    assert scheduler._preempted_req_states["load-req"].load_event == 7
    assert scheduler._pending_hash_blocks == {}


def test_recompute_cpu_offload_scheduler_update_connector_output_marks_store_ready():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._expected_worker_count = 2
    scheduler._store_event_pending_counts = {}
    scheduler._preempted_req_states = {}
    scheduler._process_preempt_store_event = MagicMock()
    output = KVConnectorOutput(
        finished_recving=set(),
        kv_connector_worker_meta=RecomputeCPUOffloadWorkerMetadata(completed_store_events={5: 1}),
    )

    scheduler.update_connector_output(output)

    assert scheduler._store_event_pending_counts == {5: 1}
    scheduler._process_preempt_store_event.assert_not_called()

    scheduler.update_connector_output(output)

    assert scheduler._store_event_pending_counts == {}
    scheduler._process_preempt_store_event.assert_called_once_with(5)
