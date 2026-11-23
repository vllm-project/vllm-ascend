#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""Same-step preempt+readmit regression (reset_prefix_cache flow).

reset_prefix_cache(reset_running_requests=True) preempts running requests
OUT OF BAND (between schedule() calls); the preempted ids survive in
scheduler.reset_preempted_req_ids until _update_after_schedule clears them
AFTER the step, so the NEXT schedule() re-admits the request from the
waiting queue in the SAME SchedulerOutput that still carries
preempted_req_ids={R}. build_connector_meta processed preempted ids before
scheduled_new_reqs and popped the registration update_state_after_alloc had
just re-created, so _process_new_request raised
"Request R is not in _unfinished_requests, but it is scheduled as a new
request" and killed the engine.

The re-admitted request must keep the registration its NEW allocation
created, but every trace of the ABORTED attempt has to be cleared: the
scheduler-side tracker (its GVA state belongs to the old block table), a
stale async-load entry, and on the worker the aborted attempt's durable
save-offset high-water mark, queued store jobs and layerwise snapshot keys
and load sessions (otherwise kv_transfer.py skips saving blocks up to the
old offset for the new attempt). The new attempt starts in the very step
that reports the preemption, so the worker-side state must be retired when
that step's metadata is bound, before the step's own load or save runs.

The tests drive the REAL KVPoolScheduler entry points
(get_num_new_matched_tokens / update_state_after_alloc /
build_connector_meta) with a REAL SchedulerOutput and NewRequestData, and
the REAL connector bind and worker get_finished paths with a REAL
KVCacheStoreSendingThread and a REAL LayerwiseSessionTracker; only the
external store client and the VllmConfig input are mocked,
following test_pool_scheduler.py conventions. Everything used here exists
on main.
"""

import threading
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.sched.output import CachedRequestData, KVConnectorBlockState, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import backend_map, get_backend_class
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.session_tracker import LayerwiseSessionTracker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreSendingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

REQ_ID = "m4b-retry-r-92350492"
OTHER_REQ_ID = "m4b-retry-r-other"
PROMPT_TOKENS = 315
BLOCK_SIZE = 16
HIT_TOKENS = 256
# The pre-preemption attempt had saved this far into the request before it
# was aborted; the re-admitted attempt must NOT inherit the mark.
OLD_SAVED_OFFSET = 128
# The new attempt's first save reaches less far than the aborted attempt had.
NEW_SAVED_OFFSET = 64
OLD_GVA = 7
OLD_GVA_OFFSET = 3


@pytest.fixture(autouse=True)
def _mock_scheduler_clients():
    """Resolve real backend capabilities while mocking external store creation."""
    with ExitStack() as stack:
        for backend_name in backend_map:
            backend_class = get_backend_class(backend_name)
            assert backend_class is not None
            stack.enter_context(patch.object(backend_class, "create_scheduler_client"))
        yield


def _make_config(*, load_async: bool = False) -> MagicMock:
    config = MagicMock()
    config.kv_transfer_config.kv_role = "kv_both"
    config.kv_transfer_config.kv_connector_extra_config = {"load_async": True} if load_async else {}
    config.kv_transfer_config.get_from_extra_config.return_value = True
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_index = 0
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.rank = 0
    config.parallel_config.world_size = 1
    config.cache_config.block_size = BLOCK_SIZE
    config.cache_config.hash_block_size = BLOCK_SIZE
    config.cache_config.prefix_cache_retention_interval = None
    config.cache_config.prefix_match_unit = None
    config.model_config.model = "org/tiny-qwen3"
    config.model_config.use_mla = False
    config.model_config.hf_text_config = MagicMock(spec=[])
    config.model_config.get_total_num_kv_heads.return_value = 2
    config.model_config.get_num_layers.return_value = 4
    # MRv2 is the only production runner; preempted requests resume through
    # scheduled_new_reqs with a full block table.
    config.use_v2_model_runner = True
    config.kv_events_config = MagicMock(enable_kv_cache_events=False)
    return config


def _make_request(req_id: str = REQ_ID) -> MagicMock:
    request = MagicMock()
    request.request_id = req_id
    request.prompt_token_ids = list(range(PROMPT_TOKENS))
    request.num_tokens = PROMPT_TOKENS
    request.num_computed_tokens = 0
    request.all_token_ids = list(range(PROMPT_TOKENS))
    request.block_hashes = [bytes([i]) * 4 for i in range(8)]
    return request


def _make_blocks() -> MagicMock:
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([0, 1, 2, 3, 4, 5, 6, 7],)
    return blocks


def _resolved_cache_config() -> SimpleNamespace:
    """Core's resolved KV cache configuration for the scheduler constructor.

    main requires one and reads kv_cache_groups, hash_block_size and
    cache_hit_alignment_tokens from it; older bases only read kv_cache_groups.
    One full-attention group of BLOCK_SIZE-token blocks."""
    return SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.attn"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=BLOCK_SIZE, num_kv_heads=2, head_size=8, dtype=torch.float32
                ),
            )
        ],
        num_blocks=64,
        hash_block_size=BLOCK_SIZE,
        cache_hit_alignment_tokens=BLOCK_SIZE,
        prefix_cache_retention_interval=None,
    )


class _LookupHit(int):
    """The store's lookup answer: an int on older bases, an object carrying
    hit_length (and tail key boundaries) since main's fine-grained prefix
    matching. One value satisfies both readings."""

    tail_key_boundaries = ()

    @property
    def hit_length(self) -> int:
        return int(self)


def _make_new_request_data(req_id: str = REQ_ID) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(range(PROMPT_TOKENS)),
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0, 1, 2, 3, 4, 5, 6, 7],),
        num_computed_tokens=0,
        lora_request=None,
        prefill_token_ids=None,
    )


def _make_scheduler_output(new_reqs: list[NewRequestData], preempted: set[str]) -> SchedulerOutput:
    scheduled = {req.req_id: PROMPT_TOKENS for req in new_reqs}
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[], new_token_ids=[], new_block_ids=[], num_computed_tokens=[], num_output_tokens=[]
        ),
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        preempted_req_ids=preempted,
        # main's scheduler reads Core's current block tables for save jobs.
        kv_connector_block_state=KVConnectorBlockState(
            req_ids={req.req_id for req in new_reqs},
            resolve_block_ids=lambda _req_id: ([0, 1, 2, 3, 4, 5, 6, 7],),
            boundary_state_offloads={},
        ),
    )


def _admit_request(scheduler: KVPoolScheduler, request: MagicMock, *, hit: bool = True) -> None:
    """One schedule()-step admission through the real entry points."""
    need, _ = scheduler.get_num_new_matched_tokens(request, 0)
    assert need == (HIT_TOKENS if hit else 0)
    scheduler.update_state_after_alloc(request, _make_blocks(), HIT_TOKENS if hit else 0)


@pytest.fixture
def scheduler_with_lookup_hit() -> KVPoolScheduler:
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient"
    ) as mock_client_cls:
        mock_client_cls.return_value.lookup.return_value = _LookupHit(HIT_TOKENS)
        yield KVPoolScheduler(_make_config(), use_layerwise=False, kv_cache_config=_resolved_cache_config())


@pytest.fixture
def scheduler_with_async_load() -> KVPoolScheduler:
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient"
    ) as mock_client_cls:
        scheduler = KVPoolScheduler(
            _make_config(load_async=True), use_layerwise=False, kv_cache_config=_resolved_cache_config()
        )
        assert scheduler.load_async is True
        scheduler.client = mock_client_cls.return_value
        scheduler.client.closed = False
    return scheduler


def _make_send_thread() -> KVCacheStoreSendingThread:
    """A real sending thread; its queue is never served, so no store is touched."""
    send_thread = KVCacheStoreSendingThread.__new__(KVCacheStoreSendingThread)
    send_thread.done_task_lock = threading.RLock()
    send_thread._saved_offsets = {}
    send_thread._retry_token_ids = {}
    send_thread.stored_requests = {}
    send_thread._next_local_store_job_id = -1
    send_thread.finished_requests = set()
    return send_thread


def _evidence_buffer():
    """The worker's evidence buffer. On trees carrying the AscendStore
    evidence path (vllm-project/vllm-ascend#22651) get_finished drops
    buffered evidence for finished | preempted requests, so the stub needs
    a real buffer; on main nothing reads the attribute and it is None."""
    try:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.evidence import EvidenceBuffer
    except ImportError:
        return None
    return EvidenceBuffer()


def _make_worker(send_thread: KVCacheStoreSendingThread | None) -> KVPoolWorker:
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.kv_send_thread = send_thread
    worker.kv_recv_thread = None
    worker.load_async = False
    worker.tp_rank = 0
    worker._retired_store_req_ids = set()
    worker.evidence_buffer = _evidence_buffer()
    return worker


# ---------------------------------------------------------------------------
# Lifecycle: the same-step overlap must not crash the scheduler
# ---------------------------------------------------------------------------


def test_same_step_preempt_readmit_delivers_new_attempt_without_crash(
    scheduler_with_lookup_hit: KVPoolScheduler,
):
    scheduler = scheduler_with_lookup_hit
    request = _make_request()

    # Step N: R admitted and registered.
    _admit_request(scheduler, request)
    assert REQ_ID in scheduler._unfinished_requests

    # reset_prefix_cache(reset_running_requests=True) between steps: the
    # connector's scheduler role receives no callback; the preemption is
    # surfaced only via the NEXT SchedulerOutput's preempted_req_ids.

    # Step N+1: R re-admitted from waiting in the SAME step whose output
    # still carries preempted_req_ids={R}.
    _admit_request(scheduler, request)
    assert REQ_ID in scheduler._unfinished_requests

    output = _make_scheduler_output([_make_new_request_data()], {REQ_ID})
    # Before the fix this raises ValueError: "Request ... is not in
    # _unfinished_requests, but it is scheduled as a new request".
    meta = scheduler.build_connector_meta(output)

    # The re-admitted request keeps the registration its new allocation
    # created, and its new attempt's metadata is delivered.
    assert REQ_ID in scheduler._unfinished_requests
    assert REQ_ID in scheduler._request_trackers
    assert [req_meta.req_id for req_meta in meta.requests] == [REQ_ID]
    assert meta.requests[0].load_spec is not None
    assert meta.requests[0].load_spec.can_load
    # A request re-admitted this step is NOT abandoned and must not appear
    # in the connector metadata's preempted set.
    assert REQ_ID not in meta.preempted_req_ids
    assert REQ_ID not in scheduler._preempted_req_ids


def test_preempt_without_readmit_still_treats_request_as_abandoned(
    scheduler_with_lookup_hit: KVPoolScheduler,
):
    """Ordinary preemption (no same-step re-admission) semantics are unchanged."""
    scheduler = scheduler_with_lookup_hit
    request = _make_request()
    _admit_request(scheduler, request)
    assert REQ_ID in scheduler._unfinished_requests

    output = _make_scheduler_output([], {REQ_ID})
    meta = scheduler.build_connector_meta(output)

    assert REQ_ID not in scheduler._unfinished_requests
    assert REQ_ID not in scheduler._request_trackers
    assert REQ_ID in scheduler._preempted_req_ids
    assert REQ_ID in meta.preempted_req_ids


def test_mixed_preempt_output_isolates_readmitted_request_from_abandoned(
    scheduler_with_lookup_hit: KVPoolScheduler,
):
    """A readmitted request in a mixed preempted set must not leak into the
    abandoned bookkeeping (nor into the metadata the worker consumes)."""
    scheduler = scheduler_with_lookup_hit
    request = _make_request()
    other = _make_request(OTHER_REQ_ID)
    _admit_request(scheduler, request)
    _admit_request(scheduler, other)

    # R was preempted out of band and re-admitted this step; Q was preempted
    # in the ordinary way and stays in waiting.
    _admit_request(scheduler, request)
    output = _make_scheduler_output([_make_new_request_data()], {REQ_ID, OTHER_REQ_ID})
    meta = scheduler.build_connector_meta(output)

    assert REQ_ID in scheduler._unfinished_requests
    assert REQ_ID in scheduler._request_trackers
    assert OTHER_REQ_ID not in scheduler._unfinished_requests
    assert OTHER_REQ_ID not in scheduler._request_trackers
    assert scheduler._preempted_req_ids == {OTHER_REQ_ID}
    assert meta.preempted_req_ids == {OTHER_REQ_ID}
    assert [req_meta.req_id for req_meta in meta.requests] == [REQ_ID]


def test_ordinary_step_touches_no_preemption_bookkeeping():
    """A step without preemptions must not touch the preemption bookkeeping
    at all: main's metadata tests build the scheduler through __new__ and never
    set _preempted_req_ids or load_specs. (RED with the unguarded updates.)"""
    scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
    scheduler.kv_role = "kv_both"
    scheduler.can_put = False
    scheduler._request_trackers = {}
    scheduler._unfinished_requests = {}
    scheduler._async_loading_req_ids = set()
    # Attributes main's own metadata stubs carry; older bases never read them.
    scheduler.use_layerwise = False
    scheduler._finished_partial_tail_metas = {}

    meta = scheduler.build_connector_meta(_make_scheduler_output([], set()))

    assert meta.requests == []
    assert meta.preempted_req_ids == set()
    assert meta.readmitted_req_ids == set()


# ---------------------------------------------------------------------------
# Stale state of the ABORTED attempt must not leak into the re-admitted one
# ---------------------------------------------------------------------------


def test_readmitted_request_does_not_inherit_aborted_attempt_tracker_state(
    scheduler_with_lookup_hit: KVPoolScheduler,
):
    """The aborted attempt's tracker (GVA state belongs to its OLD block
    table) must be cleared: _process_new_request inherits block_gvas and
    gva_block_offset from whatever tracker is still registered."""
    scheduler = scheduler_with_lookup_hit
    request = _make_request()
    # Step N: admitted and built. The build is what registers the tracker.
    _admit_request(scheduler, request)
    scheduler.build_connector_meta(_make_scheduler_output([_make_new_request_data()], set()))
    assert REQ_ID in scheduler._request_trackers
    # While running, the attempt accumulated GVA state for its block table
    # before it was preempted out of band.
    scheduler._request_trackers[REQ_ID].block_gvas.extend([OLD_GVA] * 8)
    scheduler._request_trackers[REQ_ID].gva_block_offset = OLD_GVA_OFFSET

    # Step N+1: re-admitted in the same step that reports the preemption.
    _admit_request(scheduler, request)
    meta = scheduler.build_connector_meta(_make_scheduler_output([_make_new_request_data()], {REQ_ID}))

    tracker = scheduler._request_trackers[REQ_ID]
    assert tracker.block_gvas == []
    assert tracker.gva_block_offset == 0
    assert meta.requests[0].block_gvas_np.tolist() == []
    assert meta.requests[0].gva_block_offset == 0


def test_readmitted_request_clears_stale_async_load_entry_when_new_attempt_does_not_load(
    scheduler_with_async_load: KVPoolScheduler,
):
    """With async loading, attempt 1 registered an async-load entry. The
    re-admitted attempt misses the store (num_external_tokens == 0), which
    update_state_after_alloc does not discard; the entry must not leak."""
    scheduler = scheduler_with_async_load
    request = _make_request()
    # Attempt 1 hit the store and registered an async load.
    scheduler.client.lookup.return_value = _LookupHit(HIT_TOKENS)
    _admit_request(scheduler, request)
    assert REQ_ID in scheduler._async_loading_req_ids

    # Attempt 2 (re-admitted this step) misses the store. Per the reviewer's
    # finding, update_state_after_alloc does not discard the entry itself.
    scheduler.client.lookup.return_value = None
    _admit_request(scheduler, request, hit=False)
    assert REQ_ID in scheduler._async_loading_req_ids, "precondition: entry survives the re-admission"

    meta = scheduler.build_connector_meta(_make_scheduler_output([_make_new_request_data()], {REQ_ID}))

    assert REQ_ID not in scheduler._async_loading_req_ids
    assert REQ_ID not in meta.loading_req_ids
    assert [req_meta.req_id for req_meta in meta.requests] == [REQ_ID]


def _make_connector(worker, use_layerwise: bool = False):
    """The connector facade bound to a worker, built the way the connector
    tests build it (no NPU, no __init__)."""
    connector = AscendStoreConnector.__new__(AscendStoreConnector)
    connector.connector_worker = worker
    connector.use_layerwise = use_layerwise
    return connector


def _readmitted_meta(scheduler: KVPoolScheduler) -> AscendConnectorMetadata:
    """The metadata of the step that reports R's preemption and re-admits it."""
    request = _make_request()
    _admit_request(scheduler, request)
    _admit_request(scheduler, request)
    meta = scheduler.build_connector_meta(_make_scheduler_output([_make_new_request_data()], {REQ_ID}))
    assert meta.readmitted_req_ids == {REQ_ID}
    assert REQ_ID not in meta.preempted_req_ids
    return meta


def test_bind_retires_aborted_attempt_before_this_steps_save(scheduler_with_lookup_hit: KVPoolScheduler):
    """The worker-side half. kv_transfer.py skips saving blocks up to the
    durable saved-offset high-water mark, so the aborted attempt's mark, its
    queued store job and its retry tokens must be gone BEFORE this step's
    save of the new attempt runs. The connector therefore retires the
    attempts when the step's metadata is bound; the end-of-step get_finished
    must then leave the new attempt's own mark alone."""
    meta = _readmitted_meta(scheduler_with_lookup_hit)

    send_thread = _make_send_thread()
    send_thread._saved_offsets[REQ_ID] = OLD_SAVED_OFFSET
    send_thread._retry_token_ids[REQ_ID] = (OLD_SAVED_OFFSET, [1, 2, 3])
    send_thread.stored_requests[REQ_ID] = {-1}
    connector = _make_connector(_make_worker(send_thread))

    connector.bind_connector_metadata(meta)

    assert send_thread.get_saved_offset(REQ_ID) == 0
    assert send_thread._retry_token_ids.get(REQ_ID) is None
    assert not send_thread.is_stored_request(REQ_ID)
    # This step's save of the new attempt records its own (shorter) extent...
    new_job = SimpleNamespace(req_id=REQ_ID, store_job_id=None)
    send_thread.add_stored_request(new_job)
    send_thread.record_saved_offset(new_job, NEW_SAVED_OFFSET)
    assert send_thread.get_saved_offset(REQ_ID) == NEW_SAVED_OFFSET
    # ...and the end-of-step drain leaves a request that is not abandoned alone.
    connector.connector_worker.get_finished(set(), meta)
    assert send_thread.get_saved_offset(REQ_ID) == NEW_SAVED_OFFSET
    assert send_thread.is_stored_request(REQ_ID)


def test_bind_retires_re_admitted_attempts_before_the_layerwise_step_is_prepared():
    """The layerwise step's tasks are prepared when the metadata is bound; the
    aborted attempts' sessions must already be released by then."""
    worker = MagicMock()

    _make_connector(worker, use_layerwise=True).bind_connector_metadata(
        AscendConnectorMetadata(set(), set(), readmitted_req_ids={REQ_ID})
    )

    calls = [name for name, _args, _kwargs in worker.method_calls]
    assert calls.index("retire_aborted_attempts") < calls.index("prepare_layerwise_step")
    worker.retire_aborted_attempts.assert_called_once_with({REQ_ID})

    quiet = MagicMock()
    _make_connector(quiet, use_layerwise=True).bind_connector_metadata(AscendConnectorMetadata(set(), set()))
    quiet.retire_aborted_attempts.assert_not_called()


def test_running_request_keeps_its_save_offset_across_the_step(scheduler_with_lookup_hit: KVPoolScheduler):
    """Negative control: a request that is neither preempted nor re-admitted
    keeps its durable saved-offset high-water mark through bind and drain."""
    scheduler = scheduler_with_lookup_hit
    request = _make_request()
    _admit_request(scheduler, request)
    meta = scheduler.build_connector_meta(_make_scheduler_output([_make_new_request_data()], set()))
    send_thread = _make_send_thread()
    send_thread._saved_offsets[REQ_ID] = OLD_SAVED_OFFSET
    connector = _make_connector(_make_worker(send_thread))

    connector.bind_connector_metadata(meta)
    assert send_thread.get_saved_offset(REQ_ID) == OLD_SAVED_OFFSET
    connector.connector_worker.get_finished(set(), meta)
    assert send_thread.get_saved_offset(REQ_ID) == OLD_SAVED_OFFSET


def test_abandoned_request_save_offset_is_reset_by_the_drain(scheduler_with_lookup_hit: KVPoolScheduler):
    """Control: a request that stays abandoned has no new attempt, so nothing
    is retired at bind; the end-of-step drain resets its mark, as before."""
    scheduler = scheduler_with_lookup_hit
    request = _make_request()
    _admit_request(scheduler, request)
    meta = scheduler.build_connector_meta(_make_scheduler_output([], {REQ_ID}))
    assert meta.readmitted_req_ids == set()
    send_thread = _make_send_thread()
    send_thread._saved_offsets[REQ_ID] = OLD_SAVED_OFFSET
    connector = _make_connector(_make_worker(send_thread))

    connector.bind_connector_metadata(meta)
    assert send_thread.get_saved_offset(REQ_ID) == OLD_SAVED_OFFSET
    connector.connector_worker.get_finished(set(), meta)
    assert send_thread.get_saved_offset(REQ_ID) == 0


# ---------------------------------------------------------------------------
# Layerwise worker: the aborted attempt's block-table state is stale too
# ---------------------------------------------------------------------------


def _make_layerwise_worker() -> LayerwiseKVPoolWorker:
    worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
    worker.kv_send_thread = None
    worker.kv_recv_thread = None
    worker.load_async = False
    worker.tp_rank = 0
    worker._retired_store_req_ids = set()
    worker._snapshot_keys = {}
    worker._step_load_keys = set()
    worker.use_block_key_layerwise = True
    worker._load_session_lock = threading.RLock()
    worker._layerwise_session_tracker = LayerwiseSessionTracker()
    worker.m_store = MagicMock()
    worker._layerwise_key_batches = lambda keys: [list(keys)] if keys else []
    worker.evidence_buffer = _evidence_buffer()
    return worker


def test_layerwise_retire_drops_aborted_keys_and_keeps_this_steps_keys():
    """_snapshot_keys[req] maps block index -> private key for the block table
    of the attempt that saved it. The aborted attempt's mapping must be gone
    before this step's save writes the new attempt's keys, and the end-of-step
    drain must then leave those alone."""
    worker = _make_layerwise_worker()
    worker._snapshot_keys[REQ_ID] = {0: {0: "private-old", 1: "private-old-2"}}
    worker._snapshot_keys["untouched"] = {0: {0: "keep"}}

    worker.retire_aborted_attempts({REQ_ID})

    assert worker._snapshot_keys == {"untouched": {0: {0: "keep"}}}
    worker._snapshot_keys[REQ_ID] = {0: {0: "private-new"}}
    worker.get_finished(set(), AscendConnectorMetadata(set(), set(), readmitted_req_ids={REQ_ID}))
    assert worker._snapshot_keys == {"untouched": {0: {0: "keep"}}, REQ_ID: {0: {0: "private-new"}}}


def test_layerwise_retire_releases_aborted_sessions_without_retiring_the_request():
    """A re-admitted request is NOT terminal: its aborted attempt's active
    sessions are released (retry path) while the request entries its new
    attempt's retry needs survive."""
    worker = _make_layerwise_worker()
    tracker = worker._layerwise_session_tracker
    tracker.prepare_load_entries(REQ_ID, [("key-a", 0), ("key-b", 1)])
    tracker.record_get_result("key-a", [REQ_ID], succeeded=True)
    tracker.prepare_load_entries(OTHER_REQ_ID, [("key-c", 0)])
    tracker.record_get_result("key-c", [OTHER_REQ_ID], succeeded=True)

    worker.retire_aborted_attempts({REQ_ID})

    # The aborted attempt's active session is ended ...
    assert "key-a" not in tracker._load_key_owners
    # ... but the request's load entries are retained for the retry ...
    assert set(tracker._request_load_entries[REQ_ID]) == {"key-a", "key-b"}
    # ... and other requests are untouched.
    assert tracker._load_key_owners["key-c"] == {OTHER_REQ_ID}


def test_layerwise_worker_retires_abandoned_request_completely():
    """Control: an abandoned (plainly preempted) request IS terminal; all of
    its session state is deleted, exactly as before."""
    worker = _make_layerwise_worker()
    tracker = worker._layerwise_session_tracker
    tracker.prepare_load_entries(REQ_ID, [("key-a", 0)])
    tracker.record_get_result("key-a", [REQ_ID], succeeded=True)

    meta = AscendConnectorMetadata({REQ_ID}, set())
    worker.get_finished(set(), meta)

    assert "key-a" not in tracker._load_key_owners
    assert REQ_ID not in tracker._request_load_entries


# ---------------------------------------------------------------------------
# Feature-on: the evidence half of the contract, on trees that carry it
# ---------------------------------------------------------------------------


def _terminal_attempt(evidence, generation: int):
    attempt = evidence.AscendStoreAttemptEvidence(
        req_id=REQ_ID,
        transfer_generation=generation,
        num_layers=4,
        block_size=BLOCK_SIZE,
        restored_extent=HIT_TOKENS,
        certification_mode="content",
    )
    for logical_block in range(8):
        attempt.note_block_outcome(logical_block, True)
    attempt.finalize()
    return attempt


def test_worker_keeps_readmitted_attempt_evidence_with_the_evidence_path(scheduler_with_lookup_hit):
    """On trees carrying the AscendStore evidence path
    (vllm-project/vllm-ascend#22651), get_finished drops buffered evidence
    exactly for finished | meta.preempted_req_ids. A request re-admitted in
    the step that reports its preemption has a live NEW attempt: dropping
    its evidence would fail the delivery closed to recomputation."""
    evidence = pytest.importorskip(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.evidence",
        reason="this tree carries no AscendStore evidence path",
    )
    scheduler = scheduler_with_lookup_hit
    request = _make_request()
    _admit_request(scheduler, request)
    _admit_request(scheduler, request)
    meta = scheduler.build_connector_meta(_make_scheduler_output([_make_new_request_data()], {REQ_ID}))
    assert REQ_ID not in meta.preempted_req_ids

    worker = _make_worker(None)
    new_attempt = _terminal_attempt(evidence, generation=3)
    worker.evidence_buffer.start_attempt(new_attempt)

    worker.get_finished(set(), meta)

    assert worker.evidence_buffer.get(REQ_ID) is new_attempt


def test_worker_drops_abandoned_request_evidence_with_the_evidence_path(scheduler_with_lookup_hit):
    """Control: the evidence of an abandoned (plainly preempted) request is
    dropped by the same get_finished."""
    evidence = pytest.importorskip(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.evidence",
        reason="this tree carries no AscendStore evidence path",
    )
    scheduler = scheduler_with_lookup_hit
    request = _make_request()
    _admit_request(scheduler, request)
    meta = scheduler.build_connector_meta(_make_scheduler_output([], {REQ_ID}))
    assert REQ_ID in meta.preempted_req_ids

    worker = _make_worker(None)
    worker.evidence_buffer.start_attempt(_terminal_attempt(evidence, generation=2))

    worker.get_finished(set(), meta)

    assert worker.evidence_buffer.get(REQ_ID) is None
