# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.request import RequestStatus

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.pd_pool.connector import LayerwisePoolConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.pd_pool.metadata import (
    POOL_PD_PROTOCOL,
    PoolPDMetadata,
    PoolPDRequest,
    prompt_digest,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.pd_pool.scheduler import PoolPDScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.pd_pool.worker import PoolPDWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata, ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_layout import RequestStateLayout

from .test_shared_io import MemoryBackend
from .test_shared_transfer import step, transfer


def test_pd_forwards_pool_checkpoint_completion_and_releases_pins():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

    scheduler = object.__new__(KVPoolScheduler)
    scheduler.use_hybrid = True
    scheduler.use_layerwise = False
    scheduler.num_speculative_blocks_by_group = {0: 1}
    scheduler.sending_event_id = 0
    scheduler.sending_events = {}
    scheduler.sending_blocks = {}
    scheduler._expected_worker_count = 8
    scheduler._loading_req_ids = set()
    blocks = [SimpleNamespace(ref_cnt=1) for _ in range(3)]

    def touch(items):
        for block in items:
            block.ref_cnt += 1

    def free(items):
        for block in items:
            block.ref_cnt -= 1

    scheduler._block_pool = SimpleNamespace(blocks=blocks, touch=touch, free_blocks=free)
    meta = SimpleNamespace(can_save=True, block_ids_by_group=[[1, 2]], event_id=None)
    scheduler.touch_sending_mamba_blocks(meta)
    assert blocks[1].ref_cnt == 2
    assert blocks[2].ref_cnt == 1  # speculative scratch must stay exclusive
    connector = object.__new__(LayerwisePoolConnector)
    connector.scheduler = SimpleNamespace(pool=scheduler)
    assert connector.has_pending_push_work()
    # Simulate request finish/cancellation releasing its own allocation while
    # the save still owns one reference. Only the eighth ACK may release it.
    free([blocks[1]])
    for rank in range(8):
        worker = object.__new__(KVPoolWorker)
        worker.shared_pool = SimpleNamespace(completed_events={meta.event_id: 1})
        connector.worker = SimpleNamespace(pool=worker)
        ack = connector.build_connector_worker_meta()
        assert connector.build_connector_worker_meta() is None
        connector.update_connector_output(SimpleNamespace(finished_recving=None, kv_connector_worker_meta=ack))
        assert blocks[1].ref_cnt == (0 if rank == 7 else 1)
    assert not connector.has_pending_push_work()
    assert scheduler.sending_blocks == {}


def request(tokens, **overrides):
    fields = dict(
        request_id="r",
        prompt_token_ids=tokens,
        all_token_ids=tokens,
        block_hashes=[b"h2", b"h4"],
        num_computed_tokens=0,
        kv_transfer_params={"do_remote_decode": True},
        cache_salt=None,
        mm_features=[],
        lora_request=None,
        prompt_embeds=None,
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_loaded_groups_distinguish_pd_from_prefix_after_allocation():
    scheduler = PoolPDScheduler(MagicMock())
    connector = LayerwisePoolConnector.__new__(LayerwisePoolConnector)
    connector.scheduler = scheduler
    connector.target_groups = (0, 1, 2)
    connector.prefix_groups = (0, 2)
    req = request([1, 2, 3, 4])
    scheduler.get_num_new_matched_tokens(req, 0)
    assert connector.get_loaded_kv_cache_group_ids(req) == (0, 2)
    req.kv_transfer_params = dict(do_remote_prefill=True, pool_pd_protocol=POOL_PD_PROTOCOL, remote_num_prompt_tokens=4)
    assert scheduler.get_num_new_matched_tokens(req, 0) == (3, True)
    assert connector.get_loaded_kv_cache_group_ids(req) == (0, 1, 2)
    # Allocation clears the wire flag, but async completion still needs the
    # original loaded-group contract. A later prefix lookup must reset it.
    req.kv_transfer_params["do_remote_prefill"] = False
    assert connector.get_loaded_kv_cache_group_ids(req) == (0, 1, 2)
    scheduler.get_num_new_matched_tokens(req, 0)
    assert connector.get_loaded_kv_cache_group_ids(req) == (0, 2)


@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize("length", [1, 3, 4])
def test_pd_tail_and_full_block_copy_and_retention(monkeypatch, layerwise, length):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=MagicMock), raising=False)
    backend = MemoryBackend()
    for rank in range(2):
        common, caches = transfer(backend, 2, rank, layerwise, True)
        producer = PoolPDWorker(SimpleNamespace(shared_pool=common))
        tail = PoolPDRequest("p", length, "prompt", [[1]], [b"h2", b"h4"])
        full = ReqMeta(
            "p", block_ids_by_group=[[1]], block_hashes=tail.block_hashes, can_save=True, save_end_token=length // 4 * 4
        )
        metadata = PoolPDMetadata(step(full), tails=[tail])
        producer.prepare(metadata)
        producer.start_load(metadata)
        if layerwise:
            for name in caches:
                producer.save_layer(name)
        producer.finish()
    retained = set(backend.objects)
    for rank in range(4):
        common, actual = transfer(backend, 4, rank, False, False)
        _, expected = transfer(MemoryBackend(), 4, rank, False, True)
        decoder = PoolPDWorker(SimpleNamespace(shared_pool=common))
        metadata = PoolPDMetadata(step(), receives=[PoolPDRequest("d", length, "prompt", [[2]], [b"h2", b"h4"])])
        decoder.prepare(metadata)
        decoder.start_load(metadata)
        decoder.finish()
        result = decoder.take_results()
        assert result.finished_recving == {"d"} and not result.failed_recving
        for name in actual:
            for output, wanted in zip(actual[name], expected[name], strict=True):
                flat = output[4:6].permute(1, 0, 2, 3).reshape(output.shape[1], 4, -1)
                ref = wanted[2:4].permute(1, 0, 2, 3).reshape(output.shape[1], 4, -1)
                loaded = length if length > 1 else 0
                torch.testing.assert_close(flat[:, :loaded], ref[:, :loaded], rtol=0, atol=0)
                assert torch.all(flat[:, loaded:] == -99)
        # Ordinary lookup can discover complete PD objects, never the tail.
        assert common.lookup(4, [b"h2", b"h4"]) == (4 if length == 4 else 0)
    assert set(backend.objects) == retained
    assert not backend.leases


def test_pd_adopts_per_group_attention_hits_and_restores_state():
    connector = LayerwisePoolConnector.__new__(LayerwisePoolConnector)
    config = SimpleNamespace(kv_role="kv_consumer", kv_connector_extra_config={})
    connector._kv_transfer_config = config
    assert connector.supports_divergent_local_hybrid_hits
    config.kv_connector_extra_config["consumer_is_to_load"] = True
    assert not connector.supports_divergent_local_hybrid_hits
    config.kv_role = "kv_producer"
    config.kv_connector_extra_config.clear()
    assert not connector.supports_divergent_local_hybrid_hits

    scheduler = PoolPDScheduler(MagicMock(), need_truncate=True)
    req = request(
        list(range(11)),
        block_hashes=[b"h2", b"h4", b"h6", b"h8"],
        kv_transfer_params=dict(do_remote_prefill=True, pool_pd_protocol=POOL_PD_PROTOCOL, remote_num_prompt_tokens=11),
    )
    assert scheduler.get_num_new_matched_tokens(req, 8) == (2, True)
    blocks = SimpleNamespace(
        get_block_ids=lambda: ([1, 2, 3], [4, 5, 6], [0, 0, 7], [8]),
        get_unhashed_block_ids_all_groups=lambda: [[3], [5, 6], [7], [8]],
    )
    scheduler.update_state_after_alloc(req, blocks, 2)

    def layout(group, *, state=False):
        return SimpleNamespace(
            chunk_size=4,
            block_size=4,
            hash_block_size=2,
            state=state,
            objects=lambda start, block, key: [(group, start, block)],
            partial_objects=lambda start, block, key, remainder: [(group, start, block, remainder)],
        )

    transfer = SimpleNamespace(
        layouts={0: layout(0), 1: layout(1), 2: layout(2, state=True)},
        request_states={3: SimpleNamespace(snapshot=lambda block, digest, end: [(3, end, block)])},
    )
    worker = PoolPDWorker(SimpleNamespace(shared_pool=transfer))
    receive = scheduler.receives["r"]
    # The second attention group missed a block inside the first group's
    # local hit. Restore it, both tails and the exact recurrent/ring state.
    assert worker._plans(receive, tail_only=False, writing=False) == (
        (3, 10, 8),
        (0, 8, 3, 2),
        (1, 4, 5),
        (1, 8, 6, 2),
        (2, 0, 7),
    )


def test_pd_receive_does_not_require_recomputed_token_block(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=MagicMock), raising=False)
    backend = MemoryBackend()
    # A five-token prompt needs only its first four tokens remotely. The
    # scheduler allocates one block, and the PD-only one-token tail may be absent.
    for rank in range(2):
        common, _ = transfer(backend, 2, rank, False, True)
        producer = PoolPDWorker(SimpleNamespace(shared_pool=common))
        full = ReqMeta("p", block_ids_by_group=[[1]], block_hashes=[b"h2", b"h4"], can_save=True, save_end_token=4)
        metadata = PoolPDMetadata(step(full))
        producer.prepare(metadata)
        producer.start_load(metadata)
        producer.finish()
    for rank in range(4):
        common, actual = transfer(backend, 4, rank, False, False)
        _, expected = transfer(MemoryBackend(), 4, rank, False, True)
        decoder = PoolPDWorker(SimpleNamespace(shared_pool=common))
        metadata = PoolPDMetadata(step(), receives=[PoolPDRequest("d", 5, "no-tail", [[2]], [b"h2", b"h4"])])
        decoder.prepare(metadata)
        decoder.start_load(metadata)
        decoder.finish()
        result = decoder.take_results()
        assert result.finished_recving == {"d"} and not result.failed_recving
        for name in actual:
            for output, wanted in zip(actual[name], expected[name], strict=True):
                torch.testing.assert_close(output[4:6], wanted[2:4], rtol=0, atol=0)
    assert not backend.leases


def test_pd_miss_reports_failed_receive_and_never_holds_blocks(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=MagicMock), raising=False)
    backend = MemoryBackend()
    common, _ = transfer(backend, 4, 0, False, False)
    worker = PoolPDWorker(SimpleNamespace(shared_pool=common))
    metadata = PoolPDMetadata(step(), receives=[PoolPDRequest("d", 3, "missing", [[2]], [])])
    worker.prepare(metadata)
    worker.start_load(metadata)
    worker.finish()
    result = worker.take_results()
    assert result.finished_recving == result.failed_recving == {"d"}
    connector = object.__new__(LayerwisePoolConnector)
    connector.worker = worker
    assert connector.get_block_ids_with_load_errors() == set()
    assert not backend.leases


def test_partial_pd_copy_failure_invalidates_destination_and_preserves_pool(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=MagicMock), raising=False)
    backend = MemoryBackend()
    common, _ = transfer(backend, 2, 0, False, True)
    producer = PoolPDWorker(SimpleNamespace(shared_pool=common))
    source = PoolPDRequest("p", 3, "prompt", [[1]], [])
    metadata = PoolPDMetadata(step(), tails=[source])
    producer.prepare(metadata)
    producer.start_load(metadata)
    producer.finish()
    published = set(backend.readable)
    common, caches = transfer(backend, 2, 0, False, False)
    decoder = PoolPDWorker(SimpleNamespace(shared_pool=common))
    destination = PoolPDRequest("d", 3, "prompt", [[2]], [])
    metadata = PoolPDMetadata(step(), receives=[destination])
    copy = backend.batch_copy

    def partial_failure(gvas, addresses, sizes, direction):
        assert direction == 1
        assert copy(gvas[:1], addresses[:1], sizes[:1], direction) == 0
        return -1

    monkeypatch.setattr(backend, "batch_copy", partial_failure)
    decoder.prepare(metadata)
    decoder.start_load(metadata)
    decoder.finish()
    result = decoder.take_results()
    assert result.finished_recving == result.failed_recving == {"d"}
    connector = object.__new__(LayerwisePoolConnector)
    connector.worker = decoder
    assert connector.get_block_ids_with_load_errors() == set()
    assert any(torch.any(cache != -99) for parts in caches.values() for cache in parts)
    assert backend.readable == published and not backend.leases


def test_digest_includes_prefix_salt_and_lora():
    base = request([1, 2, 3])
    reference = prompt_digest(base)
    assert prompt_digest(request([9, 2, 3])) != reference
    assert prompt_digest(request([1, 2, 3], cache_salt="tenant")) != reference
    assert prompt_digest(request([1, 2, 3], lora_request=SimpleNamespace(lora_name="a", lora_path="p"))) != reference
    assert prompt_digest(request([1, 2, 3], request_id="another")) == reference


def test_scheduler_cpd_contract_and_last_token_recompute():
    pool = MagicMock()
    pool._request_trackers = {"r": SimpleNamespace(allocated_block_ids_by_group=[[1]])}
    pool.build_connector_meta.return_value = step()
    scheduler = PoolPDScheduler(pool)
    req = request([1, 2, 3])
    scheduler.requests["r"] = req
    output = SimpleNamespace(finished_req_ids=set(), preempted_req_ids=set(), num_scheduled_tokens={"r": 2})
    assert not scheduler.build_connector_meta(output).tails
    req.num_computed_tokens = 2
    output.num_scheduled_tokens["r"] = 1
    assert len(scheduler.build_connector_meta(output).tails) == 1
    delay, params = scheduler.request_finished(req, [[1]])
    assert not delay
    assert params == dict(
        do_remote_prefill=True, do_remote_decode=False, pool_pd_protocol=POOL_PD_PROTOCOL, remote_num_prompt_tokens=3
    )
    decoder = PoolPDScheduler(MagicMock())
    req.kv_transfer_params = params
    assert decoder.get_num_new_matched_tokens(req, 0) == (2, True)
    decoder.update_state_after_alloc(
        req, SimpleNamespace(get_block_ids=lambda: ([2],), get_unhashed_block_ids_all_groups=lambda: [[2]]), 2
    )
    assert decoder.receives["r"].prompt_digest == prompt_digest(req)
    assert not req.kv_transfer_params["do_remote_prefill"]
    decoder.pool.update_state_after_alloc.assert_called_once()
    assert decoder.pool.update_state_after_alloc.call_args.args[-1] == 0


def test_scheduler_rejects_unpublished_or_different_handoff():
    scheduler = PoolPDScheduler(MagicMock())
    req = request([1, 2, 3])
    with pytest.raises(RuntimeError, match="publication"):
        scheduler.request_finished(req, [[1]])
    req.kv_transfer_params = {"do_remote_prefill": True}
    with pytest.raises(ValueError, match="compatible"):
        scheduler.get_num_new_matched_tokens(req, 0)
    req.kv_transfer_params.update(pool_pd_protocol=POOL_PD_PROTOCOL, remote_num_prompt_tokens=4)
    with pytest.raises(ValueError, match="length"):
        scheduler.get_num_new_matched_tokens(req, 0)


def test_preempted_producer_republishes_from_new_blocks_without_retruncating():
    pool = MagicMock()
    pool.build_connector_meta.return_value = step()
    pool._request_trackers = {"r": SimpleNamespace(allocated_block_ids_by_group=[[1]])}
    scheduler = PoolPDScheduler(pool, need_truncate=True)
    req = request(
        [1, 2, 3], num_prompt_tokens=3, _all_token_ids=[1, 2, 3], update_block_hashes=MagicMock(), max_tokens=1
    )
    digest = prompt_digest(req)
    scheduler.on_new_request(req)
    scheduler.get_num_new_matched_tokens(req, 0)
    output = SimpleNamespace(finished_req_ids=set(), preempted_req_ids=set(), num_scheduled_tokens={"r": 2})
    first = scheduler.build_connector_meta(output).tails[0]
    assert first.block_ids == [[1]]
    output.preempted_req_ids = {"r"}
    output.num_scheduled_tokens = {}
    assert not scheduler.build_connector_meta(output).tails
    with pytest.raises(RuntimeError, match="publication"):
        scheduler.request_finished(req, [[1]])
    scheduler.on_new_request(req)
    assert req.prompt_token_ids == [1, 2]
    assert req.update_block_hashes.call_count == 1
    pool._request_trackers["r"].allocated_block_ids_by_group = [[5]]
    output.preempted_req_ids = set()
    output.num_scheduled_tokens = {"r": 2}
    retry = scheduler.build_connector_meta(output).tails[0]
    assert retry.block_ids == [[5]]
    assert retry.prompt_digest == first.prompt_digest == digest
    assert retry.cache_tokens == first.cache_tokens == 2


def test_finished_decode_clears_pending_receive_and_request_identity():
    pool = MagicMock()
    pool.build_connector_meta.return_value = step()
    scheduler = PoolPDScheduler(pool, need_truncate=True)
    req = request(
        [1, 2, 3],
        kv_transfer_params=dict(do_remote_prefill=True, pool_pd_protocol=POOL_PD_PROTOCOL, remote_num_prompt_tokens=3),
    )
    assert scheduler.get_num_new_matched_tokens(req, 0) == (2, True)
    scheduler.update_state_after_alloc(
        req, SimpleNamespace(get_block_ids=lambda: ([4],), get_unhashed_block_ids_all_groups=lambda: [[4]]), 2
    )
    assert "r" in scheduler.receives
    output = SimpleNamespace(finished_req_ids={"r"}, preempted_req_ids=set(), num_scheduled_tokens={})
    metadata = scheduler.build_connector_meta(output)
    assert not metadata.receives
    assert not scheduler.requests and not scheduler.receives and not scheduler.matches
    assert not scheduler.pd_load_requests and not scheduler.original_prompts and not scheduler.tail_scheduled


@pytest.mark.parametrize("length", [1, 4, 5, 8, 9])
def test_stateful_prefill_preserves_original_identity_and_is_idempotent(length):
    tokens = list(range(length))
    producer = PoolPDScheduler(MagicMock(), need_truncate=True)
    req = request(
        tokens.copy(),
        num_prompt_tokens=length,
        _all_token_ids=tokens.copy(),
        update_block_hashes=MagicMock(),
        max_tokens=32,
    )
    digest = prompt_digest(req)
    producer.on_new_request(req)
    producer.on_new_request(req)
    computed = max(1, length - 1)
    assert len(req.prompt_token_ids) == req.num_prompt_tokens == len(req._all_token_ids) == computed
    assert req.update_block_hashes.call_count == int(length > 1)
    record = producer._request(req, [[1]])
    assert (record.prompt_tokens, record.prompt_digest, record.cache_tokens) == (length, digest, computed)
    producer.tail_scheduled.add(req.request_id)
    _, params = producer.request_finished(req, [[1]])
    assert params["remote_num_prompt_tokens"] == length
    decoder = PoolPDScheduler(MagicMock(), need_truncate=True)
    incoming = request(tokens.copy(), kv_transfer_params=params)
    assert decoder.get_num_new_matched_tokens(incoming, 0) == (length - 1, length > 1)
    restored = decoder._request(incoming, [[2]])
    assert restored.prompt_digest == record.prompt_digest
    assert restored.cache_tokens == record.cache_tokens


def test_stateful_attention_transfer_ends_before_recomputed_token():
    common, _ = transfer(MemoryBackend(), 2, 0, False, False)
    worker = PoolPDWorker(SimpleNamespace(shared_pool=common))
    handoff = PoolPDRequest("d", 5, "original", [[2]], [b"h2", b"h4"], cache_tokens=4)
    assert worker._plans(handoff, tail_only=True, writing=True) == ()
    plans = worker._plans(handoff, tail_only=False, writing=False)
    assert plans
    assert all("partial_" not in plan.key for plan in plans)


@pytest.mark.parametrize("layerwise", [False, True])
def test_pd_circular_state_copies_payload_without_padding(monkeypatch, layerwise):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=MagicMock), raising=False)
    backend = MemoryBackend()
    tensors = []
    keys = []
    for writing in (True, False):
        common, _ = transfer(backend, 2, 0, layerwise, writing)
        owner = "model.layers.0.indexer.tail_cache"
        spec = AscendIndexerKPoolTailSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=3,
            dtype=torch.float32,
            sliding_window=4,
            compress_ratio=4,
        )
        backing = torch.full((3, 40), -9.0)
        cache = backing.as_strided((3, 4, 6), (40, 6, 1))
        if writing:
            cache[1] = torch.arange(24).reshape(4, 6)
        layout = RequestStateLayout(KeyMetadata("model", 0, 0, 0), {owner: spec}, {owner: (cache,)}, 3, "weights", 0)
        common.layouts = {}
        common.request_states = {0: layout}
        worker = PoolPDWorker(SimpleNamespace(shared_pool=common))
        req = PoolPDRequest("r", 7, "prompt", [[1 if writing else 2]], [], cache_tokens=6)
        keys.append(worker._plans(req, tail_only=writing, writing=writing)[0].key)
        meta = PoolPDMetadata(step(), tails=[req] if writing else [], receives=[] if writing else [req])
        worker.prepare(meta)
        worker.start_load(meta)
        if writing and layerwise:
            worker.save_layer(owner)
        worker.finish()
        assert not worker.take_results().failed_recving
        assert common.lookup(8, [b"a", b"b"]) == 0
        assert torch.all(backing[:, 24:] == -9)
        tensors.append(cache)
    assert keys[0] == keys[1]
    torch.testing.assert_close(tensors[0][1], tensors[1][2], rtol=0, atol=0)


def test_pd_layerwise_restores_checkpoint_before_runner_state_copy():
    connector = object.__new__(LayerwisePoolConnector)
    connector.requires_mamba_state_copy_after_layer_load = True
    connector._mamba_state = None
    calls = []
    connector.worker = SimpleNamespace(
        transfer=SimpleNamespace(layerwise=True, load_layer=lambda name: calls.append(("load", name)))
    )
    state = SimpleNamespace(do_mamba_copy_for_layer=lambda name: calls.append(("copy", name)))
    assert connector.prepare_mamba_state_copy(state)
    connector.wait_for_layer_load("kda")
    assert calls == [("load", "kda"), ("copy", "kda")]
    connector.finish_mamba_state_copy()
    connector.wait_for_layer_load("other")
    assert calls[-1] == ("load", "other")
    with pytest.raises(TypeError, match="MRV2"):
        connector.prepare_mamba_state_copy(object())
    connector.requires_mamba_state_copy_after_layer_load = False
    assert not connector.prepare_mamba_state_copy(state)


@pytest.mark.parametrize("producer_tokens,decoder_tokens", [(0, 0), (1, 1), (3, 3), (0, 1), (1, 0), (1, 3)])
def test_pd_checks_mtp_cache_contract_before_allocating(producer_tokens, decoder_tokens):
    producer = PoolPDScheduler(MagicMock(), need_truncate=True, mtp_tokens=producer_tokens)
    req = request([1, 2, 3])
    producer.tail_scheduled.add(req.request_id)
    _, params = producer.request_finished(req, [[1]])
    decoder = PoolPDScheduler(MagicMock(), need_truncate=True, mtp_tokens=decoder_tokens)
    req.kv_transfer_params = params
    if producer_tokens == decoder_tokens:
        assert decoder.get_num_new_matched_tokens(req, 0) == (2, True)
    else:
        with pytest.raises(ValueError, match="matching P/D"):
            decoder.get_num_new_matched_tokens(req, 0)
        assert not decoder.matches and not decoder.receives


@pytest.mark.parametrize("length", [3, 4, 5])
def test_protected_handoff_holds_full_blocks_and_tail_until_deadline(monkeypatch, length):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=MagicMock), raising=False)
    backend = MemoryBackend()
    common, _ = transfer(backend, 2, 0, False, True)
    worker = PoolPDWorker(SimpleNamespace(shared_pool=common))
    req = PoolPDRequest("p", length, "prompt", [[1, 2]], [b"h2", b"h4"], handoff_deadline=time.time() + 60)
    full = ReqMeta(
        "p", block_ids_by_group=[[1, 2]], block_hashes=req.block_hashes, can_save=True, save_end_token=(length // 4) * 4
    )
    meta = PoolPDMetadata(step(full), tails=[req])
    worker.prepare(meta)
    worker.start_load(meta)
    worker.finish()
    assert worker.handoff_holds
    keys = {p.key for p in worker._plans(req, tail_only=False, writing=True)}
    assert keys and keys <= backend.leases
    worker._reap_handoffs()
    assert keys <= backend.leases
    worker._reap_handoffs(all_holds=True)
    assert not backend.leases and not worker.handoff_holds


def test_expired_handoff_never_copies(monkeypatch):
    backend = MemoryBackend()
    common, _ = transfer(backend, 2, 0, False, False)
    worker = PoolPDWorker(SimpleNamespace(shared_pool=common))
    req = PoolPDRequest("d", 3, "prompt", [[2]], [], handoff_deadline=time.time() - 1)
    monkeypatch.setattr(backend, "batch_copy", lambda *a: pytest.fail("expired handoff copied"))
    worker.start_load(PoolPDMetadata(step(), receives=[req]))
    assert worker.take_results().failed_recving == {"d"}


def test_scheduler_protected_handoff_requires_fresh_deadline():
    scheduler = PoolPDScheduler(MagicMock(), handoff_ttl_s=60)
    req = request([1, 2, 3])
    req.kv_transfer_params = dict(do_remote_prefill=True, pool_pd_protocol=POOL_PD_PROTOCOL, remote_num_prompt_tokens=3)
    for deadline in [None, float("nan"), time.time() - 1]:
        req.kv_transfer_params["pool_pd_handoff_deadline"] = deadline
        with pytest.raises(ValueError, match="protection expired"):
            scheduler.get_num_new_matched_tokens(req, 0)
    req.kv_transfer_params["pool_pd_handoff_deadline"] = time.time() + 60
    assert scheduler.get_num_new_matched_tokens(req, 0) == (2, True)


def test_handoff_acquisition_failure_cannot_emit_protection(monkeypatch):
    backend = MemoryBackend()
    common, _ = transfer(backend, 2, 0, False, True)
    worker = PoolPDWorker(SimpleNamespace(shared_pool=common))
    worker.handoff_requests = [PoolPDRequest("missing", 3, "prompt", [[1]], [], handoff_deadline=time.time() + 60)]
    with pytest.raises(RuntimeError, match="read lease failed"):
        worker._protect_handoffs()
    assert not worker.handoff_holds
