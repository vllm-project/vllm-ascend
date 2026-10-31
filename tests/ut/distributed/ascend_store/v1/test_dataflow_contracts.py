"""Validate request planning, failure handoff, publication and framework result contracts."""

from dataclasses import replace
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorRole

from tests.ut.distributed.ascend_store.v1.helpers import make_topology, make_worker
from tests.ut.distributed.ascend_store.v1.scheduler.fixtures import (
    FakeBlockPool,
    FakeBlocks,
    FakeLookupClient,
    make_config,
    make_output,
    make_request,
    new_request_data,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector import AscendStoreV1Connector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration import (
    resolve_kv_pool_config,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import LookupClient, LookupServer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    RangeStoreCommand,
    StoreCommandBatch,
    StoreSourceReleaseMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.observations import StoreEvents
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler import (
    AsynchronousScheduler,
    SynchronousScheduler,
)
from vllm_ascend.worker.v2.kv_connector import MambaCopyKVConnector


def bind_connector(monkeypatch, worker):
    config = worker._topology.config
    monkeypatch.setattr(KVConnectorBase_V1, "__init__", lambda *args, **kwargs: None)
    monkeypatch.setattr(connector_module, "resolve_kv_pool_config", lambda *_: config)
    monkeypatch.setattr(connector_module, "validate_kv_pool_config", lambda *_: None)
    monkeypatch.setattr(AscendStoreV1Connector, "_create_worker", lambda *_: worker)
    return AscendStoreV1Connector(
        SimpleNamespace(parallel_config=SimpleNamespace(rank=1)), KVConnectorRole.WORKER, None
    )


@pytest.mark.parametrize("scheduler_type", (SynchronousScheduler, AsynchronousScheduler))
def test_repeated_lookup_replaces_candidate_even_when_next_query_returns_early(scheduler_type):
    lookup = FakeLookupClient(4)
    scheduler = scheduler_type(make_config(store=True), lookup)
    request = make_request(prompt_tokens=8, tokens=9)
    try:
        assert scheduler.get_num_new_matched_tokens(request, 0)[0] == 4
        assert scheduler.get_num_new_matched_tokens(request, 8) == (0, False)
        scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3]), 0)
        step = scheduler.build_step(
            make_output(new=(new_request_data("request", ([1, 2, 3],), 8),), scheduled_tokens={"request": 1})
        )
        assert not step.load_commands
        assert [(command.start_token, command.end_token) for command in step.store.commands] == [(0, 8)]
    finally:
        scheduler.close()


@pytest.mark.parametrize("entry", ("new", "resumed"))
def test_recovery_entrypoints_preserve_the_same_recomputation_and_store_range(entry):
    scheduler = SynchronousScheduler(make_config(load=False, store=True), FakeLookupClient())
    request = make_request(prompt_tokens=4, tokens=12, hashes=3)
    try:
        scheduler.confirm_allocation(request, FakeBlocks([4, 5, 6]), 0)
        if entry == "new":
            data = new_request_data("request", ([4, 5, 6],), 4)
            data.prefill_token_ids = list(range(12))
            first = make_output(new=(data,), scheduled_tokens={"request": 4})
        else:
            first = make_output(
                cached=(("request", ([4, 5, 6],), 4),), resumed=("request",), scheduled_tokens={"request": 4}
            )
        restored = scheduler.build_step(first)
        tail = scheduler.build_step(make_output(cached=(("request", None, 8),), scheduled_tokens={"request": 4}))
        assert [
            (command.start_token, command.end_token) for command in restored.store.commands + tail.store.commands
        ] == [(0, 8), (8, 12)]
    finally:
        scheduler.close()


def test_consumer_decode_policy_is_applied_before_worker_command_preparation(configuration_case):
    config, cache = configuration_case
    config.kv_transfer_config.kv_role = "kv_consumer"
    config.kv_transfer_config.kv_connector_extra_config["save_decode_cache"] = True
    parsed = resolve_kv_pool_config(config, cache)
    assert parsed.store_enabled and not parsed.store_prefill_enabled
    scheduler = SynchronousScheduler(parsed, FakeLookupClient())
    request = make_request(prompt_tokens=4)
    try:
        scheduler.confirm_allocation(request, FakeBlocks([1]), 0)
        prefill = scheduler.build_step(
            make_output(new=(new_request_data("request", ([1],), 0),), scheduled_tokens={"request": 4})
        )
        assert prefill.store.commands == ()
        request.num_tokens = 8
        request.block_hashes.append(b"decode")
        decode = scheduler.build_step(make_output(cached=(("request", ([2],), 4),), scheduled_tokens={"request": 4}))
        assert [(command.start_token, command.end_token) for command in decode.store.commands] == [(4, 8)]
    finally:
        scheduler.close()


@pytest.mark.parametrize("groups", ((0, 1), (1,)))
@pytest.mark.parametrize("finished", (frozenset(), frozenset({"r"})))
def test_async_grouped_failures_reach_framework_snapshot_with_request_ownership(monkeypatch, groups, finished):
    worker, resources, backend = make_worker(topology=make_topology(group_ids=groups), async_load=True)
    connector = bind_connector(monkeypatch, worker)
    backend.get_result = [-7] * len(groups)
    try:
        connector.bind_connector_metadata(KVTransferStep((LoadCommand("r", 0, 4, ((1,), (2,)), (b"a",)),)))
        connector.start_load_kv(None)
        worker.transfer.executor._write.submit(lambda: None).result(timeout=2)
        result = connector.get_transfer_results(set(finished))
        assert result.finished_recving == result.failed_recving == {"r"}
        assert result.finished_sending == set()
        assert connector.get_block_ids_with_load_errors() == set()
        connector.clear_connector_metadata()
        connector.bind_connector_metadata(KVTransferStep())
        connector.start_load_kv(None)
        assert not connector.get_transfer_results(set()).finished_recving
        assert not connector.get_block_ids_with_load_errors()
        connector.clear_connector_metadata()
    finally:
        connector.shutdown()
    assert resources.closed


@pytest.mark.parametrize("partial", (False, True))
def test_failed_load_invalidates_only_its_store_sources_and_releases_every_job(partial):
    worker, resources, backend = make_worker(async_load=False)
    backend.get_result = [-7]
    stores = [RangeStoreCommand("bad", 4, 8, ((1, 2),), (b"a", b"b"), 8, 17)]
    if partial:
        stores.append(RangeStoreCommand("good", 0, 4, ((3,),), (b"c",), 4, 18))
    try:
        worker.begin_step(
            KVTransferStep((LoadCommand("bad", 0, 4, ((1, 2),), (b"a", b"b")),), StoreCommandBatch(tuple(stores)))
        )
        worker.start_load()
        worker.finish_step()
        results = worker.fence_previous_store()
        assert [result.succeeded for result in results] == ([False, True] if partial else [False])
        keys = [key for call in backend.calls if call[0] == "put" for key in call[1]]
        assert all(key.endswith("@63") for key in keys)
        assert bool(keys) is partial
        assert worker.take_released_store_job_ids() == ({17, 18} if partial else {17})
        assert worker.collect_load_result().failed_request_ids == {"bad"}
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


def test_source_invalidation_does_not_survive_into_valid_recomputation():
    worker, _, backend = make_worker(async_load=False)
    try:
        backend.get_result = [-7]
        command = RangeStoreCommand("r", 4, 8, ((1, 2),), (b"a", b"b"), 8, 17)
        step = KVTransferStep((LoadCommand("r", 0, 4, ((1, 2),), (b"a", b"b")),), StoreCommandBatch((command,)))
        worker.begin_step(step)
        worker.start_load()
        worker.finish_step()
        assert not worker.fence_previous_store()[0].succeeded
        assert worker.take_released_store_job_ids() == {17}
        assert worker.collect_load_result().failed_request_ids == {"r"}
        worker.end_step()
        command = RangeStoreCommand("r", 0, 4, ((1,),), (b"a",), 4, 18)
        worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
        worker.finish_step()
        assert worker.fence_previous_store()[0].succeeded
        assert worker.take_released_store_job_ids() == {18}
        assert any(call[0] == "put" for call in backend.calls)
        worker.end_step()
    finally:
        worker.close()


def test_existing_store_objects_do_not_require_a_valid_local_source():
    worker, _, backend = make_worker(async_load=False, requires_exists_before_put=True)
    backend.get_result = [-7]
    backend.presence = [1, 0]
    command = RangeStoreCommand("r", 4, 8, ((1, 2),), (b"a", b"b"), 8, 17)
    healthy = RangeStoreCommand("healthy", 0, 4, ((3,),), (b"c",), 4, 18)
    try:
        worker.begin_step(
            KVTransferStep((LoadCommand("r", 0, 4, ((1, 2),), (b"a", b"b")),), StoreCommandBatch((command, healthy)))
        )
        worker.start_load()
        worker.finish_step()
        results = worker.fence_previous_store()
        assert all(result.succeeded and result.source_release_confirmed and result.error is None for result in results)
        keys = [key for call in backend.calls if call[0] == "put" for key in call[1]]
        assert len(keys) == 1 and keys[0].endswith("@63")
        assert worker.take_released_store_job_ids() == {17, 18}
        worker.collect_load_result()
        worker.end_step()
    finally:
        worker.close()


@pytest.mark.parametrize(
    "load_async,layerwise,deferred", ((False, False, True), (True, False, False), (True, True, False))
)
def test_real_connector_exposes_model_state_consumption_order(monkeypatch, load_async, layerwise, deferred):
    topology = make_topology()
    topology = replace(topology, config=replace(topology.config, use_layerwise=layerwise))
    worker, _, _ = make_worker(topology=topology, async_load=load_async, layerwise=layerwise)
    connector = bind_connector(monkeypatch, worker)
    try:
        assert connector.requires_mamba_state_copy_after_load is deferred
    finally:
        connector.shutdown()


def test_cache_reset_crosses_rpc_without_closing_admission_and_discards_old_lookup():
    worker, resources, backend = make_worker(async_load=False)
    backend.presence = [1]

    def reset():
        backend.presence = [0]
        return True

    backend.reset_cache = Mock(side_effect=reset)
    directory = TemporaryDirectory(prefix="kvreset-", dir="/tmp")
    address = f"ipc://{directory.name}/reset.sock"
    server = LookupServer(worker.lookup, address, reset=worker.reset_cache)
    scheduler = SynchronousScheduler(worker._topology.config, LookupClient(address, timeout_ms=2000))
    request = make_request(prompt_tokens=4, tokens=5, hashes=1)
    try:
        assert scheduler.get_num_new_matched_tokens(request, 0) == (4, False)
        assert scheduler.reset_cache() is True
        backend.reset_cache.assert_called_once()
        assert not scheduler._pending_lookups
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        store = RangeStoreCommand("after-reset", 0, 4, ((1,),), (b"a",), 4, 17, event_tokens=(0, (1, 2, 3, 4)))
        worker.begin_step(KVTransferStep(store=StoreCommandBatch((store,))))
        assert not worker.reset_cache()
        worker.finish_step()
        assert worker.fence_previous_store()[0].succeeded
        worker.end_step()
        assert scheduler.reset_cache() is True
        assert worker.take_events() == []
        assert not resources.closed
    finally:
        scheduler.close()
        server.close()
        worker.close()
        directory.cleanup()


@pytest.mark.parametrize("backend,layerwise,expected", (("memcache", False, None), ("mooncake", True, False)))
def test_cache_reset_reports_unsupported_routes_without_claiming_success(monkeypatch, backend, layerwise, expected):
    lookup = FakeLookupClient()
    monkeypatch.setattr(
        lookup, "reset", Mock(side_effect=AssertionError("Unsupported reset must not reach Backend")), raising=False
    )
    scheduler = SynchronousScheduler(replace(make_config(), backend_name=backend, use_layerwise=layerwise), lookup)
    try:
        assert scheduler.reset_cache() is expected
    finally:
        scheduler.close()


def test_uncertain_cache_reset_invalidates_candidates_without_fabricating_success(monkeypatch):
    lookup = FakeLookupClient(4)
    monkeypatch.setattr(lookup, "reset", lambda: False, raising=False)
    scheduler = SynchronousScheduler(make_config(), lookup)
    connector = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    connector.scheduler, connector.worker = scheduler, None
    connector._kv_events = StoreEvents([])
    try:
        assert scheduler.get_num_new_matched_tokens(make_request(), 0)[0] == 4
        assert connector.reset_cache() is False
        assert not scheduler._pending_lookups
        assert connector._kv_events is None
    finally:
        scheduler.close()


def test_cache_reset_waits_for_source_release_and_metrics_count_requests_not_jobs(monkeypatch):
    lookup = FakeLookupClient()
    reset = Mock(return_value=True)
    monkeypatch.setattr(lookup, "reset", reset, raising=False)
    scheduler = SynchronousScheduler(make_config(load=False, store=True, recycling=True), lookup)
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(prompt_tokens=8)
    try:
        scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 0)
        first = scheduler.build_step(
            make_output(new=(new_request_data("request", ([1, 2],), 0),), scheduled_tokens={"request": 4})
        )
        second = scheduler.build_step(make_output(cached=(("request", None, 4),), scheduled_tokens={"request": 4}))
        stats = scheduler.get_stats().reduce()
        assert stats["ascend_store_delayed_release_requests"] == 1
        assert stats["ascend_store_delayed_release_blocks"] == 2
        assert scheduler.reset_cache() is False
        reset.assert_not_called()
        for command in first.store.commands + second.store.commands:
            scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 1}))
        assert scheduler.reset_cache() is True
        assert not scheduler.has_pending_push_work()
    finally:
        scheduler.close()


@pytest.mark.parametrize("write_failure", (False, True))
def test_store_events_follow_publication_and_preserve_present_object_facts(monkeypatch, write_failure):
    worker, resources, backend = make_worker()
    scheduler = SynchronousScheduler(
        replace(worker._topology.config, enable_kv_events=True, save_decode_cache=True), FakeLookupClient()
    )
    connector = bind_connector(monkeypatch, worker)
    scheduler_connector = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    scheduler_connector.worker, scheduler_connector.scheduler = None, scheduler
    scheduler_connector._kv_events = None
    worker.store._requires_exists = True
    backend.presence = [1, 0]
    backend.put_result = [-7] if write_failure else [0]
    request = make_request(prompt_tokens=8)
    request.all_token_ids = list(range(8))
    try:
        scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 0)
        step = scheduler.build_step(
            make_output(new=(new_request_data("request", ([1, 2],), 0),), scheduled_tokens={"request": 8})
        )
        connector.bind_connector_metadata(step)
        connector.start_load_kv(None)
        connector.wait_for_save()
        connector.handle_preemptions(KVTransferStep(released_request_ids=frozenset({"request"})))
        events = connector.get_kv_connector_kv_cache_events()
        assert events is not None
        scheduler_connector.update_connector_output(
            SimpleNamespace(kv_cache_events=events, kv_connector_worker_meta=None)
        )
        observed = list(scheduler_connector.take_events())
        assert [event.token_ids for event in observed] == (
            [list(range(4))] if write_failure else [list(range(4)), list(range(4, 8))]
        )
        assert list(scheduler_connector.take_events()) == []
        assert connector.get_kv_connector_kv_cache_events() is None
        connector.clear_connector_metadata()
        request.num_tokens = 12
        request.all_token_ids.extend(range(8, 12))
        request.block_hashes.append(b"decode")
        backend.presence = [0]
        tail = scheduler.build_step(make_output(cached=(("request", ([3],), 8),), scheduled_tokens={"request": 4}))
        assert tail.store.commands[0].event_tokens == (8, (8, 9, 10, 11))
        connector.bind_connector_metadata(tail)
        connector.start_load_kv(None)
        connector.wait_for_save()
        connector.handle_preemptions(KVTransferStep(released_request_ids=frozenset({"request"})))
        events = connector.get_kv_connector_kv_cache_events()
        if write_failure:
            assert events is None
        else:
            assert events is not None and events.get_all_events()[0].token_ids == [8, 9, 10, 11]
        connector.clear_connector_metadata()
    finally:
        scheduler.close()
        connector.shutdown()
    assert resources.closed


def test_load_metrics_are_consumed_once_through_connector(monkeypatch):
    worker, _, _ = make_worker(async_load=False)
    connector = bind_connector(monkeypatch, worker)
    try:
        connector.bind_connector_metadata(KVTransferStep((LoadCommand("r", 0, 4, ((1,),), (b"a",)),)))
        connector.start_load_kv(None)
        connector.wait_for_save()
        result = connector.get_transfer_results(set())
        assert not result.finished_recving
        connector.get_block_ids_with_load_errors()
        stats = connector.get_kv_connector_stats()
        assert stats is not None and stats.reduce()["ascend_store_load_get_count"] == 1
        assert stats.reduce()["ascend_store_load_get_keys"] == 1
        assert connector.get_kv_connector_stats() is None
        connector.clear_connector_metadata()
    finally:
        connector.shutdown()


@pytest.mark.parametrize("load_raises", (False, True))
def test_model_state_copy_consumes_real_sync_load_before_forward(monkeypatch, load_raises):
    worker, _, backend = make_worker(async_load=False)
    connector = bind_connector(monkeypatch, worker)
    memory = {"loaded": 7, "running": 0}

    def load(keys, addresses, sizes):
        if load_raises:
            raise ValueError("destination safety is unknown")
        memory["loaded"] = 42
        return (0,) * len(keys)

    def copy_state():
        memory["running"] = memory["loaded"]

    monkeypatch.setattr(backend, "load", load)
    inner = SimpleNamespace(pre_forward=lambda *_args, **_kwargs: connector.start_load_kv(None))
    runner_connector = (
        MambaCopyKVConnector(inner, copy_state) if connector.requires_mamba_state_copy_after_load else inner
    )
    connector.bind_connector_metadata(KVTransferStep((LoadCommand("r", 0, 4, ((1,),), (b"a",)),)))
    if load_raises:
        with pytest.raises(ValueError, match="destination safety is unknown"):
            runner_connector.pre_forward(None)
        assert memory["running"] == 0
        with pytest.raises(ValueError, match="destination safety is unknown"):
            connector.shutdown()
    else:
        try:
            runner_connector.pre_forward(None)
            assert memory["running"] == 42
            connector.wait_for_save()
            connector.get_transfer_results(set())
            connector.get_block_ids_with_load_errors()
            connector.clear_connector_metadata()
        finally:
            connector.shutdown()
