"""Connector hooks propagate Scheduler and Worker metadata and lifecycle results."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

from tests.ut.distributed.ascend_store.v1.scheduler.fixtures import (
    FakeBlocks,
    make_output,
    make_request,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector import (
    AscendStoreV1Connector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    KVTransferStep,
    LoadCommand,
    LoadCommandBatch,
    StoreSourceReleaseMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.result import (
    LoadResult,
)


def test_connector_scheduler_hooks_are_thin_delegations() -> None:
    calls: list[tuple[Any, ...]] = []
    expected_step = KVTransferStep()

    def get_num_new_matched_tokens(*args):
        calls.append(("lookup", args))
        return 3, True

    def build_step(output):
        calls.append(("step", output))
        return expected_step

    def register_finished_partial_tail(*args):
        calls.append(("tail", args))
        return False

    scheduler = SimpleNamespace(
        get_num_new_matched_tokens=get_num_new_matched_tokens,
        confirm_allocation=lambda *args: calls.append(("allocation", args)),
        build_step=build_step,
        accept_worker_metadata=lambda metadata: calls.append(("worker", metadata)),
        register_finished_partial_tail=register_finished_partial_tail,
        bind_gpu_block_pool=lambda pool: calls.append(("pool", pool)),
        has_pending_push_work=lambda: True,
        has_pending_block_frees=lambda: True,
        close=lambda: calls.append(("close",)),
    )
    connector = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    connector.scheduler = scheduler
    connector.worker = None
    connector.lookup_server = None
    request = make_request()
    blocks = FakeBlocks([1, 2])
    output = make_output()
    worker_metadata = object()

    assert connector.get_num_new_matched_tokens(request, 2) == (3, True)
    connector.update_state_after_alloc(request, blocks, 3)
    assert connector.build_connector_meta(output) is expected_step
    connector.update_connector_output(SimpleNamespace(kv_connector_worker_meta=worker_metadata))
    connector.register_finished_partial_tail(request, ([1, 2],), [(0, 2, 8)])
    connector.bind_gpu_block_pool("pool")
    assert connector.has_pending_push_work()
    assert connector.has_pending_block_frees()
    connector.shutdown()

    assert calls == [
        ("lookup", (request, 2)),
        ("allocation", (request, blocks, 3)),
        ("step", output),
        ("worker", worker_metadata),
        ("tail", (request, ([1, 2],), [(0, 2, 8)])),
        ("pool", "pool"),
        ("close",),
    ]


def test_worker_connector_hooks_delegate_and_drain_worker_results() -> None:
    calls: list[tuple[Any, ...]] = []
    released_store_job_ids = {17}

    def take_released_store_job_ids():
        released = set(released_store_job_ids)
        released_store_job_ids.clear()
        return released

    load_result = LoadResult(frozenset({"request"}), frozenset(), frozenset({7}))
    worker = SimpleNamespace(
        bind_kv_caches=lambda caches: calls.append(("bind", caches)),
        start_load=lambda: calls.append(("start_load",)),
        wait_for_layer_load=lambda layer: calls.append(("wait_layer", layer)),
        save_layer=lambda layer: calls.append(("save_layer", layer)),
        finish_step=lambda: calls.append(("finish",)),
        fence_previous_store=lambda request_ids=None: calls.append(("fence", request_ids)),
        take_released_store_job_ids=take_released_store_job_ids,
        collect_load_result=lambda: load_result,
        close=lambda: calls.append(("close",)),
    )
    connector = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    connector.scheduler = None
    connector.worker = worker
    connector.lookup_server = None
    connector._pending_load_result = None
    connector._released_store_job_ids = set()

    connector.register_kv_caches({"layer": "cache"})
    connector.start_load_kv(None)
    connector.wait_for_layer_load("layer.0")
    connector.save_kv_layer("layer.0", None, None)
    connector.wait_for_save()
    connector.handle_preemptions(KVTransferStep())
    assert connector.get_finished(set()) == (set(), {"request"})
    assert connector.get_block_ids_with_load_errors() == {7}
    metadata = connector.build_connector_worker_meta()
    assert isinstance(metadata, StoreSourceReleaseMetadata)
    assert metadata.released_store_jobs == {17: 1}
    assert connector.build_connector_worker_meta() is None
    connector.shutdown()

    assert calls == [
        ("bind", {"layer": "cache"}),
        ("start_load",),
        ("wait_layer", "layer.0"),
        ("save_layer", "layer.0"),
        ("finish",),
        ("fence", frozenset()),
        ("close",),
    ]


@pytest.mark.parametrize(
    "released_ids,has_load",
    (
        (frozenset(), False),
        (frozenset(("finished", "preempted")), False),
        (frozenset(), True),
        (frozenset(("finished", "preempted")), True),
    ),
)
def test_store_fence_preserves_released_request_ids_in_both_load_and_decode_steps(released_ids, has_load) -> None:
    fenced_requests = []
    connector = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    connector.scheduler = None
    connector.worker = SimpleNamespace(
        fence_previous_store=lambda request_ids=None: fenced_requests.append(request_ids),
        take_released_store_job_ids=lambda: set(),
    )
    connector._released_store_job_ids = set()
    load = LoadCommand("load", TokenRange(0, 4), ((7,),), (b"a",))
    metadata = KVTransferStep(load=LoadCommandBatch((load,) if has_load else ()), released_request_ids=released_ids)

    connector.handle_preemptions(metadata)

    assert fenced_requests == [released_ids]


@pytest.mark.parametrize("pending_push,pending_frees", ((False, False), (True, False), (True, True)))
def test_pending_block_frees_are_not_inferred_from_background_push_work(pending_push, pending_frees) -> None:
    connector = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    connector.scheduler = SimpleNamespace(
        has_pending_push_work=lambda: pending_push,
        has_pending_block_frees=lambda: pending_frees,
    )
    assert connector.has_pending_push_work() is pending_push
    assert connector.has_pending_block_frees() is pending_frees
    connector.scheduler = None
    assert not connector.has_pending_push_work()
    assert not connector.has_pending_block_frees()


def test_worker_connector_rolls_back_metadata_when_worker_rejects_step(monkeypatch) -> None:
    calls: list[tuple[Any, ...]] = []
    step = KVTransferStep()

    monkeypatch.setattr(
        KVConnectorBase_V1,
        "bind_connector_metadata",
        lambda _self, metadata: calls.append(("bind", metadata)),
    )
    monkeypatch.setattr(
        KVConnectorBase_V1,
        "clear_connector_metadata",
        lambda _self: calls.append(("clear",)),
        raising=False,
    )

    def reject_step(metadata):
        calls.append(("begin", metadata))
        raise RuntimeError("previous Store failed")

    connector = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    connector.worker = SimpleNamespace(begin_step=reject_step)

    with pytest.raises(RuntimeError, match="previous Store failed"):
        connector.bind_connector_metadata(step)

    assert calls == [("bind", step), ("begin", step), ("clear",)]


def test_connector_retains_worker_on_lookup_close_failure_and_can_retry_shutdown() -> None:
    calls = []
    query_running = True

    def close_lookup():
        calls.append("lookup")
        if query_running:
            raise RuntimeError("Lookup is still running")

    connector = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    connector.scheduler = None
    connector.lookup_server = SimpleNamespace(close=close_lookup)
    connector.worker = SimpleNamespace(close=lambda: calls.append("worker"))
    with pytest.raises(RuntimeError, match="Lookup is still running"):
        connector.shutdown()
    assert calls == ["lookup"]
    query_running = False
    connector.shutdown()
    assert calls == ["lookup", "lookup", "worker"]


def test_connector_closes_created_worker_if_lookup_server_cannot_start(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(KVConnectorBase_V1, "__init__", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        connector_module,
        "create_kv_pool_worker",
        lambda *_args: SimpleNamespace(lookup=lambda *_args: None, close=lambda: calls.append("worker")),
    )
    monkeypatch.setattr(AscendStoreV1Connector, "_resolve_lookup_address", lambda *_args: "unsupported://lookup")
    with pytest.raises(RuntimeError, match="Failed to start Lookup server"):
        AscendStoreV1Connector(SimpleNamespace(parallel_config=SimpleNamespace(rank=0)), None, None)
    assert calls == ["worker"]
