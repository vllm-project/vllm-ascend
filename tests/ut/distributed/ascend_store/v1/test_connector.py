"""Connector hooks propagate Scheduler and Worker metadata and lifecycle results."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector import (
    AscendStoreV1Connector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
)


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
    load = LoadCommand("load", 0, 4, ((7,),), (b"a",))
    metadata = KVTransferStep(load_commands=(load,) if has_load else (), released_request_ids=released_ids)

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
    monkeypatch.setattr(
        connector_module, "resolve_kv_pool_config", lambda *_: SimpleNamespace(use_layerwise=False, load_async=False)
    )
    monkeypatch.setattr(connector_module, "validate_kv_pool_config", lambda *_: None)
    monkeypatch.setattr(KVConnectorBase_V1, "__init__", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        AscendStoreV1Connector,
        "_create_worker",
        lambda *_args: SimpleNamespace(lookup=lambda *_args: None, close=lambda: calls.append("worker")),
    )
    monkeypatch.setattr(AscendStoreV1Connector, "_resolve_lookup_address", lambda *_args: "unsupported://lookup")
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(rank=0),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
    )
    with pytest.raises(RuntimeError, match="Failed to start RPC server"):
        AscendStoreV1Connector(config, None, None)
    assert calls == ["worker"]
