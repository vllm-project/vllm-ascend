# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import worker as worker_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.connector import MooncakeSFACompatibleConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.metadata import MooncakeSFAConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.worker import (
    MooncakeSFAConnectorWorker,
    MooncakeSFAPullRecvingThread,
)

from .mooncake_sfa_helpers import make_thread
from .mooncake_v2.helpers import make_metadata_groups, make_pp_metadata, make_transfer_metadata
from .mooncake_v2.test_pull_worker import make_req_meta


def test_sfa_facade_forwards_sparse_preparation_to_its_worker() -> None:
    connector = object.__new__(MooncakeSFACompatibleConnector)
    worker = MagicMock(spec=MooncakeSFAConnectorWorker)
    connector.connector_worker = worker
    metadata = MooncakeSFAConnectorMetadata()
    prepare = getattr(connector, "prepare_sparse_kv_offload", None)
    assert prepare is not None
    prepare(metadata)
    worker.prepare_sparse_kv_offload.assert_called_once_with(metadata)


def test_sfa_facade_forwards_decode_handoff_without_exposing_state() -> None:
    connector = object.__new__(MooncakeSFACompatibleConnector)
    worker = MagicMock(spec=MooncakeSFAConnectorWorker)
    connector.connector_worker = worker
    batch, slots, tables = object(), object(), ()
    fills: dict[int, tuple[int, int]] = {}
    worker.prepare_copy_sfa_decode.return_value = (True, fills, None)
    assert connector.prepare_copy_sfa_decode(batch, slots, tables, fills, restore_tails=True) == (True, fills, None)
    worker.prepare_copy_sfa_decode.assert_called_once_with(batch, slots, tables, fills, restore_tails=True)


def test_sfa_receiver_rejects_missing_reader_before_starting() -> None:
    with pytest.raises(ValueError, match="requires a staging reader"):
        MooncakeSFAPullRecvingThread(staging_reader=None)


@pytest.mark.parametrize("result", [0, -1])
def test_execute_bucket_waits_for_staged_landing_and_propagates_failure(result):
    reader = MagicMock()
    reader.batch_transfer_sync_read.return_value = result
    thread = make_thread(engine=MagicMock(), staging_reader=reader)
    remote = make_pp_metadata()
    entries = {0: {(0, 0): [("request", [1], [2])]}}
    if result < 0:
        with pytest.raises(RuntimeError, match="transfer failed"):
            thread._execute_tp_transfer_bucket(0, 0, 0, 1, 1, 1, remote, entries)
    else:
        thread._execute_tp_transfer_bucket(0, 0, 0, 1, 1, 1, remote, entries)
    reader.batch_transfer_sync_read.assert_called_once_with("10.0.0.1:9000", [1128], [5256], [128], "request")
    thread.engine.batch_transfer_sync_read.assert_not_called()


@pytest.mark.parametrize("outcome", [set(), {"request"}, RuntimeError("transfer failed")])
def test_receive_drains_before_returning_failures_or_propagating_an_error(monkeypatch, outcome):
    events = []
    reader = MagicMock()
    reader.drain_pending_landing.side_effect = lambda: events.append("drained")
    thread = make_thread(staging_reader=reader)

    def transfer(*_args):
        events.append("submitted")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(MooncakePullRecvingThread, "_handle_requests", transfer)
    if isinstance(outcome, Exception):
        with pytest.raises(RuntimeError, match="transfer failed"):
            thread._handle_requests("engine", "host", 6000, {})
    else:
        assert thread._handle_requests("engine", "host", 6000, {}) == outcome
    assert events == ["submitted", "drained"]
    reader.drain_pending_landing.assert_called_once_with()


def test_thread_factory_injects_reader_and_leaves_startup_to_public_worker(monkeypatch):
    worker = MooncakeSFAConnectorWorker.__new__(MooncakeSFAConnectorWorker)
    worker._sfa_host_kv = SimpleNamespace(main_group_id=0)
    worker.xfer_handshake_metadata = make_transfer_metadata()
    worker.engine = object()
    worker.ascend_config = SimpleNamespace(mooncake_staging_config=object())
    reader = MagicMock()
    monkeypatch.setattr(worker_module, "MooncakeSFAStagingReader", MagicMock(return_value=reader))
    worker._build_recving_thread = MagicMock()
    ready = object()
    thread = worker._create_recving_thread(ready)
    worker._build_recving_thread.assert_called_once_with(MooncakeSFAPullRecvingThread, ready, staging_reader=reader)
    worker_module.MooncakeSFAStagingReader.assert_called_once_with(
        worker.engine,
        worker._sfa_host_kv,
        worker.xfer_handshake_metadata.block_lens,
        staging_config=worker.ascend_config.mooncake_staging_config,
    )
    thread.start.assert_not_called()


def test_executor_buckets_keep_batch_context_until_join_and_drain():
    reader = MagicMock()
    thread = make_thread(staging_reader=reader)
    remote = make_metadata_groups(pp_metadata=make_pp_metadata())
    thread._get_remote_metadata = MagicMock(return_value=remote)
    thread.remote_tp_rank_groups = {"engine": {0: {(0, 0): [[0]]}}}
    thread.remote_layer_index_pairs = {"engine": {0: [(0, 0)]}}
    events = []

    def buckets(*_args):
        request_id = next(iter(thread._sfa_batch_requests))
        return (
            {(0, 0): {0: {(0, 0): [(request_id, [1], [2])]}}},
            {(0, 0): {request_id}},
        )

    def execute(*_args):
        request_id = next(iter(thread._sfa_batch_requests))
        events.append(("read", request_id))
        if request_id == "failed":
            raise RuntimeError("READ failed")

    def drain():
        events.append(("drain", next(iter(thread._sfa_batch_requests))))

    thread._build_transfer_block_buckets = buckets
    thread._execute_tp_transfer_bucket = execute
    reader.drain_pending_landing.side_effect = drain
    with ThreadPoolExecutor(max_workers=2) as executor:
        thread.executor = executor
        for request_id in ["first", "failed", "last"]:
            requests = {request_id: make_req_meta()}
            failed = thread._handle_requests("engine", "host", 6000, requests)
            assert failed == ({request_id} if request_id == "failed" else set())
            assert thread._sfa_batch_requests is None
    assert events == [
        ("read", "first"),
        ("drain", "first"),
        ("read", "failed"),
        ("drain", "failed"),
        ("read", "last"),
        ("drain", "last"),
    ]
