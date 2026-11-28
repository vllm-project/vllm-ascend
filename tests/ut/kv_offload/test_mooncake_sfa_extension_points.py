# SPDX-License-Identifier: Apache-2.0
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePullConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import worker as worker_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.adapter import MooncakeSFAPDState
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.connector import MooncakeSFACompatibleConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.metadata import MooncakeSFAConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.worker import (
    MooncakeSFAConnectorWorker,
    MooncakeSFAPullRecvingThread,
)

from .mooncake_sfa_helpers import make_thread
from .mooncake_v2.helpers import make_pp_metadata, make_transfer_metadata
from .mooncake_v2.test_pull_worker import make_req_meta


@pytest.mark.parametrize("remote", [False, True])
def test_preemption_boundary_binds_without_preemptions_and_preserves_committed_decode(monkeypatch, remote):
    connector = object.__new__(MooncakeSFACompatibleConnector)
    worker = object.__new__(MooncakeSFAConnectorWorker)
    worker._sfa_pd_state = state = MooncakeSFAPDState(main_group_id=1)
    connector.connector_worker = worker
    metadata = MooncakeSFAConnectorMetadata({"request": 0})
    if remote:
        metadata.requests = {"request": make_req_meta()}
    events = []

    def handle_preemptions(_self, value):
        assert value is metadata
        assert not state.bindings
        events.append("preemptions")

    monkeypatch.setattr(MooncakePullConnector, "handle_preemptions", handle_preemptions)
    connector.handle_preemptions(metadata)
    assert events == ["preemptions"]
    assert state.bindings == {"request": 0}
    assert state.pending == {"request"}
    assert state.received == (set() if remote else {"request"})

    state.record_received({"request"}, set())
    state.commit({"request"})
    monkeypatch.setattr(MooncakePullConnector, "handle_preemptions", lambda _self, _metadata: None)
    connector.handle_preemptions(metadata)
    assert state.bindings == {"request": 0}
    assert not state.pending and not state.received


def test_failed_parent_preemption_does_not_publish_sfa_metadata(monkeypatch):
    connector = object.__new__(MooncakeSFACompatibleConnector)
    connector.connector_worker = MagicMock(spec=MooncakeSFAConnectorWorker)
    parent = MagicMock(side_effect=RuntimeError("preemption failed"))
    monkeypatch.setattr(MooncakePullConnector, "handle_preemptions", parent)
    metadata = MooncakeSFAConnectorMetadata({"request": 0})
    with pytest.raises(RuntimeError, match="preemption failed"):
        connector.handle_preemptions(metadata)
    parent.assert_called_once_with(metadata)
    connector.connector_worker.prepare_sparse_kv_offload.assert_not_called()


@pytest.mark.parametrize("worker_present", [False, True])
def test_empty_preemption_step_needs_no_sfa_state_or_device_io(monkeypatch, worker_present):
    connector = object.__new__(MooncakeSFACompatibleConnector)
    worker = object.__new__(MooncakeSFAConnectorWorker) if worker_present else None
    connector.connector_worker = worker
    monkeypatch.setattr(MooncakePullConnector, "handle_preemptions", lambda _self, _metadata: None)
    connector.handle_preemptions(MooncakeSFAConnectorMetadata())


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


@pytest.mark.parametrize("mode", [None, "batch", "request", "invalid"])
def test_sfa_inherits_upstream_pull_modes_and_default(mode):
    extra = {} if mode is None else {"pull_transfer_mode": mode}
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda key, default: extra.get(key, default))
    )
    if mode == "invalid":
        for cls in [MooncakeSFAPullRecvingThread, MooncakePullRecvingThread]:
            with pytest.raises(ValueError, match="must be 'batch' or 'request'"):
                cls._get_pull_transfer_mode(config)
    else:
        expected = mode or "request"
        assert MooncakeSFAPullRecvingThread._get_pull_transfer_mode(config) == expected
        assert MooncakePullRecvingThread._get_pull_transfer_mode(config) == expected


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


@pytest.mark.parametrize("outcome", [0, -1, RuntimeError("transfer failed")])
def test_read_failure_drains_before_propagating_to_the_future(outcome):
    events = []
    reader = MagicMock()
    reader.drain_pending_landing.side_effect = lambda: events.append("drained")
    thread = make_thread(staging_reader=reader)

    def transfer(*_args, **_kwargs):
        events.append("submitted")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    reader.batch_transfer_sync_read.side_effect = transfer
    if isinstance(outcome, Exception):
        with pytest.raises(RuntimeError, match="transfer failed"):
            thread._read_transfer_bucket("peer", [], [], [], ["request"], None)
        assert events == ["submitted", "drained"]
        reader.drain_pending_landing.assert_called_once_with()
    else:
        assert thread._read_transfer_bucket("peer", [], [], [], ["request"], None) == outcome
        assert events == ["submitted"]
        reader.drain_pending_landing.assert_not_called()


def test_thread_factory_injects_reader_and_leaves_startup_to_public_worker(monkeypatch):
    worker = MooncakeSFAConnectorWorker.__new__(MooncakeSFAConnectorWorker)
    worker._sfa_host_kv = SimpleNamespace(main_group_id=0)
    worker.xfer_handshake_metadata = make_transfer_metadata()
    worker.engine = object()
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
    )
    thread.start.assert_not_called()


def _prepare_sfa_receive(thread, pp_ranks=(0,)):
    """Use real bucket tasks; replace address layout and device I/O boundaries."""
    pp = SimpleNamespace(
        metadata_by_pcp_rank={
            0: SimpleNamespace(metadata_by_tp_rank={0: SimpleNamespace(local_ip="peer", te_rpc_port=9)})
        }
    )
    remote = SimpleNamespace(metadata_by_pp_rank={rank: pp for rank in pp_ranks}, pcp_size=1, tp_size=1, dcp_size=1)
    thread._get_remote_metadata = MagicMock(return_value=remote)
    thread.remote_tp_rank_groups = {"engine": {rank: {(0, 0): [[0]]} for rank in pp_ranks}}
    thread.remote_layer_index_pairs = {"engine": {rank: [(0, 0)] for rank in pp_ranks}}
    thread.layer_names = ["main"]
    thread.local_metadata = SimpleNamespace(local_ip="local")
    thread.tp_rank = 0
    thread._transfer_targets = None
    seen = []

    def buckets(_metadata, _layers, _groups, _pcp, _dcp, requests, _cache, **_kwargs):
        entries = [(req_id, [1], [2]) for req_id in requests]
        return {(0, 0): {0: {(0, 0): entries}}}, {(0, 0): set(requests)}

    def additional(_metadata, _pcp, _tp, entries, requests, src, dst, lengths, _targets):
        request_ids = {req_id for spec in entries.values() for pairs in spec.values() for req_id, _, _ in pairs}
        assert requests is not None and request_ids <= requests.keys()
        seen.append((request_ids, dict(requests)))
        src.append(1)
        dst.append(2)
        lengths.append(1)

    thread._build_transfer_block_buckets = buckets
    thread._append_spec_transfer_addresses = lambda **_kwargs: None
    thread._append_additional_reads = additional
    return remote, seen


@pytest.mark.parametrize("pp_ranks", [(0,), (0, 1)])
def test_executor_tasks_keep_private_request_context_after_queue_advances(pp_ranks):
    reader = MagicMock()
    reader.batch_transfer_sync_read.return_value = 0
    thread = make_thread(staging_reader=reader)
    remote, seen = _prepare_sfa_receive(thread, pp_ranks=pp_ranks)
    entered, release = threading.Event(), threading.Event()
    first, second = {"first": make_req_meta()}, {"second": make_req_meta()}
    original_first = dict(first)

    def block_pool():
        entered.set()
        assert release.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        thread.executor = executor
        blocker = executor.submit(block_pool)
        try:
            assert entered.wait(timeout=5)
            first_tasks, error = thread._submit_transfer_group("engine", remote, first)
            assert error is None
            first.clear()
            second_tasks, error = thread._submit_transfer_group("engine", remote, second)
            assert error is None
        finally:
            release.set()
        blocker.result(timeout=5)
        for future in [*first_tasks, *second_tasks]:
            future.result(timeout=5)
    assert seen == [({"first"}, original_first)] * len(pp_ranks) + [({"second"}, second)] * len(pp_ranks)
    reader.drain_pending_landing.assert_not_called()


@pytest.mark.parametrize("mode", ["request", "batch"])
def test_sfa_submission_preserves_request_selection_offsets(mode):
    reader = MagicMock()
    reader.batch_transfer_sync_read.return_value = 0
    thread = make_thread(staging_reader=reader, pull_transfer_mode=mode)
    _prepare_sfa_receive(thread)
    build_buckets = thread._build_transfer_block_buckets
    offsets = []

    def record_offset(*args, selection_index_offset=0, **kwargs):
        offsets.append(selection_index_offset)
        return build_buckets(*args, selection_index_offset=selection_index_offset, **kwargs)

    thread._build_transfer_block_buckets = record_offset
    thread.add_requests("engine", {"a": make_req_meta(), "b": make_req_meta()})
    with ThreadPoolExecutor(max_workers=2) as executor:
        thread.executor = executor
        thread._process_next()

    assert offsets == ([0, 1] if mode == "request" else [0])
    assert sorted(thread.finished_requests.get_nowait() for _ in range(2)) == [("a", False), ("b", False)]
    assert thread.request_queue.unfinished_tasks == 0


@pytest.mark.parametrize("mode", ["request", "batch"])
@pytest.mark.parametrize("failed", [False, True])
def test_sfa_future_stays_pending_until_its_host_landing_finishes(mode, failed):
    entered, release = threading.Event(), threading.Event()
    reader = MagicMock()
    thread = make_thread(staging_reader=reader, pull_transfer_mode=mode)
    _prepare_sfa_receive(thread)

    def landing():
        entered.set()
        assert release.wait(timeout=5), "Test did not release the pending Host landing"

    def read(*_args, **_kwargs):
        if failed:
            raise RuntimeError("Host landing wait failed")
        landing()
        return 0

    reader.batch_transfer_sync_read.side_effect = read
    reader.drain_pending_landing.side_effect = landing
    thread.add_requests("engine", {"request": make_req_meta(local=([10],))})
    with ThreadPoolExecutor(max_workers=1) as transfers:
        thread.executor = transfers
        with ThreadPoolExecutor(max_workers=1) as processors:
            processing = processors.submit(thread._process_next)
            try:
                assert entered.wait(timeout=5)
                assert thread.finished_requests.empty()
                assert thread.get_and_clear_invalid_block_ids() == set()
            finally:
                release.set()
            processing.result(timeout=5)
    assert thread.finished_requests.get_nowait() == ("request", failed)
    assert thread.get_and_clear_invalid_block_ids() == ({10} if failed else set())
    assert thread.request_queue.unfinished_tasks == 0


@pytest.mark.parametrize("failure", ["read", "submit", "wait"])
def test_sfa_request_completion_does_not_wait_for_an_unrelated_failed_landing(monkeypatch, failure):
    entered, release = threading.Event(), threading.Event()
    reader = MagicMock()
    thread = make_thread(staging_reader=reader, pull_transfer_mode="request")
    _, seen = _prepare_sfa_receive(thread, pp_ranks=(0, 1) if failure == "submit" else (0,))

    def landing():
        entered.set()
        assert release.wait(timeout=5), "Test did not release request a's landing"

    def read(_session, _src, _dst, _lengths, request_id, **_kwargs):
        if request_id == "a":
            if failure in ("read", "wait"):
                raise RuntimeError("Host landing wait failed")
            landing()
        return 0

    def drain():
        if failure == "wait" and not release.is_set():
            raise RuntimeError("Host drain still pending")
        landing()

    class Transfers(ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            if failure == "submit" and args[0] == 1 and "a" in fn.keywords["requests"]:
                raise RuntimeError("second bucket submission failed")
            return super().submit(fn, *args, **kwargs)

    reader.batch_transfer_sync_read.side_effect = read
    reader.drain_pending_landing.side_effect = drain
    if failure == "wait":
        monkeypatch.setattr(worker_module.time, "sleep", lambda _seconds: landing())
    requests = {"a": make_req_meta(local=([10],)), "b": make_req_meta(local=([20],))}
    thread.add_requests("engine", requests)
    with Transfers(max_workers=3) as executor:
        thread.executor = executor
        try:
            thread._process_next()
            assert entered.wait(timeout=5)
            assert thread.finished_requests.get(timeout=5) == ("b", False)
            assert thread.finished_requests.empty()
            assert thread.get_and_clear_invalid_block_ids() == set()
        finally:
            release.set()
    assert thread.finished_requests.get_nowait() == ("a", True)
    assert thread.finished_requests.empty()
    assert thread.get_and_clear_invalid_block_ids() == {10}
    assert all(set(context) == request_ids for request_ids, context in seen)
    if failure == "wait":
        assert reader.drain_pending_landing.call_count >= 2
