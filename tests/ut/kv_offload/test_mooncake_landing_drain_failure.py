# SPDX-License-Identifier: Apache-2.0
"""Do not publish receive completion while Host copies may still be running."""

import ctypes
import threading
import weakref
from contextlib import nullcontext
from typing import TypedDict
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import worker as pull_worker

from .mooncake_sfa_helpers import make_thread
from .mooncake_v2.test_pull_worker import make_req_meta
from .test_dsa_pd_mooncake_connector_v2 import make_reader


class StopWorkerLoop(Exception):
    pass


class PendingReceiveState(TypedDict):
    pending: bool
    wait_fails: bool
    waits: int
    descriptors: list[weakref.ReferenceType[torch.Tensor]]


def pending_receive(monkeypatch, *, pipeline, land_raises):
    monkeypatch.setenv("VLLM_ASCEND_DSA_PD_MOONCAKE_COPY_STREAMS_PER_BANK", "1")
    monkeypatch.setattr(torch.npu, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.npu, "Event", MagicMock())
    reader = make_reader(MagicMock(), pipeline=pipeline, window=2)
    reader._pipeline_banks = 1
    destination = torch.zeros(64, dtype=torch.int8)
    source = torch.arange(64, dtype=torch.int8)
    reader.kv_registry.register_offloaded_tensor(destination)
    state: PendingReceiveState = {"pending": False, "wait_fails": True, "waits": 0, "descriptors": []}

    def read(_session, local, remote, lengths):
        assert not state["pending"], "Reused staging before the previous copy stopped"
        for dst, src, length in zip(local, remote, lengths):
            ctypes.memmove(dst, src, length)
        return 0

    def enqueue(*_args):
        state["pending"] = True

    original_land = reader.host_kv.land

    def land(copies, staging, owners):
        original_land(copies, staging, owners)
        state["descriptors"].extend(weakref.ref(tensor) for tensor in owners[-1])
        if land_raises:
            raise RuntimeError("injected landing failure after submission")

    def synchronize(device=None):
        state["waits"] += 1
        assert all(ref() is not None for ref in state["descriptors"])
        if not pipeline:
            assert device == reader._dsa_pd_mooncake_staging_tensor.device
        if state["wait_fails"]:
            raise RuntimeError("injected NPU wait failure")
        state["pending"] = False

    reader.engine.batch_transfer_sync_read.side_effect = read
    reader.host_kv.manager.copy_sfa_kv = enqueue
    reader.host_kv.land = land
    reader._landing_stream.synchronize.side_effect = synchronize
    monkeypatch.setattr(torch.npu, "synchronize", synchronize)
    # Event failure must fall back to the actual copy stream, which may also fail.
    monkeypatch.setattr(torch.npu, "Event", lambda: MagicMock(synchronize=MagicMock(side_effect=synchronize)))

    def receive():
        return reader.batch_transfer_sync_read("peer", [destination.data_ptr()], [source.data_ptr()], [64], "request")

    return reader, state, receive


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("land_raises", [False, True])
def test_failed_wait_retains_owners_and_blocks_staging_reuse(monkeypatch, pipeline, land_raises):
    reader, state, receive = pending_receive(monkeypatch, pipeline=pipeline, land_raises=land_raises)
    with pytest.raises(RuntimeError, match="injected NPU wait failure"):
        receive()
    bank = reader._dsa_pd_mooncake_staging_banks[0]
    assert state["pending"] and bank.inflight_owners
    assert bank.completion_pending or bank.needs_stream_drain or bank.pending_device is not None
    assert all(ref() is not None for ref in state["descriptors"])

    with pytest.raises(RuntimeError, match="injected NPU wait failure"):
        receive()
    assert reader.engine.batch_transfer_sync_read.call_count == 1
    assert state["pending"] and bank.inflight_owners
    assert all(ref() is not None for ref in state["descriptors"])

    state["wait_fails"] = False
    reader.drain_pending_landing()
    assert not state["pending"]
    assert not bank.inflight_owners
    assert not bank.completion_pending and not bank.needs_stream_drain and bank.pending_device is None
    assert all(ref() is None for ref in state["descriptors"])
    waits = state["waits"]
    reader.drain_pending_landing()
    assert state["waits"] == waits, "An already drained receive must not add an NPU wait"


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("can_report", [False, True])
@pytest.mark.parametrize("persistent_failure", [False, True])
def test_worker_defers_results_and_invalid_blocks_until_drain(monkeypatch, pipeline, can_report, persistent_failure):
    reader, state, receive = pending_receive(monkeypatch, pipeline=pipeline, land_raises=True)
    # A submission error with a successful drain is safe to report immediately.
    state["wait_fails"] = persistent_failure
    worker = make_thread(
        device=0, ready_event=threading.Event(), staging_reader=reader, can_report_invalid_block_ids=can_report
    )
    requests = {"request": make_req_meta(local=([10],))}
    worker.request_queue = MagicMock()
    worker.request_queue.get.side_effect = [("engine-p", requests), StopWorkerLoop()]
    worker._handle_requests = lambda *_args: receive()
    monkeypatch.setattr(torch.npu, "set_device", MagicMock())
    waiting = threading.Event()
    resume = threading.Event()
    errors = []

    def retry_sleep(_seconds):
        waiting.set()
        assert resume.wait(timeout=5), "Test did not release the failed drain"

    monkeypatch.setattr(pull_worker.time, "sleep", retry_sleep)

    def run():
        try:
            worker.run()
        except StopWorkerLoop:
            pass
        except BaseException as error:
            errors.append(error)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    try:
        if persistent_failure:
            assert waiting.wait(timeout=5)
            assert state["pending"]
            assert all(ref() is not None for ref in state["descriptors"])
            assert worker.get_and_clear_finished_requests() == KVConnectorTransferResults()
            assert worker.get_and_clear_invalid_block_ids() == set()
            worker.request_queue.task_done.assert_not_called()
            assert worker.request_queue.get.call_count == 1
    finally:
        state["wait_fails"] = False
        resume.set()
        runner.join(timeout=5)
    assert not runner.is_alive() and not errors
    assert not state["pending"]
    assert all(ref() is None for ref in state["descriptors"])
    assert worker.get_and_clear_finished_requests() == KVConnectorTransferResults(
        finished_recving={"request"}, failed_recving={"request"}
    )
    assert worker.get_and_clear_finished_requests() == KVConnectorTransferResults()
    assert worker.get_and_clear_invalid_block_ids() == ({10} if can_report else set())
    worker.request_queue.task_done.assert_called_once_with()
