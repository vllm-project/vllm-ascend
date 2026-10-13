# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that TP diagnostics observe transfers without changing their results or lifetime."""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest
import torch

from tests.e2e.pull_request.four_card.ascendstore_v1_tp_timing import AscendStoreTPBenchmarkWorkerExtension
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk.worker import SynchronousBulkWorker


@pytest.fixture
def tp_worker(monkeypatch):
    def make(implementation, producer):
        extension = AscendStoreTPBenchmarkWorkerExtension()
        role = "kv_producer" if producer else "kv_consumer"
        extension.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_role=role))
        backend = SimpleNamespace(exists=Mock(return_value=[1, 1]))
        operation = "store" if producer else "load"
        method = ("put" if producer else "get") if implementation == "production" else operation
        transfer = Mock(return_value=None if method == "put" else [0, 0])
        setattr(backend, method, transfer)
        if implementation == "production":
            worker = KVPoolWorker.__new__(KVPoolWorker)
            worker.m_store = backend
            worker.tp_mismatch = True
            worker.tp_rank, worker.tp_size, worker.num_sub_keys = (0, 1, 2) if producer else (1, 2, 1)
            worker.kv_send_thread = SimpleNamespace(done_task_lock=threading.Lock(), stored_requests={"request": 1})
            worker.handle_released_saves = create_autospec(
                worker.handle_released_saves,
                side_effect=lambda requests: worker.kv_send_thread.stored_requests.clear(),
            )
            parameter_method = "_build_tp_mismatch_keys_and_addrs"
            worker._build_tp_mismatch_keys_and_addrs = create_autospec(worker._build_tp_mismatch_keys_and_addrs)
            connector_type = type(
                "AscendStoreConnector",
                (),
                {"__module__": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector"},
            )
            connector = connector_type()
            connector.connector_worker = worker
        else:
            worker = SynchronousBulkWorker.__new__(SynchronousBulkWorker)
            worker._resources = SimpleNamespace(backend=backend)
            worker._pending_store_batch = None
            worker._topology = SimpleNamespace(
                tp_rank=0 if producer else 1,
                tp_size=1 if producer else 2,
                tp_partition=SimpleNamespace(tp_mismatch=True, key_slices_per_rank=2 if producer else 1),
            )
            parameter_method = "_materialize_concrete_bulk_arguments"
            for name in (
                "_build_store_candidates",
                "_materialize_store_candidates",
                "_build_load_batch",
                "_materialize_concrete_bulk_arguments",
                "fence_previous_store",
            ):
                setattr(worker, name, create_autospec(getattr(worker, name)))
            connector_type = type(
                "AscendStoreV1Connector",
                (),
                {"__module__": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector"},
            )
            connector = connector_type()
            connector.worker = worker
        monkeypatch.setitem(
            sys.modules, "vllm.distributed.kv_transfer", SimpleNamespace(get_kv_transfer_group=lambda: connector)
        )
        monkeypatch.setattr(torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False)
        return extension, worker, backend, method, transfer, parameter_method

    return make


@pytest.mark.parametrize("implementation", ["production", "v1"])
@pytest.mark.parametrize("producer", [True, False])
def test_tp_timing_preserves_results_and_drains_only_producers(tp_worker, implementation, producer):
    extension, worker, backend, method, transfer, parameter_method = tp_worker(implementation, producer)
    keys, addresses, sizes = ["first", "second"], [[10, 20], [30]], [[4, 4], [8]]
    parameter_result = object()
    getattr(worker, parameter_method).return_value = parameter_result
    identity = extension.start_ascendstore_tp_timing()
    assert identity["tp_size"] == (1 if producer else 2)
    assert identity["key_slices_per_rank"] == (2 if producer else 1)
    if implementation == "production":
        result = getattr(worker, parameter_method)([], [], 0)
    else:
        result = getattr(worker, parameter_method)(None, store=producer)
    assert result is parameter_result
    assert getattr(backend, method)(keys, addresses, sizes) is transfer.return_value
    assert backend.exists(keys) == [1, 1]
    observed = extension.stop_ascendstore_tp_timing()
    records = observed["records"]
    assert len(records) == 3 and all(record["host_ms"] >= 0 for record in records)
    transfer_record = next(record for record in records if "keys" in record)
    assert transfer_record["range_count"] == 3 and transfer_record["byte_count"] == 16
    assert transfer_record["result_codes"] == (None if method == "put" else [0, 0])
    assert transfer_record["keys"] == keys
    fence = worker.handle_released_saves if implementation == "production" else worker.fence_previous_store
    assert fence.call_count == int(producer)
    assert getattr(backend, method) is transfer
    getattr(backend, method)(keys, addresses, sizes)
    assert len(records) == 3, "Uninstrumented requests must not be recorded"
    assert not hasattr(extension, "_ascendstore_tp_timing")


@pytest.mark.parametrize("failure", ["load", "drain", "install"])
def test_tp_timing_reports_failure_and_restores_hooks(tp_worker, failure):
    producer = failure == "drain"
    extension, worker, backend, method, transfer, _ = tp_worker("v1", producer)
    if failure == "install":
        del backend.exists
        with pytest.raises(AttributeError):
            extension.start_ascendstore_tp_timing()
    else:
        extension.start_ascendstore_tp_timing()
        if failure == "load":
            transfer.side_effect = RuntimeError("transfer failed")
            with pytest.raises(RuntimeError, match="transfer failed"):
                backend.load(["key"], [[10]], [[8]])
            records = extension.stop_ascendstore_tp_timing()["records"]
            assert records[0]["error"] == "RuntimeError: transfer failed"
        else:
            worker.fence_previous_store.side_effect = RuntimeError("drain failed")
            with pytest.raises(RuntimeError, match="drain failed"):
                extension.stop_ascendstore_tp_timing()
    assert getattr(backend, method) is transfer
    assert not hasattr(extension, "_ascendstore_tp_timing")
