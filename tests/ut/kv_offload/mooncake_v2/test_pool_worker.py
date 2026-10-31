# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import json
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import pool_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_control import PoolManifestRegistry, pool_control_request
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import (
    LayerwisePoolConnectorMetadata,
    PoolObjectRef,
    PoolObjectSlice,
    PoolSaveRequest,
    PoolSourceProfile,
    PoolTransferMetadata,
    PoolWorkerReport,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.key_codec import PoolKeyIdentity
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_writer import LayerWriteRange


def make_worker(monkeypatch, *, owned_layers=None):
    backend = MagicMock()
    backend.layerwise_data_plane = "gva"
    backend.batch_add_lease.return_value = [0]
    backend.batch_remove_lease.return_value = 0
    backend.batch_copy_gva.return_value = 0
    backend.batch_alloc.return_value = [10000]
    backend.batch_write_finish.return_value = [0]
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=2, dtype=torch.float16)
    config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=spec),
        ],
    )
    worker = pool_worker.LayerwisePoolWorker(
        backend,
        config,
        worker_id="pp0/tp0",
        tp_rank=0,
        tp_size=1,
        dcp_rank=0,
        dcp_size=1,
        lease_ttl_ms=10000,
    )
    monkeypatch.setattr(
        pool_worker,
        "collect_configured_register_regions",
        MagicMock(
            return_value=SimpleNamespace(ptrs=[1000], lengths=[64]),
        ),
    )
    worker.register_kv_caches({"layer.0": torch.empty((2, 1, 4, 2), dtype=torch.float16)}, owned_layers=owned_layers)
    return worker, backend


def make_transfer(worker, *, format_override=None):
    cache = worker.build_target_slices(([0],), 0, 4)[0].cache
    if format_override is not None:
        cache = replace(cache, format_id=format_override)
    return PoolTransferMetadata(
        "transfer",
        "attempt",
        "digest",
        0,
        4,
        "retention",
        (PoolSourceProfile("profile", "incarnation", PoolKeyIdentity("memcache-v1", "model", 1, 1)),),
        (PoolObjectRef("profile", 0, "hash", 0, 0, 16, (PoolObjectSlice(cache, 0),)),),
    )


def test_prefix_probe_reconstructs_keys_without_allocating_or_touching_cache_bytes(monkeypatch):
    worker, backend = make_worker(monkeypatch)
    try:
        profile = PoolSourceProfile("source", "engine", PoolKeyIdentity("memcache-v1", "model", 1, 1))
        worker.enable_manifest_control(profile, "127.0.0.1", retention_seconds=5)
        backend.batch_is_readable.side_effect = lambda keys: [key.endswith(":h0@0") for key in keys]
        query = {
            "op": "prefix",
            "incarnation": "engine",
            "mode": "probe",
            "group_ids": [0],
            "granularity": 4,
            "request": {
                "transfer_id": "lookup",
                "attempt_id": "attempt",
                "request_digest": "digest",
                "token_start": 0,
                "token_end": 8,
                "group_hashes": [["h0", "h1"]],
            },
        }
        result = worker.probe_prefix(query)
        assert result == {"worker_id": worker.worker_id, "ready_ends": [4]}
        query["mode"] = "manifest"
        query["request"]["token_end"] = 4
        report = PoolWorkerReport.from_wire(worker.probe_prefix(query))
        assert report.committed
        keys = {part.key for ref in report.objects for part in ref.resolve(profile.key_identity)}
        assert len(keys) == 1 and next(iter(keys)).endswith(":h0@0")
        assert "address" not in json.dumps(report.to_wire())
        backend.batch_alloc.assert_not_called()
        backend.batch_copy_gva.assert_not_called()
    finally:
        worker.shutdown()


@pytest.mark.parametrize("failed_owner", ["writer", "reader", "executor", "_manifest_registry"])
def test_shutdown_preserves_backend_until_all_owners_release(failed_owner):
    worker = pool_worker.LayerwisePoolWorker.__new__(pool_worker.LayerwisePoolWorker)
    for name in ("writer", "reader", "executor", "_manifest_registry", "_control_server", "backend"):
        setattr(worker, name, MagicMock())
    worker._writes = {"pending": object()}
    failed_close = getattr(worker, failed_owner).close
    failed_close.side_effect = RuntimeError("native cleanup failed")
    with pytest.raises(RuntimeError, match="unresolved cleanup"):
        worker.shutdown()
    worker.backend.close.assert_not_called()
    assert "pending" in worker._writes
    for name in ("writer", "reader", "executor", "_manifest_registry"):
        getattr(worker, name).close.assert_called_once()
    failed_close.side_effect = None
    worker.shutdown()
    worker.backend.close.assert_called_once()
    assert not worker._writes


def test_worker_bulk_restore_reports_only_after_backend_completion(monkeypatch):
    worker, backend = make_worker(monkeypatch)
    try:
        backend.batch_get_key_info.return_value = [SimpleNamespace(size=lambda: 16, gva_list=lambda: [10000])]
        worker.restore_request("request", make_transfer(worker), ([1],))
        worker.reader._active["request"].future.result(timeout=5)
        results = worker.get_transfer_results()
        assert results.finished_recving == {"request"}
        assert not results.failed_recving
        backend.batch_copy_gva.assert_called_once()
        backend.batch_remove_lease.assert_called_once_with(["model@hash@0"])
        stats = worker.get_kv_connector_stats()
        assert stats.data == {"restore_completed": 1, "restore_bytes": 16}
        assert worker.get_kv_connector_stats() is None
    finally:
        worker.shutdown()


@pytest.mark.parametrize("cancelled", [False, True])
def test_async_native_failure_preserves_correlated_diagnostic(monkeypatch, cancelled):
    worker, backend = make_worker(monkeypatch)
    error_log = MagicMock()
    monkeypatch.setattr(pool_worker.logger, "error", error_log)
    try:
        backend.batch_get_key_info.return_value = [SimpleNamespace(size=lambda: 16, gva_list=lambda: [10000])]
        backend.batch_copy_gva.return_value = -123
        worker.restore_request("failed-request", make_transfer(worker), ([1],))
        with pytest.raises(RuntimeError) as failure:
            worker.reader._active["failed-request"].future.result(timeout=5)
        if cancelled:
            worker.reader.cancel("failed-request")
        results = worker.get_transfer_results()
        assert results.finished_recving == results.failed_recving == {"failed-request"}
        message, *arguments = error_log.call_args.args
        diagnostic = message % tuple(arguments)
        assert "worker=pp0/tp0" in diagnostic
        assert "request=failed-request" in diagnostic and "attempt=attempt" in diagnostic
        assert f"cancelled={cancelled}" in diagnostic and str(failure.value) in diagnostic
        assert "-123" in diagnostic
        error_log.assert_called_once()
        worker.get_transfer_results()
        error_log.assert_called_once()
        backend.batch_remove_lease.assert_called_once_with(["model@hash@0"])
    finally:
        worker.shutdown()


def test_worker_incompatible_format_fails_before_any_copy(monkeypatch):
    worker, backend = make_worker(monkeypatch)
    try:
        worker.restore_request("request", make_transfer(worker, format_override="incompatible"), ([1],))
        results = worker.get_transfer_results()
        assert results.finished_recving == results.failed_recving == {"request"}
        assert not worker.get_block_ids_with_load_errors()
        backend.batch_copy_gva.assert_not_called()
        assert worker.get_kv_connector_stats().data == {"restore_failed": 1}
    finally:
        worker.shutdown()


@pytest.mark.parametrize("failure", ["layout", "native_copy"])
def test_request_failure_waits_for_all_workers_without_invalidating_shared_prefix(monkeypatch, failure):
    first, first_backend = make_worker(monkeypatch)
    slow, slow_backend = make_worker(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    for backend in (first_backend, slow_backend):
        backend.batch_get_key_info.return_value = [SimpleNamespace(size=lambda: 16, gva_list=lambda: [10000])]

    def blocked_copy(*_):
        entered.set()
        assert release.wait(timeout=5)
        return 0

    def tail_transfer(worker, **kwargs):
        transfer = make_transfer(worker, **kwargs)
        ref = transfer.objects[0]
        cache = replace(ref.slices[0].cache, token_start=4, token_end=8)
        return replace(
            transfer, token_start=4, token_end=8, objects=(replace(ref, slices=(PoolObjectSlice(cache, 0),)),)
        )

    def worker_output(worker):
        result = worker.get_transfer_results()
        return ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            kv_connector_output=KVConnectorOutput(
                finished_recving=result.finished_recving,
                failed_recving=result.failed_recving,
                invalid_block_ids=worker.get_block_ids_with_load_errors(),
            ),
        )

    try:
        slow_backend.batch_copy_gva.side_effect = blocked_copy
        slow.restore_request("request", tail_transfer(slow), ([0, 1],))
        assert entered.wait(timeout=5)
        if failure == "layout":
            first.restore_request("request", tail_transfer(first, format_override="incompatible"), ([0, 1],))
        else:
            first_backend.batch_copy_gva.return_value = -1
            first.restore_request("request", tail_transfer(first), ([0, 1],))
            with pytest.raises(RuntimeError):
                first.reader._active["request"].future.result(timeout=5)
            assert list(first_backend.batch_copy_gva.call_args.args[1]) == [
                first.build_target_slices(([0, 1],), 4, 8)[0].address
            ]

        aggregator = KVOutputAggregator(expected_finished_count=2)
        partial = aggregator.aggregate([worker_output(first), worker_output(slow)]).kv_connector_output
        assert not partial.finished_recving
        assert not partial.failed_recving
        assert not partial.invalid_block_ids  # Block 0 may be a shared, untouched prefix.

        release.set()
        slow.reader._active["request"].future.result(timeout=5)
        final = aggregator.aggregate([worker_output(first), worker_output(slow)]).kv_connector_output
        assert final.finished_recving == final.failed_recving == {"request"}
        assert not final.invalid_block_ids

        # Exercise the installed vLLM request-failure handler, including a
        # second request which may share the same local prefix block.
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.recompute_kv_load_failures = True
        scheduler.failed_recving_kv_req_ids = set()
        scheduler.requests = {
            req_id: SimpleNamespace(status=RequestStatus.WAITING_FOR_REMOTE_KVS, num_computed_tokens=4)
            for req_id in ("request", "unrelated")
        }
        assert not scheduler._handle_failed_recving(final.failed_recving)
        assert scheduler.requests["request"].num_computed_tokens == 0
        assert scheduler.requests["unrelated"].num_computed_tokens == 4
        assert scheduler.failed_recving_kv_req_ids == {"request"}
    finally:
        release.set()
        first.shutdown()
        slow.shutdown()


def test_cancelled_restore_metadata_never_starts_new_native_io(monkeypatch):
    worker, backend = make_worker(monkeypatch)
    try:
        metadata = LayerwisePoolConnectorMetadata()
        metadata.loads["request"] = make_transfer(worker)
        metadata.local_block_ids["request"] = ([1],)
        metadata.cancelled.add("request")
        worker.start_load_kv(metadata)
        results = worker.get_transfer_results()
        assert results.finished_recving == {"request"}
        assert not results.failed_recving
        backend.batch_add_lease.assert_not_called()
        backend.batch_copy_gva.assert_not_called()
        assert not worker.get_block_ids_with_load_errors()
        assert worker.get_kv_connector_stats().data == {"restore_cancelled": 1}
    finally:
        worker.shutdown()


def test_worker_repeated_metadata_cannot_replace_in_flight_attempt(monkeypatch):
    worker, backend = make_worker(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    backend.batch_get_key_info.return_value = [SimpleNamespace(size=lambda: 16, gva_list=lambda: [10000])]

    def copy(*_):
        entered.set()
        assert release.wait(timeout=5)
        return 0

    backend.batch_copy_gva.side_effect = copy
    try:
        transfer = make_transfer(worker)
        worker.restore_request("request", transfer, ([1],))
        assert entered.wait(timeout=5)
        worker.restore_request("request", transfer, ([1],))
        with pytest.raises(RuntimeError, match="drains"):
            worker.restore_request("request", replace(transfer, attempt_id="retry"), ([1],))
        assert not worker.get_transfer_results().finished_recving
        release.set()
        worker.reader._active["request"].future.result(timeout=5)
        assert worker.get_transfer_results().finished_recving == {"request"}
        backend.batch_copy_gva.assert_called_once()
    finally:
        release.set()
        worker.shutdown()


def test_worker_write_report_follows_layer_commit(monkeypatch):
    worker, backend = make_worker(monkeypatch)
    try:
        worker.prepare_write(
            "request",
            "transfer",
            "attempt",
            {"key": 8},
            {
                "layer.0": (LayerWriteRange("key", 1000, 0, 8),),
            },
            (),
        )
        assert worker.take_write_reports() == []
        worker.save_kv_layer("layer.0", MagicMock())
        worker.wait_for_save()
        report = worker.take_write_reports()[0]
        assert report.committed
        assert report.worker_id == "pp0/tp0"
        backend.batch_write_finish.assert_called_once_with(["key"], [0])
        assert worker.get_kv_connector_stats().data == {"save_committed": 1, "save_bytes": 8}
    finally:
        worker.shutdown()


def test_worker_missing_layer_never_reports_committed(monkeypatch):
    worker, backend = make_worker(monkeypatch)
    try:
        worker.prepare_write(
            "request",
            "transfer",
            "attempt",
            {"key": 8},
            {
                "draft.0": (LayerWriteRange("key", 1000, 0, 8),),
            },
            (),
        )
        worker.wait_for_save()
        report = worker.take_write_reports()[0]
        assert not report.committed
        assert "Required layers" in report.error
        backend.batch_write_finish.assert_called_once_with(["key"], [-1])
        assert worker.get_kv_connector_stats().data == {"save_failed": 1}
    finally:
        worker.shutdown()


def test_slow_retention_acquisition_reports_failure_and_keeps_cleanup_ownership(monkeypatch):
    worker, backend = make_worker(monkeypatch)
    now = [0.0]
    transfer = make_transfer(worker)
    worker._profile = transfer.profiles[0]
    worker._manifest_registry = PoolManifestRegistry(
        "incarnation", worker._release_report, retention_seconds=3, clock=lambda: now[0]
    )

    def acquire(*args):
        now[0] = 3.0
        return [0]

    backend.batch_add_lease.side_effect = acquire
    try:
        key = transfer.objects[0].resolve(worker._profile.key_identity)[0].key
        worker.prepare_write(
            "request",
            "transfer",
            "attempt",
            {key: 16},
            {"layer.0": (LayerWriteRange(key, 1000, 0, 16),)},
            transfer.objects,
        )
        worker.save_kv_layer("layer.0", MagicMock())
        worker.wait_for_save()
        (report,) = worker.take_write_reports()
        assert not report.committed
        assert "before manifest publication" in report.error
        assert ("transfer", "attempt") in worker._retained
        stored, _ = worker._manifest_registry._reports[("transfer", "attempt")]
        assert stored == report
        now[0] = 6.0
        worker._manifest_registry.expire()
        backend.batch_remove_lease.assert_called_once_with([key])
        assert not worker._retained
    finally:
        worker.shutdown()


def test_worker_rejects_invalid_parallel_group_coordinates():
    with pytest.raises(ValueError, match="coordinates"):
        pool_worker.LayerwisePoolWorker(
            MagicMock(), MagicMock(), worker_id="x", tp_rank=2, tp_size=1, dcp_rank=0, dcp_size=1, lease_ttl_ms=10000
        )


def test_kvpp_empty_owner_reports_commit_and_receive_without_native_io(monkeypatch):
    worker, backend = make_worker(monkeypatch)
    transfer = make_transfer(worker)
    # Recreate registration with an explicit empty ownership set: these
    # registered tensors are KVPP scratch, not persistent transfer targets.
    worker.shutdown()
    worker, backend = make_worker(monkeypatch, owned_layers=set())
    try:
        profile = transfer.profiles[0]
        request = PoolSaveRequest("transfer", "attempt", "digest", 0, 4, ((1,),), (("hash",),))
        worker.prepare_save_request("request", request, profile)
        worker.wait_for_save()
        assert worker.take_write_reports() == [PoolWorkerReport("transfer", "attempt", worker.worker_id, True)]
        worker.restore_request("request", transfer, ([1],))
        worker.reader._active["request"].future.result(timeout=5)
        result = worker.get_transfer_results()
        assert result.finished_recving == {"request"}
        assert not result.failed_recving
        backend.batch_alloc.assert_not_called()
        backend.batch_add_lease.assert_not_called()
        backend.batch_copy_gva.assert_not_called()
        backend.batch_write_finish.assert_not_called()
        with pytest.raises(RuntimeError, match="already registered"):
            worker.register_kv_caches({"layer.0": torch.empty((2, 1, 4, 2), dtype=torch.float16)})
    finally:
        worker.shutdown()


def test_packed_mla_read_refuses_to_overwrite_a_partial_local_prefix(monkeypatch):
    worker, _ = make_worker(monkeypatch)
    try:
        spec = AscendMLAAttentionSpec(
            block_size=4, num_kv_heads=1, head_size=8, dtype=torch.uint8, cache_sparse_sfa_c8=True
        )
        config = SimpleNamespace(
            num_blocks=2, kv_cache_groups=[SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=spec)]
        )
        worker.cache_config = config
        worker.specs = {"layer.0": spec}
        worker.bindings = pool_worker.describe_cache_bindings(
            config, {"layer.0": torch.empty((2, 32), dtype=torch.uint8)}
        )
        with pytest.raises(ValueError, match="partial local prefix"):
            worker.build_target_slices(([1],), 1, 4)
        pages = worker.build_target_slices(([1],), 0, 3)
        assert len(pages) == 1 and pages[0].cache.vector_bytes == 32
    finally:
        worker.shutdown()


def test_failed_retention_acquisition_keeps_partial_ownership_until_cleanup_retry(monkeypatch):
    worker, backend = make_worker(monkeypatch)
    try:
        transfer = make_transfer(worker)
        profile = transfer.profiles[0]
        worker.enable_manifest_control(profile, "127.0.0.1", retention_seconds=5)
        backend.batch_add_lease.side_effect = TimeoutError("native timeout after partial acquisition")
        backend.batch_remove_lease.side_effect = [1, 0]
        report = PoolWorkerReport("transfer", "attempt", worker.worker_id, True, transfer.objects)
        with pytest.raises(RuntimeError, match="source retention failed"):
            worker._retain_report(report)
        assert worker.executor._pending_release
        failed = replace(report, committed=False, objects=(), error="native timeout")
        worker._manifest_registry.publish(failed)
        worker._manifest_registry.close()
        assert not worker._retained
        assert backend.batch_remove_lease.call_count == 2
    finally:
        worker.shutdown()


def test_missing_retention_key_does_not_leave_nonexistent_lease_for_shutdown(monkeypatch):
    worker, backend = make_worker(monkeypatch)
    try:
        transfer = make_transfer(worker)
        worker.enable_manifest_control(transfer.profiles[0], "127.0.0.1", retention_seconds=5)
        backend.absent_read_lease_keys.side_effect = lambda keys, _: set(keys)
        backend.batch_add_lease.return_value = [-3102]
        report = PoolWorkerReport("transfer", "attempt", worker.worker_id, True, transfer.objects)
        with pytest.raises(RuntimeError, match="source retention failed"):
            worker._retain_report(report)
        assert not worker._retained
        backend.batch_remove_lease.assert_not_called()
    finally:
        worker.shutdown()


@pytest.mark.parametrize("existing_prefix", [False, True])
def test_real_cpu_cache_bytes_survive_layerwise_pool_handoff_and_tp_scatter(monkeypatch, existing_prefix):
    # Exercise real pointers and all shared modules together. Only the native
    # store transport is replaced; object naming/coverage/layout are production code.
    memory = {}
    committed = set()
    backend = MagicMock()
    backend.layerwise_data_plane = "gva"

    def allocate(keys, sizes, ttl):
        for key, size in zip(keys, sizes, strict=True):
            assert key not in memory
            memory[key] = ctypes.create_string_buffer(size)
        return [ctypes.addressof(memory[key]) for key in keys]

    def copy(gvas, buffers, sizes, direction):
        for gva, buffer, size in zip(gvas, buffers, sizes, strict=True):
            ctypes.memmove(gva if direction == 0 else buffer, buffer if direction == 0 else gva, size)
        return 0

    def commit(keys, results):
        committed.update(key for key, result in zip(keys, results, strict=True) if result == 0)
        return [0] * len(keys)

    backend.batch_alloc.side_effect = allocate
    backend.batch_copy_gva.side_effect = copy
    backend.batch_write_finish.side_effect = commit
    backend.batch_is_readable.side_effect = lambda keys: [key in committed for key in keys]
    backend.batch_add_lease.side_effect = lambda keys, ttl: [0 if key in committed else -1 for key in keys]
    backend.batch_remove_lease.return_value = 0
    backend.batch_get_key_info.side_effect = lambda keys, **kwargs: [
        SimpleNamespace(size=lambda key=key: len(memory[key]), gva_list=lambda key=key: [ctypes.addressof(memory[key])])
        for key in keys
    ]
    monkeypatch.setattr(
        pool_worker,
        "collect_configured_register_regions",
        MagicMock(
            return_value=SimpleNamespace(ptrs=[], lengths=[]),
        ),
    )
    source_spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=2, dtype=torch.float16)
    target_spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=2, dtype=torch.float16)
    source_config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=source_spec),
        ],
    )
    target_config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=target_spec),
        ],
    )
    p = pool_worker.LayerwisePoolWorker(
        backend,
        source_config,
        worker_id="p0",
        tp_rank=0,
        tp_size=1,
        dcp_rank=0,
        dcp_size=1,
        lease_ttl_ms=10000,
        total_num_kv_heads=2,
    )
    d = pool_worker.LayerwisePoolWorker(
        backend,
        target_config,
        worker_id="d1",
        tp_rank=1,
        tp_size=2,
        dcp_rank=0,
        dcp_size=1,
        lease_ttl_ms=10000,
        total_num_kv_heads=2,
    )
    source = torch.arange(32, dtype=torch.float16).reshape(2, 2, 4, 2)
    target = torch.full((2, 4, 1, 2), -1, dtype=torch.float16)  # Different NHD layout.
    try:
        p.register_kv_caches({"layer.0": source})
        d.register_kv_caches({"layer.0": target})
        profile = PoolSourceProfile("p-profile", "p-incarnation", PoolKeyIdentity("memcache-v1", "model", 1, 1))
        handshake = p.enable_manifest_control(profile, "127.0.0.1", retention_seconds=5)
        profile = handshake.profile
        request = PoolSaveRequest("transfer", "attempt", "digest", 0, 6, ((0, 1),), (("full-hash",),))
        _, _, source_refs = p.build_source_plan(request, profile)
        ordinary_key = next(
            ref.resolve(profile.key_identity)[0].key for ref in source_refs if ref.object_id.startswith("pool-full-v1:")
        )
        expected_prefix = source[0].numpy().tobytes()
        if existing_prefix:
            # Seed the same component-major bytes an ordinary pool writer
            # publishes. PD must borrow them, not allocate or overwrite them.
            memory[ordinary_key] = ctypes.create_string_buffer(len(expected_prefix))
            ctypes.memmove(ctypes.addressof(memory[ordinary_key]), expected_prefix, len(expected_prefix))
            committed.add(ordinary_key)
        p.prepare_save_request("request", request, profile)
        p.save_kv_layer("layer.0", MagicMock())
        p.wait_for_save()
        report = p.take_write_reports()[0]
        assert report.committed
        assert ctypes.string_at(ctypes.addressof(memory[ordinary_key]), len(expected_prefix)) == expected_prefix
        control = {
            "op": "manifest",
            "incarnation": profile.engine_incarnation,
            "transfer_id": "transfer",
            "attempt_id": "attempt",
        }
        queried = pool_control_request(handshake.control_endpoint, control)
        assert PoolWorkerReport.from_wire(queried) == report
        handoff = PoolTransferMetadata("transfer", "attempt", "digest", 0, 6, "retention", (profile,), report.objects)
        received = PoolTransferMetadata.from_wire(json.loads(json.dumps(handoff.to_wire())))
        d.restore_request("request", received, ([0, 1],))
        d.reader._active["request"].future.result(timeout=5)
        assert d.get_transfer_results().finished_recving == {"request"}
        torch.testing.assert_close(target[0, :, 0, :], source[0, 1, :, :])
        torch.testing.assert_close(target[1, :2, 0, :], source[1, 1, :2, :])
        assert torch.all(target[1, 2:, 0, :] == -1)  # Invalid tail is untouched.
        pool_control_request(handshake.control_endpoint, control | {"op": "release"})
        assert not p._retained
        _, _, next_refs = p.build_source_plan(replace(request, attempt_id="retry"), profile)
        assert {item.object_id for item in next_refs} == {item.object_id for item in report.objects}
        assert any(item.resolve(profile.key_identity)[0].key == ordinary_key for item in next_refs)
        assert any(item.object_id.startswith("pool-component-v1:") for item in next_refs)
    finally:
        p.shutdown()
        d.shutdown()
