# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for V2 routing and the shared DSA landing lifecycle."""

import ctypes
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import staging as staging_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import transfer_targets as registry_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.adapter import MooncakeSFAHostKV
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.staging import MooncakeSFAStagingReader

from .mooncake_sfa_helpers import make_thread
from .mooncake_v2.helpers import make_pp_metadata

registry = registry_module.MooncakeKVRegistry()


@pytest.fixture(autouse=True)
def clear_registry(monkeypatch):
    # These regressions exercise CPU byte copies and lifecycle ordering.
    # Do not pass the fake streams into an installed torch_npu runtime.
    monkeypatch.setattr(torch.npu, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.npu, "Event", MagicMock())
    monkeypatch.setattr(torch.npu, "synchronize", MagicMock())
    registry.clear()
    yield
    registry.clear()


def make_host(kv_registry):
    # Byte fixtures need only registered ranges and a CPU implementation of the
    # Manager copy interface. Keep the real Host adapter's landing/ABI checks.
    host = MooncakeSFAHostKV.__new__(MooncakeSFAHostKV)
    host.registry = kv_registry
    host.transfer_caches = {}

    def copy_bytes(sources, destinations, lengths, count):
        assert sources.dtype == destinations.dtype == torch.int64
        assert lengths.dtype == count.dtype == torch.int32
        assert count.tolist() == [sources.numel()] == [destinations.numel()] == [lengths.numel()]
        for src, dst, length in zip(sources.tolist(), destinations.tolist(), lengths.tolist()):
            ctypes.memmove(dst, src, length)

    host.manager = SimpleNamespace(copy_sfa_kv=copy_bytes)
    return host


def make_reader(engine, *, pipeline=False, window=2, bank_workers=0, kv_registry=None):
    reader = MooncakeSFAStagingReader.__new__(MooncakeSFAStagingReader)
    reader.engine = engine
    reader.kv_registry = kv_registry if kv_registry is not None else registry_module.MooncakeKVRegistry()
    reader.host_kv = make_host(reader.kv_registry)
    reader.kv_caches = {}
    reader.block_len_per_addr = [[128]]
    reader._dsa_pd_mooncake_staging_lock = threading.Lock()
    reader._get_dsa_pd_mooncake_staging_window = lambda: window
    reader._dsa_pd_mooncake_staging_tensor = torch.empty(window * 2 * 1024 * 1024, dtype=torch.int8)
    reader._dsa_pd_mooncake_staging_bytes = reader._dsa_pd_mooncake_staging_tensor.nbytes
    reader._pipeline_landing_enabled = pipeline
    reader._pipeline_banks = 2
    reader._landing_streams = [MagicMock(), MagicMock()]
    reader._landing_stream = reader._landing_streams[0]
    if bank_workers is not None:
        # Lifecycle/order tests can isolate the serial path or executor sizes.
        # None exercises the production policy derived from effective banks.
        reader._get_dsa_pd_mooncake_bank_workers = lambda banks: min(bank_workers, len(banks))
    return reader


def test_readers_receive_independent_registry_instances(monkeypatch):
    monkeypatch.setattr(MooncakeSFAStagingReader, "_init_dsa_pd_mooncake_staging", lambda self: None)
    left_registry = registry_module.MooncakeKVRegistry()
    right_registry = registry_module.MooncakeKVRegistry()
    target = torch.zeros(64, dtype=torch.int8)
    left_registry.register_offloaded_tensor(target)
    left_host = SimpleNamespace(transfer_caches={}, registry=left_registry)
    right_host = SimpleNamespace(transfer_caches={}, registry=right_registry)
    left = MooncakeSFAStagingReader(MagicMock(), left_host, [[128]])
    right = MooncakeSFAStagingReader(MagicMock(), right_host, [[128]])
    assert left.kv_registry.resolve_offloaded_tensor(target.data_ptr(), target.nbytes)[0] is target
    assert right.kv_registry.resolve_offloaded_tensor(target.data_ptr(), target.nbytes) is None
    right_registry.clear()
    assert left.kv_registry.resolve_offloaded_tensor(target.data_ptr(), target.nbytes)[0] is target


@pytest.mark.parametrize("pipeline", [False, True])
def test_byte_views_and_resolved_targets_are_reused_per_receive(monkeypatch, pipeline):
    chunk = 2 * 1024 * 1024
    target = torch.zeros(3 * chunk, dtype=torch.int8)
    registry.register_offloaded_tensor(target)
    reader = make_reader(MagicMock(), pipeline=pipeline, window=1 if not pipeline else 2, kv_registry=registry)
    reader._ensure_dsa_pd_mooncake_staging_tensor = MagicMock(wraps=reader._ensure_dsa_pd_mooncake_staging_tensor)
    resolve = MagicMock(wraps=registry.resolve_offloaded_byte_tensor)
    monkeypatch.setattr(registry, "resolve_offloaded_byte_tensor", resolve)
    original_view = torch.Tensor.view
    flattened: list[int] = []

    def view(tensor, *args):
        if tensor is target:
            flattened.append(id(tensor))
        return original_view(tensor, *args)

    monkeypatch.setattr(torch.Tensor, "view", view)
    reads: list[int] = []

    def read(_session, local, _remote, lengths):
        value = len(reads) % 3 + 1
        reads.append(value)
        for dst, length in zip(local, lengths):
            ctypes.memset(dst, value, length)
        return 0

    reader.engine.batch_transfer_sync_read.side_effect = read
    for request in range(2):
        assert (
            reader.batch_transfer_sync_read("peer", [target.data_ptr()], [10 * chunk], [target.nbytes], str(request))
            == 0
        )
        assert len(flattened) == 1
        assert reader._ensure_dsa_pd_mooncake_staging_tensor.call_count == request + 1
        assert resolve.call_count == request + 1
        for piece in range(3):
            assert torch.all(target[piece * chunk : (piece + 1) * chunk] == piece + 1)


def test_receive_does_not_retain_old_host_tensor():
    reader = make_reader(MagicMock(), kv_registry=registry)
    reader.engine.batch_transfer_sync_read.return_value = 0
    target = torch.empty(128, dtype=torch.int8)
    registry.register_offloaded_tensor(target)
    ref = weakref.ref(target)
    assert reader.batch_transfer_sync_read("peer", [target.data_ptr()], [0xA00000], [128], "request") == 0
    del target
    gc.collect()
    assert ref() is None


def test_registry_byte_view_aliases_storage_and_expires_with_kv_tensor():
    local_registry = registry_module.MooncakeKVRegistry()
    target = torch.zeros(32, dtype=torch.int16)
    local_registry.register_offloaded_tensor(target)
    first, offset = local_registry.resolve_offloaded_byte_tensor(target.data_ptr() + 4, 8)
    second, _ = local_registry.resolve_offloaded_byte_tensor(target.data_ptr(), target.nbytes)
    assert first is second and offset == 4
    first.fill_(1)
    assert torch.all(target == 257)
    local_registry.clear()
    assert local_registry.resolve_offloaded_byte_tensor(target.data_ptr(), target.nbytes) is None
    local_registry.register_offloaded_tensor(target)
    renewed, _ = local_registry.resolve_offloaded_byte_tensor(target.data_ptr(), target.nbytes)
    assert renewed is not first
    target_ref, view_ref = weakref.ref(target), weakref.ref(renewed)
    del first, second, renewed, target
    gc.collect()
    assert target_ref() is None and view_ref() is None


@pytest.mark.parametrize("pipeline", [False, True])
def test_v2_native_bucket_routes_mixed_destinations_and_preserves_bytes(pipeline):
    source = torch.arange(1024, dtype=torch.int64).to(torch.uint8)
    host = torch.zeros_like(source)
    hbm = torch.zeros_like(source)
    registry.register_offloaded_tensor(host)
    reads: list[tuple[int, int, int]] = []

    def read(session, local, remote, lengths):
        reads.extend(zip(local, remote, lengths))
        for dst, src, size in zip(local, remote, lengths):
            ctypes.memmove(dst, src, size)
        return 0

    engine = MagicMock()
    engine.batch_transfer_sync_read.side_effect = read
    reader = make_reader(engine, pipeline=pipeline, kv_registry=registry)
    names = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    thread = make_thread(
        engine=engine,
        staging_reader=reader,
        layer_names=names,
        kv_caches_base_addr=[[host.data_ptr()], [hbm.data_ptr()]],
        block_strides=[[128], [128]],
        block_lens=[[128], [128]],
        block_shapes=[[(1, 16, 4)], [(1, 16, 4)]],
        block_size_scales=[[1], [1]],
    )
    remote = make_pp_metadata(layer_names=names, tp_base_addrs={0: [[source.data_ptr()], [source.data_ptr()]]})
    entries = {0: {(0, 0): [("request", [1, 2], [3, 4])], (1, 1): [("request", [1, 2], [3, 4])]}}
    thread._execute_tp_transfer_bucket(0, 0, 0, 1, 1, 1, remote, entries)
    assert torch.equal(host[128:384], source[384:640])
    assert torch.equal(hbm[128:384], source[384:640])
    assert torch.count_nonzero(host[:128]) == 0
    assert torch.count_nonzero(host[384:]) == 0
    assert all(not (host.data_ptr() <= dst < host.data_ptr() + host.nbytes) for dst, _, _ in reads)
    assert any(dst == hbm.data_ptr() + 128 for dst, _, _ in reads)
    assert reader._dsa_pd_mooncake_staging_lock.acquire(blocking=False)
    reader._dsa_pd_mooncake_staging_lock.release()


@pytest.mark.parametrize("failure", ["negative", "read_exception", "copy_exception"])
def test_pipeline_drains_before_releasing_staging_on_failure(monkeypatch, failure):
    chunk = 2 * 1024 * 1024
    host = torch.empty(chunk * 3, dtype=torch.int8)
    registry.register_offloaded_tensor(host)
    engine = MagicMock()
    reader = make_reader(engine, pipeline=True, kv_registry=registry)
    event = MagicMock()
    monkeypatch.setattr(torch.npu, "Event", MagicMock(return_value=event))
    reader.host_kv.manager.copy_sfa_kv = MagicMock()
    if failure == "copy_exception":
        engine.batch_transfer_sync_read.return_value = 0
        reader.host_kv.manager.copy_sfa_kv.side_effect = [None, RuntimeError("copy failed")]
    else:
        engine.batch_transfer_sync_read.side_effect = [0, -1 if failure == "negative" else RuntimeError("read failed")]
    args = ("peer", [host.data_ptr()], [chunk * 10], [host.nbytes], "request")
    if failure == "negative":
        assert reader.batch_transfer_sync_read(*args) == -1
    else:
        with pytest.raises(RuntimeError, match="failed"):
            reader.batch_transfer_sync_read(*args)
    event.synchronize.assert_called_once()
    if failure == "copy_exception":
        reader._landing_streams[1].synchronize.assert_called_once()
    assert reader._dsa_pd_mooncake_staging_lock.acquire(blocking=False)
    reader._dsa_pd_mooncake_staging_lock.release()


def test_peer_bucket_cannot_overwrite_staging_during_landing():
    host = torch.empty(256, dtype=torch.int8)
    registry.register_offloaded_tensor(host)
    engine = MagicMock()
    engine.batch_transfer_sync_read.return_value = 0
    reader = make_reader(engine, kv_registry=registry)
    landing_entered = threading.Event()
    release_landing = threading.Event()
    second_started = threading.Event()

    def land(*args, **kwargs):
        landing_entered.set()
        assert release_landing.wait(5)

    reader.host_kv.manager.copy_sfa_kv = land
    args = ("peer", [host.data_ptr()], [0xA00000], [128], "request")

    def second():
        second_started.set()
        return reader.batch_transfer_sync_read(*args)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(lambda: reader.batch_transfer_sync_read(*args))
        try:
            assert landing_entered.wait(5)
            other = executor.submit(second)
            assert second_started.wait(5)
            assert engine.batch_transfer_sync_read.call_count == 1
        finally:
            release_landing.set()
        assert first.result(timeout=5) == other.result(timeout=5) == 0


def test_missing_host_view_and_mismatched_descriptors_fail_without_read():
    host = torch.empty(128, dtype=torch.uint8)
    registry.register_cpu_buffer(host.data_ptr(), host.nbytes)
    reader = make_reader(MagicMock(), kv_registry=registry)
    with pytest.raises(RuntimeError, match="Missing DSA"):
        reader.batch_transfer_sync_read("peer", [host.data_ptr()], [0xA00000], [128], "request")
    with pytest.raises(ValueError, match="equal sizes"):
        reader.batch_transfer_sync_read("peer", [1], [], [128], "request")
    reader.engine.batch_transfer_sync_read.assert_not_called()


@pytest.mark.parametrize("pipeline", [False, True])
def test_receive_reuses_staging_for_every_landing(pipeline):
    chunk = 2 * 1024 * 1024
    target = torch.zeros(3 * chunk, dtype=torch.int8)
    registry.register_offloaded_tensor(target)
    engine = MagicMock()
    reader = make_reader(engine, pipeline=pipeline, window=2 if pipeline else 1, kv_registry=registry)
    reader._ensure_dsa_pd_mooncake_staging_tensor = MagicMock(wraps=reader._ensure_dsa_pd_mooncake_staging_tensor)

    def read(_session, local, _remote, lengths):
        for dst, length in zip(local, lengths):
            ctypes.memset(dst, 7, length)
        return 0

    engine.batch_transfer_sync_read.side_effect = read
    assert reader.batch_transfer_sync_read("peer", [target.data_ptr()], [10 * chunk], [target.nbytes], "request") == 0
    reader._ensure_dsa_pd_mooncake_staging_tensor.assert_called_once()
    assert torch.all(target == 7)


@pytest.mark.parametrize("read_result", [0, -7])
def test_read_probe_only_synchronizes_successful_reads_before_landing(monkeypatch, read_result):
    target = torch.empty(64, dtype=torch.int8)
    registry.register_offloaded_tensor(target)
    reader = make_reader(MagicMock(), pipeline=True, kv_registry=registry)
    order = []

    def read(*_args):
        order.append("read")
        return read_result

    reader.engine.batch_transfer_sync_read.side_effect = read
    reader.host_kv.manager.copy_sfa_kv = lambda *_args: order.append("land")
    monkeypatch.setenv("VLLM_ASCEND_DSA_PD_MOONCAKE_READ_PROBE", "1")
    monkeypatch.setattr(torch.npu, "synchronize", lambda: order.append("probe"))
    info, error = MagicMock(), MagicMock()
    monkeypatch.setattr(staging_module.logger, "info", info)
    monkeypatch.setattr(staging_module.logger, "error", error)

    assert reader.batch_transfer_sync_read("peer", [target.data_ptr()], [0xA00000], [64], "probe") == read_result
    assert order == (["read", "probe", "land"] if read_result == 0 else ["read"])
    assert not reader._dsa_pd_mooncake_staging_lock.locked()
    if read_result == 0:
        info.assert_called_once()
        assert info.call_args.args[3:7] == (1, 1, 1, 1)  # batches, chunks, copies, plan
        error.assert_not_called()
    else:
        info.assert_not_called()
        error.assert_called_once()
