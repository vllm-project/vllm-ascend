# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for V2 routing and the shared DSA landing lifecycle."""

import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker, pull_worker
from vllm_ascend.kv_offload import dsa_pd_mooncake as registry
from vllm_ascend.kv_offload.dsa_pd_mooncake_staging import DSAPDMooncakeStagingReader

from .mooncake_v2.helpers import make_full_spec, make_kv_cache_tensor, make_pp_metadata, make_transfer_metadata
from .mooncake_v2.test_pull_worker import make_thread


@pytest.fixture(autouse=True)
def clear_registry():
    registry.reset_dsa_pd_mooncake_kv_registries()
    yield
    registry.reset_dsa_pd_mooncake_kv_registries()


def make_reader(engine, *, pipeline=False, window=2):
    reader = DSAPDMooncakeStagingReader.__new__(DSAPDMooncakeStagingReader)
    reader.engine = engine
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
    return reader


@pytest.mark.parametrize("host_only", [False, True])
def test_host_allocation_excluded_but_metadata_and_hbm_registration_preserved(monkeypatch, host_only):
    # Padding after the Host view belongs to its allocation too. The registry
    # must exclude that entire registration region, not only the logical view.
    host_storage = torch.empty(2048, dtype=torch.uint8)
    host = host_storage[:512].view(torch.float16).view(2, 16, 1, 8)
    device = torch.empty_like(host)
    registry.register_dsa_pd_mooncake_cpu_kv_buffer(host_storage.data_ptr(), host_storage.nbytes)
    registry.register_dsa_pd_mooncake_offloaded_kv_tensor(host)
    caches = {"layer.host": host}
    if not host_only:
        caches["layer.device"] = device
    worker = base_worker.MooncakeBaseConnectorWorker.__new__(base_worker.MooncakeBaseConnectorWorker)
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.pcp_rank = worker.tp_rank = 0
    worker.tp_size = 1
    worker.engine_id, worker.te_rpc_port = "D", 9000
    worker.block_size, worker.side_channel_host, worker.handshake_port = 16, "127.0.0.1", 5000
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            make_kv_cache_tensor(size=c.nbytes, layers=[name], layer_stride=c.nbytes, block_stride=256)
            for name, c in caches.items()
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=list(caches), kv_cache_spec=make_full_spec())],
    )
    transfer = MagicMock()
    monkeypatch.setattr(base_worker, "global_te", transfer)
    worker.register_kv_caches(caches)
    assert worker.xfer_handshake_metadata.kv_caches_base_addr == [[c.data_ptr()] for c in caches.values()]
    if host_only:
        transfer.register_buffer.assert_not_called()
    else:
        transfer.register_buffer.assert_called_once_with([device.data_ptr()], [device.nbytes])


@pytest.mark.parametrize("pipeline", [False, True])
def test_v2_native_bucket_routes_mixed_destinations_and_preserves_bytes(pipeline):
    source = torch.arange(1024, dtype=torch.int64).to(torch.uint8)
    host = torch.zeros_like(source)
    hbm = torch.zeros_like(source)
    registry.register_dsa_pd_mooncake_offloaded_kv_tensor(host)
    reads = []

    def read(session, local, remote, lengths):
        reads.extend(zip(local, remote, lengths))
        for dst, src, size in zip(local, remote, lengths):
            ctypes.memmove(dst, src, size)
        return 0

    engine = MagicMock()
    engine.batch_transfer_sync_read.side_effect = read
    reader = make_reader(engine, pipeline=pipeline)
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
    registry.register_dsa_pd_mooncake_offloaded_kv_tensor(host)
    engine = MagicMock()
    reader = make_reader(engine, pipeline=True)
    event = MagicMock()
    monkeypatch.setattr(torch.npu, "Event", MagicMock(return_value=event))
    reader._copy_dsa_pd_mooncake_staging_to_offloaded_kv = MagicMock()
    if failure == "copy_exception":
        engine.batch_transfer_sync_read.return_value = 0
        reader._copy_dsa_pd_mooncake_staging_to_offloaded_kv.side_effect = [None, RuntimeError("copy failed")]
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
    registry.register_dsa_pd_mooncake_offloaded_kv_tensor(host)
    engine = MagicMock()
    engine.batch_transfer_sync_read.return_value = 0
    reader = make_reader(engine)
    landing_entered = threading.Event()
    release_landing = threading.Event()
    second_started = threading.Event()

    def land(*args):
        landing_entered.set()
        assert release_landing.wait(5)

    reader._copy_dsa_pd_mooncake_staging_to_offloaded_kv = land
    args = ("peer", [host.data_ptr()], [0xA00000], [128], "request")

    def second():
        second_started.set()
        return reader.batch_transfer_sync_read(*args)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(reader.batch_transfer_sync_read, *args)
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
    registry.register_dsa_pd_mooncake_cpu_kv_buffer(host.data_ptr(), host.nbytes)
    reader = make_reader(MagicMock())
    with pytest.raises(RuntimeError, match="Missing DSA"):
        reader.batch_transfer_sync_read("peer", [host.data_ptr()], [0xA00000], [128], "request")
    with pytest.raises(ValueError, match="equal sizes"):
        reader.batch_transfer_sync_read("peer", [1], [], [128], "request")
    reader.engine.batch_transfer_sync_read.assert_not_called()


def test_host_only_v2_consumer_preregisters_staging_before_receiver_start(monkeypatch):
    host = torch.empty(512, dtype=torch.int8)
    registry.register_dsa_pd_mooncake_offloaded_kv_tensor(host)
    order = []
    worker = pull_worker.MooncakePullConnectorWorker.__new__(pull_worker.MooncakePullConnectorWorker)
    worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
    worker.engine = MagicMock()
    worker.vllm_config = worker.kv_cache_config = MagicMock()
    worker.kv_cache_specs = [make_full_spec()]
    worker.layer_name_to_group_index = worker.layer_name_to_spec_index = {"model.layers.0.self_attn": 0}
    worker.tp_rank = worker.pp_rank = worker.dp_rank = worker.pcp_rank = worker.dcp_rank = 0
    worker.tp_size = worker.pp_size = worker.dp_size = worker.pcp_size = worker.dcp_size = 1

    def register(instance, caches):
        instance.xfer_handshake_metadata = make_transfer_metadata()

    def preregister(instance):
        order.append("staging_registered")

    def receiver(**kwargs):
        assert isinstance(kwargs["staging_reader"], DSAPDMooncakeStagingReader)
        order.append("receiver_created")
        thread = MagicMock()
        thread.start.side_effect = kwargs["ready_event"].set
        thread.is_alive.return_value = True
        return thread

    monkeypatch.setattr(base_worker.MooncakeBaseConnectorWorker, "register_kv_caches", register)
    monkeypatch.setattr(DSAPDMooncakeStagingReader, "_ensure_dsa_pd_mooncake_staging_tensor", preregister)
    monkeypatch.setattr(pull_worker, "MooncakePullRecvingThread", receiver)
    worker.register_kv_caches({"model.layers.0.self_attn": host})
    assert order == ["staging_registered", "receiver_created"]
