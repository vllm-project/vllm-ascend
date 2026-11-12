# SPDX-License-Identifier: Apache-2.0
"""Local KV identities bypass address lookup while preserving routing safety."""

import ctypes
import gc
import threading
import weakref
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import transfer_targets as registry_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.worker import MooncakeSFAPullRecvingThread

from .mooncake_sfa_helpers import make_thread
from .mooncake_v2.helpers import make_full_spec, make_mamba_spec, make_pp_metadata, make_transfer_metadata
from .test_dsa_pd_mooncake_connector_v2 import make_reader

registry = registry_module.MooncakeKVRegistry()


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    registry.clear()
    monkeypatch.setattr(torch.npu, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.npu, "Event", MagicMock())
    monkeypatch.setattr(torch.npu, "synchronize", MagicMock())
    yield
    registry.clear()


def forbid_address_queries(monkeypatch, reader=None):
    if reader is not None:
        # Isolate reader address discovery from the Host adapter's separate
        # landing range guard, which must still validate the original registry.
        resolve = registry.resolve_offloaded_tensor
        monkeypatch.setattr(reader.host_kv, "contains", lambda ptr, size=1: resolve(ptr, size) is not None)
    for name in ["resolve_offloaded_tensor", "resolve_offloaded_byte_tensor", "get_register_location"]:
        monkeypatch.setattr(
            registry,
            name,
            MagicMock(side_effect=AssertionError("per-address registry query in metadata path")),
        )


def test_local_descriptors_classify_host_and_direct_and_preserve_view_offsets(monkeypatch):
    host = torch.zeros(128, dtype=torch.int16)
    direct = torch.zeros(32, dtype=torch.int8)
    registry.register_offloaded_tensor(host)
    view = host[4:32]
    targets = registry.create_transfer_targets([[view], [direct]])
    assert targets[0][0].offloaded and not targets[1][0].offloaded
    forbid_address_queries(monkeypatch)
    prepared = registry.prepare_transfer_targets(
        [targets[0][0], targets[1][0]], [view.data_ptr() + 6, direct.data_ptr()], [8, 16]
    )
    assert prepared[0][1] == 14 and prepared[1] is None
    prepared[0][0][14:22].fill_(1)
    assert torch.all(host.view(torch.int8)[14:22] == 1)


@pytest.mark.parametrize("ptr_offset,length", [(-1, 1), (120, 9), (0, -1)])
def test_declared_tensor_bounds_are_checked_without_queries(ptr_offset, length):
    tensor = torch.empty(128, dtype=torch.int8)
    targets = registry.create_transfer_targets([[tensor]])[0]
    with pytest.raises(ValueError, match="boundary"):
        registry.prepare_transfer_targets(targets, [tensor.data_ptr() + ptr_offset], [length])


def test_strided_direct_tensor_bounds_use_physical_span():
    storage = torch.empty(256, dtype=torch.int8)
    tensor = storage[::2]
    target = registry.create_transfer_targets([[tensor]])[0][0]
    assert target.nbytes == 255
    assert registry.prepare_transfer_targets([target], [tensor.data_ptr() + 254], [1]) == [None]
    with pytest.raises(ValueError, match="boundary"):
        registry.prepare_transfer_targets([target], [tensor.data_ptr() + 254], [2])


@pytest.mark.parametrize("change", ["reset", "cpu_registration", "tensor_registration"])
def test_layout_changes_invalidate_descriptors(change):
    tensor = torch.empty(128, dtype=torch.int8)
    target = registry.create_transfer_targets([[tensor]])[0]
    other = torch.empty(128, dtype=torch.int8)
    if change == "reset":
        registry.clear()
    elif change == "cpu_registration":
        registry.register_cpu_buffer(other.data_ptr(), other.nbytes)
    else:
        registry.register_offloaded_tensor(other)
    with pytest.raises(RuntimeError, match="Stale"):
        registry.prepare_transfer_targets(target, [tensor.data_ptr()], [16])


def test_duplicate_registration_does_not_invalidate_descriptors():
    tensor = torch.empty(128, dtype=torch.int8)
    registry.register_cpu_buffer(tensor.data_ptr(), tensor.nbytes)
    registry.register_offloaded_tensor(tensor)
    target = registry.create_transfer_targets([[tensor]])[0]
    registry.register_cpu_buffer(tensor.data_ptr(), tensor.nbytes)
    registry.register_offloaded_tensor(tensor)
    assert registry.prepare_transfer_targets(target, [tensor.data_ptr()], [16])[0][1] == 0


@pytest.mark.parametrize("offloaded", [False, True])
def test_descriptors_do_not_own_tensor_lifetime(offloaded):
    tensor = torch.empty(128, dtype=torch.int8)
    if offloaded:
        registry.register_offloaded_tensor(tensor)
    ptr = tensor.data_ptr()
    target = registry.create_transfer_targets([[tensor]])[0]
    registry.prepare_transfer_targets(target, [ptr], [16])
    tensor_ref = weakref.ref(tensor)
    del tensor
    gc.collect()
    assert tensor_ref() is None
    with pytest.raises(RuntimeError, match="released"):
        registry.prepare_transfer_targets(target, [ptr], [16])


def test_overlapping_aliases_and_partial_registrations_keep_generic_path():
    storage = torch.empty(128, dtype=torch.int8)
    small, large = storage[16:32], storage[:]
    registry.register_offloaded_tensor(small)
    assert registry.create_transfer_targets([[large]]) is None
    registry.register_offloaded_tensor(large)
    assert registry.create_transfer_targets([[large]]) is None
    assert registry.resolve_offloaded_tensor(small.data_ptr(), 4)[0] is small


def test_missing_host_registration_keeps_cpu_guard_path():
    tensor = torch.empty(128, dtype=torch.int8)
    registry.register_cpu_buffer(tensor.data_ptr(), tensor.nbytes)
    assert registry.create_transfer_targets([[tensor]]) is None
    reader = make_reader(MagicMock(), kv_registry=registry)
    with pytest.raises(RuntimeError, match="Missing.*Host READ"):
        reader.batch_transfer_sync_read("peer", [tensor.data_ptr()], [1024], [16], "missing-host")
    reader.engine.batch_transfer_sync_read.assert_not_called()


def test_descriptor_count_and_local_metadata_mismatches():
    tensor = torch.empty(128, dtype=torch.int8)
    reader = make_reader(MagicMock(), kv_registry=registry)
    reader.kv_caches = {"layer": tensor}
    with pytest.raises(ValueError, match="layer names"):
        reader.get_transfer_targets(["layer"], [])
    with pytest.raises(ValueError, match="identity"):
        reader.get_transfer_targets(["layer"], [[tensor.data_ptr() + 1]])
    with pytest.raises(ValueError, match="count"):
        reader.batch_transfer_sync_read("peer", [tensor.data_ptr()], [1024], [16], "bad", transfer_targets=[])
    with pytest.raises(ValueError, match="count"):
        registry.prepare_transfer_targets([], [tensor.data_ptr()], [16])


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("host_first", [False, True])
def test_native_bucket_carries_targets_and_copies_bytes_without_queries(monkeypatch, pipeline, host_first):
    source = torch.arange(1024, dtype=torch.int64).to(torch.uint8)
    host, direct = torch.zeros_like(source), torch.zeros_like(source)
    registry.register_offloaded_tensor(host)
    engine = MagicMock()
    reader = make_reader(engine, pipeline=pipeline, kv_registry=registry)
    ordered = [host, direct] if host_first else [direct, host]
    names = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    reader.kv_caches = dict(zip(names, ordered))
    bases = [[tensor.data_ptr()] for tensor in ordered]
    targets = reader.get_transfer_targets(names, bases)
    thread = make_thread(
        engine=engine,
        staging_reader=reader,
        layer_names=names,
        kv_caches_base_addr=bases,
        block_strides=[[128], [128]],
        block_lens=[[128], [128]],
        block_shapes=[[(1, 16, 4)], [(1, 16, 4)]],
        block_size_scales=[[1], [1]],
        _transfer_targets=targets,
    )
    remote = make_pp_metadata(layer_names=names, tp_base_addrs={0: [[source.data_ptr()], [source.data_ptr()]]})
    entries = {0: {(0, 0): [("request", [1, 2], [3, 4])], (1, 1): [("request", [1, 2], [3, 4])]}}

    def read(_session, local, remote, lengths):
        assert reader._dsa_pd_mooncake_staging_lock.locked()
        for dst, src, size in zip(local, remote, lengths):
            assert not host.data_ptr() <= dst < host.data_ptr() + host.nbytes
            ctypes.memmove(dst, src, size)
        return 0

    engine.batch_transfer_sync_read.side_effect = read
    forbid_address_queries(monkeypatch, reader)
    thread._execute_tp_transfer_bucket(0, 0, 0, 1, 1, 1, remote, entries)
    assert torch.equal(host[128:384], source[384:640])
    assert torch.equal(direct[128:384], source[384:640])
    assert torch.count_nonzero(host[:128]) == 0 and torch.count_nonzero(host[384:]) == 0
    assert not reader._dsa_pd_mooncake_staging_lock.locked()


def test_receiver_initialization_binds_layer_cache_identities():
    host = torch.empty(128, dtype=torch.int8)
    registry.register_offloaded_tensor(host)
    metadata = replace(make_transfer_metadata(), kv_caches_base_addr=[[host.data_ptr()]])
    reader = make_reader(MagicMock(), kv_registry=registry)
    reader.kv_caches = {metadata.layer_names[0]: host}
    thread = MooncakeSFAPullRecvingThread(
        engine=reader.engine,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_mla=False, hf_text_config=SimpleNamespace()),
            speculative_config=None,
        ),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
        kv_cache_specs=[make_full_spec()],
        layer_name_to_group_index={metadata.layer_names[0]: 0},
        layer_name_to_spec_index={metadata.layer_names[0]: 0},
        local_metadata=metadata,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_rank=0,
        dp_size=1,
        pcp_rank=0,
        pcp_size=1,
        dcp_rank=0,
        dcp_size=1,
        device="cpu",
        ready_event=threading.Event(),
        staging_reader=reader,
    )
    try:
        assert thread._transfer_targets[0][0].offloaded
        assert thread._transfer_targets[0][0].base_ptr == host.data_ptr()
    finally:
        thread.executor.shutdown()


def test_direct_only_metadata_avoids_staging_lock_and_queries(monkeypatch):
    tensor = torch.empty(128, dtype=torch.int8)
    reader = make_reader(MagicMock(), kv_registry=registry)
    targets = registry.create_transfer_targets([[tensor]])[0]
    reader.engine.batch_transfer_sync_read.side_effect = lambda *_: int(reader._dsa_pd_mooncake_staging_lock.locked())
    forbid_address_queries(monkeypatch, reader)
    assert (
        reader.batch_transfer_sync_read("peer", [tensor.data_ptr()], [1024], [16], "direct", transfer_targets=targets)
        == 0
    )


def test_stale_metadata_is_rejected_before_read_and_releases_lock():
    host = torch.empty(128, dtype=torch.int8)
    registry.register_offloaded_tensor(host)
    targets = registry.create_transfer_targets([[host]])[0]
    reader = make_reader(MagicMock(), kv_registry=registry)
    registry.clear()
    with pytest.raises(RuntimeError, match="Stale"):
        reader.batch_transfer_sync_read("peer", [host.data_ptr()], [1024], [16], "stale", transfer_targets=targets)
    reader.engine.batch_transfer_sync_read.assert_not_called()
    assert not reader._dsa_pd_mooncake_staging_lock.locked()


@pytest.mark.parametrize("failure", ["negative", "read_exception", "copy_exception"])
def test_tagged_pipeline_drains_on_failure(monkeypatch, failure):
    chunk = 2 * 1024 * 1024
    host = torch.empty(3 * chunk, dtype=torch.int8)
    registry.register_offloaded_tensor(host)
    targets = registry.create_transfer_targets([[host]])[0]
    reader = make_reader(MagicMock(), pipeline=True, window=2, kv_registry=registry)
    event = MagicMock()
    monkeypatch.setattr(torch.npu, "Event", MagicMock(return_value=event))
    reader.host_kv.manager.copy_sfa_kv = MagicMock()
    if failure == "copy_exception":
        reader.engine.batch_transfer_sync_read.return_value = 0
        reader.host_kv.manager.copy_sfa_kv.side_effect = [None, RuntimeError("copy failed")]
    else:
        reader.engine.batch_transfer_sync_read.side_effect = [
            0,
            -1 if failure == "negative" else RuntimeError("read failed"),
        ]
    forbid_address_queries(monkeypatch, reader)
    if failure == "negative":
        assert (
            reader.batch_transfer_sync_read(
                "peer", [host.data_ptr()], [10 * chunk], [host.nbytes], "failed", transfer_targets=targets
            )
            == -1
        )
    else:
        with pytest.raises(RuntimeError, match="failed"):
            reader.batch_transfer_sync_read(
                "peer", [host.data_ptr()], [10 * chunk], [host.nbytes], "failed", transfer_targets=targets
            )
    event.synchronize.assert_called()
    assert not reader._dsa_pd_mooncake_staging_lock.locked()


def test_mamba_bucket_keeps_legacy_reader_contract():
    reader = MagicMock()
    reader.batch_transfer_sync_read.return_value = 0
    thread = make_thread(
        engine=MagicMock(), staging_reader=reader, kv_cache_specs=[make_mamba_spec()], _transfer_targets=[[object()]]
    )
    thread._append_mamba_transfer_addresses = MagicMock(
        side_effect=lambda *args: (args[-3].append(1000), args[-2].append(2000), args[-1].append(16))
    )
    thread._execute_tp_transfer_bucket(0, 0, 0, 1, 1, 1, make_pp_metadata(), {0: {(0, 0): [("request", [1], [2])]}})
    reader.batch_transfer_sync_read.assert_called_once_with("10.0.0.1:9000", [1000], [2000], [16], "request")


def test_interleaved_destinations_use_identity_instead_of_input_position(monkeypatch):
    tensors = [torch.zeros(64, dtype=torch.int8) for _ in range(6)]
    for tensor in tensors[1::2]:
        registry.register_offloaded_tensor(tensor)
    targets = [row[0] for row in registry.create_transfer_targets([[t] for t in tensors])]
    source = torch.arange(64, dtype=torch.int8)
    reader = make_reader(MagicMock(), pipeline=True, kv_registry=registry)

    def read(_session, local, remote, lengths):
        for dst, src, size in zip(local, remote, lengths):
            ctypes.memmove(dst, src, size)
        return 0

    reader.engine.batch_transfer_sync_read.side_effect = read
    forbid_address_queries(monkeypatch, reader)
    assert (
        reader.batch_transfer_sync_read(
            "peer",
            [t.data_ptr() for t in tensors],
            [source.data_ptr()] * 6,
            [64] * 6,
            "interleaved",
            transfer_targets=targets,
        )
        == 0
    )
    assert all(torch.equal(tensor, source) for tensor in tensors)


def test_empty_tagged_receive_performs_no_read_or_query(monkeypatch):
    reader = make_reader(MagicMock(), kv_registry=registry)
    forbid_address_queries(monkeypatch, reader)
    assert reader.batch_transfer_sync_read("peer", [], [], [], "empty", transfer_targets=[]) == 0
    reader.engine.batch_transfer_sync_read.assert_not_called()
