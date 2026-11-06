# SPDX-License-Identifier: Apache-2.0
"""Byte-level transport and worker wiring checks for Mooncake SFA pull."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.mooncake_backend import (
    STAGING_ALIGNMENT_BYTES,
    MooncakePullBackend,
    build_mooncake_consumer_backend,
    get_device_cache_regions,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import (
    SFAPDRD2HConsumerWorker,
    SFAPDRD2HProducerWorker,
    _get_pull_engine,
    _resolve_kv_transfer_backend,
)

_MODULE = "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.mooncake_backend"
_TE = "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine.global_te"
_WORKER = "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker"
_REMOTE = 0x10000
_HOST = 0x20000
_DEVICE = 0x30000
_STAGING = 0x40000


def _byte_backend(*, capacity=7, fail_call=None, fail_landing=False):
    remote = bytes(range(128))
    host, device, staging = bytearray(128), bytearray(128), bytearray(capacity)
    calls, order = [], []

    def read(session, locals_, peers, sizes):
        calls.append((session, locals_, peers, sizes))
        order.append("read")
        if len(calls) == fail_call:
            return 17
        for local, peer, size in zip(locals_, peers, sizes):
            payload = remote[peer - _REMOTE : peer - _REMOTE + size]
            if local == _STAGING:
                staging[:size] = payload
            else:
                assert _DEVICE <= local < _DEVICE + len(device)
                device[local - _DEVICE : local - _DEVICE + size] = payload
        return 0

    def land(destination, source, size):
        order.append("land")
        assert source == _STAGING
        if fail_landing:
            raise RuntimeError("local copy failed")
        host[destination - _HOST : destination - _HOST + size] = staging[:size]

    backend = MooncakePullBackend(
        SimpleNamespace(batch_transfer_sync_read=read),
        host_regions=[(_HOST, len(host))],
        device_regions=[(_DEVICE, len(device))],
        staging_ptr=_STAGING,
        staging_bytes=capacity,
        land=land,
    )
    return backend, remote, host, device, calls, order


@pytest.mark.parametrize("length", [1, 7, 8, 33, 128])
def test_host_reads_preserve_bytes_and_reuse_bounded_staging(length):
    backend, remote, host, _, calls, order = _byte_backend()
    assert backend.batch_transfer_sync_read("p:1234", [_HOST], [_REMOTE], [length]) == 0
    assert host[:length] == remote[:length]
    assert all(c[1] == [_STAGING] and c[3][0] <= 7 for c in calls)
    assert sum(c[3][0] for c in calls) == length
    assert order == [step for _ in calls for step in ("read", "land")]


def test_mixed_main_indexer_and_tail_destinations_are_routed_correctly():
    backend, remote, host, device, calls, order = _byte_backend()
    assert (
        backend.batch_transfer_sync_read(
            "p:1234", [_HOST + 10, _DEVICE + 4, _DEVICE + 20], [_REMOTE + 30, _REMOTE + 60, _REMOTE + 80], [19, 8, 5]
        )
        == 0
    )
    assert host[10:29] == remote[30:49]
    assert device[4:12] == remote[60:68]
    assert device[20:25] == remote[80:85]
    assert calls[-1] == ("p:1234", [_DEVICE + 4, _DEVICE + 20], [_REMOTE + 60, _REMOTE + 80], [8, 5])
    assert order == ["read", "land"] * 3 + ["read"]


@pytest.mark.parametrize(
    "locals_,peers,sizes",
    [
        ([_HOST], [], [1]),
        ([_HOST], [_REMOTE], [-1]),
        ([0], [_REMOTE], [1]),
        ([_HOST], [0], [1]),
        ([_HOST, _HOST + 127], [_REMOTE, _REMOTE], [1, 2]),
        ([_DEVICE + 128], [_REMOTE], [1]),
    ],
)
def test_invalid_whole_plan_is_rejected_before_any_copy(locals_, peers, sizes):
    backend, _, _, _, calls, order = _byte_backend()
    with pytest.raises(ValueError):
        backend.batch_transfer_sync_read("p:1234", locals_, peers, sizes)
    assert calls == order == []


def test_empty_and_zero_length_plans_do_not_contact_peer():
    backend, _, _, _, calls, _ = _byte_backend()
    assert backend.batch_transfer_sync_read("p:1234", [], [], []) == 0
    assert backend.batch_transfer_sync_read("p:1234", [_HOST], [_REMOTE], [0]) == 0
    assert calls == []


def test_failed_network_chunk_is_not_landed_or_reported_as_success():
    backend, remote, host, _, calls, order = _byte_backend(fail_call=2)
    assert backend.batch_transfer_sync_read("p:1234", [_HOST], [_REMOTE], [20]) == 17
    assert len(calls) == 2
    assert host[:7] == remote[:7]
    assert host[7:20] == bytes(13)
    assert order == ["read", "land", "read"]


def test_failed_local_copy_aborts_before_staging_window_is_reused():
    backend, _, _, _, calls, _ = _byte_backend(fail_landing=True)
    with pytest.raises(RuntimeError, match="local copy failed"):
        backend.batch_transfer_sync_read("p:1234", [_HOST], [_REMOTE], [20])
    assert len(calls) == 1


def test_failed_device_read_is_returned_without_a_landing_copy():
    backend, _, _, _, calls, order = _byte_backend(fail_call=1)
    assert backend.batch_transfer_sync_read("p:1234", [_DEVICE], [_REMOTE], [20]) == 17
    assert len(calls) == 1
    assert order == ["read"]


def test_host_pool_addresses_are_excluded_from_device_registration():
    host = torch.empty(8)
    device = MagicMock(spec=torch.Tensor)
    device.device = SimpleNamespace(type="npu")
    device.numel.return_value = 8
    collector = "vllm_ascend.distributed.kv_transfer.utils.utils.collect_storage_merged_register_regions"
    with patch(collector) as collect:
        get_device_cache_regions({"layer": (host, [device, host])})
    collect.assert_called_once_with({"layer": (device,)})


@pytest.mark.parametrize("capacity", [0, -1, 1, 2.5, True])
def test_invalid_staging_configuration_fails_before_registration(capacity):
    with patch(_TE + ".register_buffer") as register, pytest.raises(ValueError, match="mooncake_staging_bytes"):
        build_mooncake_consumer_backend(
            MagicMock(), manager=MagicMock(), kv_caches={}, num_blocks=1, staging_bytes=capacity
        )
    register.assert_not_called()


def test_runtime_landing_uses_shared_pool_copy_and_synchronizes_before_return():
    manager = SimpleNamespace(
        gvas_k_bases=[_HOST], gvas_v_bases=[_HOST + 64], cpu_block_lens=[(32, 32)], copy_sfa_kv=MagicMock()
    )
    device_regions = SimpleNamespace(ptrs=[_DEVICE], lengths=[128], count=1)
    raw = MagicMock()
    raw.batch_transfer_sync_read.return_value = 0
    stream = MagicMock()
    real_device = torch.device

    def cpu_device(*args, **kwargs):
        return real_device("cpu")

    with (
        patch("torch.device", side_effect=cpu_device),
        patch("torch.npu.current_device", return_value=0),
        patch("torch.npu.Stream", return_value=stream),
        patch(_MODULE + ".get_device_cache_regions", return_value=device_regions),
        patch("vllm_ascend.distributed.kv_transfer.utils.utils.validate_register_region_count"),
        patch(_TE + ".register_buffer") as register,
    ):
        backend = build_mooncake_consumer_backend(
            raw, manager=manager, kv_caches={}, num_blocks=2, staging_bytes=STAGING_ALIGNMENT_BYTES
        )
        assert backend.batch_transfer_sync_read("p:1234", [_HOST + 8], [_REMOTE], [12]) == 0
    assert register.call_count == 2
    ptrs, sizes = register.call_args.args
    assert ptrs[0] % STAGING_ALIGNMENT_BYTES == 0
    assert sizes == [STAGING_ALIGNMENT_BYTES]
    src, dst, lengths, count = manager.copy_sfa_kv.call_args.args
    assert src.tolist() == ptrs
    assert dst.tolist() == [_HOST + 8]
    assert lengths.tolist() == [12]
    assert count.tolist() == [1]
    stream.wait_stream.assert_called_once()
    stream.synchronize.assert_called_once()


def _config(backend):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"transfer_backend": backend}),
    )


def test_mooncake_selection_initializes_only_the_mooncake_transport():
    with patch(_TE + ".get_transfer_engine") as get_engine, patch(_WORKER + ".global_memfabric_te") as memfabric:
        assert _get_pull_engine(_config("mooncake"), "10.1.1.1", "Decode") is get_engine.return_value
    get_engine.assert_called_once_with("10.1.1.1", device_name=None)
    memfabric.configure.assert_not_called()
    memfabric.get_transfer_engine.assert_not_called()


def test_unknown_backend_is_rejected_and_producer_advertises_mooncake_rpc_session():
    with pytest.raises(ValueError, match="transfer_backend"):
        _resolve_kv_transfer_backend(_config("invalid"))
    worker = object.__new__(SFAPDRD2HProducerWorker)
    worker._backend = "mooncake"
    worker.side_channel_host = "10.1.1.1"
    worker.te_rpc_port = 4321
    worker.last_layer_idx = 0
    worker.layer_metadata = {}
    worker.main_group_idx = 0
    worker.indexer_group_idx = 1
    worker.block_size = [128, 128]
    worker.layer_storage_slots = {}
    worker.pp_rank = 0
    worker.pp_size = 1
    assert worker._build_producer_send_state().p_session == "10.1.1.1:4321"


def test_consumer_installs_staging_adapter_on_a_non_owner_tp_rank():
    name = "model.layers.0.self_attn"
    worker = object.__new__(SFAPDRD2HConsumerWorker)
    worker.tp_rank = 1
    worker.vllm_config = _config("mooncake")
    worker.vllm_config.kv_transfer_config.kv_connector_extra_config["mooncake_staging_bytes"] = STAGING_ALIGNMENT_BYTES
    worker.kv_cache_config = SimpleNamespace(num_blocks=2, kv_cache_groups=[SimpleNamespace(layer_names=[name])])
    worker.offload_manager = SimpleNamespace(
        offload_layer_names=[name],
        k_caches_cpu=[],
        v_caches_cpu=[],
        gvas_k_bases=[_HOST],
        gvas_v_bases=[_HOST + 64],
        cpu_block_lens=[(32, 32)],
    )
    worker.dspark_draft_layer_names = ()
    worker.layer_metadata = {}
    worker.engine = MagicMock()
    worker.side_channel_port = 12000
    worker._ensure_engine = MagicMock()
    worker._bind_copy_sfa_tail_destinations = MagicMock()
    worker._build_consumer_read_state = MagicMock()
    caches = {name: ()}
    with (
        patch(_WORKER + ".build_mooncake_consumer_backend") as build,
        patch(_WORKER + ".MembPullReadThread") as read_thread,
    ):
        read_thread.return_value.startup_error = None
        worker._register_memfabric_pull(caches)
    assert worker._cpu_pools == [None]
    assert worker._main_gva_bases == [(_HOST, _HOST + 64)]
    assert build.call_args.kwargs == {
        "manager": worker.offload_manager,
        "kv_caches": caches,
        "num_blocks": 2,
        "staging_bytes": STAGING_ALIGNMENT_BYTES,
    }
    assert worker.engine is build.return_value
    assert read_thread.call_args.kwargs["engine"] is build.return_value
    read_thread.return_value.start.assert_called_once()


def test_mooncake_producer_rejects_host_sources_before_transport_registration():
    name = "model.layers.0.self_attn"
    worker = object.__new__(SFAPDRD2HProducerWorker)
    worker._backend = "mooncake"
    worker.kv_cache_config = SimpleNamespace(num_blocks=2, kv_cache_groups=[SimpleNamespace(layer_names=[name])])
    worker.dspark_draft_layer_names = ()
    with patch(_TE + ".register_buffer") as register, pytest.raises(ValueError, match="must reside in HBM"):
        worker.register_kv_caches({name: (torch.empty(2, 4), torch.empty(2, 4))})
    register.assert_not_called()
