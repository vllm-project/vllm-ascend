# SPDX-License-Identifier: Apache-2.0
import ctypes
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.distributed.kv_transfer.kv_p2p.hisparse_pd import (
    _HiSparsePDWorker,
    _HiSparsePullThread,
    validate_hisparse_pd_config,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import pull_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullConnectorWorker

from .helpers import make_full_spec


@pytest.mark.parametrize("shared,rank,expected", [(True, 0, [0, 1]), (True, 1, [1]), (False, 1, [0, 1])])
def test_shared_host_is_imported_once_while_each_rank_keeps_indexer(monkeypatch, shared, rank, expected):
    source = torch.zeros((5, 1, 16, 8), dtype=torch.float16)
    indexer = torch.zeros((2, 1, 16, 8), dtype=torch.float16)
    spec = make_full_spec()
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.kv_transfer_config = NS(is_kv_consumer=True)
    worker.tp_rank, worker.tp_size, worker.pcp_rank = rank, 2, 0
    worker.vllm_config = NS(
        parallel_config=NS(cp_kv_cache_interleave_size=16),
        compilation_config=NS(static_forward_context={"indexer": NS()}),
    )
    worker.ascend_config = NS(kvpp_config=NS(size=1))
    worker.engine_id, worker.te_rpc_port = "consumer", 9000
    worker.block_size, worker.side_channel_host, worker.handshake_port = 16, "127.0.0.1", 5000
    config = KVCacheConfig(
        num_blocks=2,
        hisparse_host_num_blocks=5,
        hisparse_shared_host_pool=shared,
        kv_cache_tensors=[
            KVCacheTensor(source.nbytes, ["source"], source.nbytes, spec.page_size_bytes, host_resident=True),
            KVCacheTensor(indexer.nbytes, ["indexer"], indexer.nbytes, spec.page_size_bytes),
            KVCacheTensor(indexer.nbytes, ["local_only"], indexer.nbytes, spec.page_size_bytes),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["source"], spec, host_resident=True),
            KVCacheGroupSpec(["indexer"], spec, role="hisparse_indexer"),
            KVCacheGroupSpec(["local_only"], spec, enable_kv_transfer=False),
        ],
    )
    worker.kv_cache_config = config
    engine, thread = MagicMock(), MagicMock()
    monkeypatch.setattr(pull_worker, "global_memfabric_te", engine)

    def build_thread(thread_cls, ready_event):
        assert thread_cls is _HiSparsePullThread
        assert worker.xfer_handshake_metadata.block_size_scales == [[1]] * len(expected)
        thread.start.side_effect = ready_event.set
        return thread

    monkeypatch.setattr(worker, "_build_recving_thread", build_thread)
    worker.register_kv_caches({"source": source, "indexer": indexer})
    assert worker.xfer_handshake_metadata.group_indices == expected
    assert len(engine.register_buffer.call_args.args[0]) == len(expected)
    assert worker.kv_cache_config is config


def test_hisparse_pd_worker_slices_indexer_and_keeps_host_address():
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.kv_transfer_config = NS(is_kv_consumer=True)
    spec = make_full_spec()
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=2,
        hisparse_host_num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["source"], spec, host_resident=True, role="hisparse_source"),
            KVCacheGroupSpec(["indexer"], spec, role="hisparse_indexer"),
        ],
    )
    worker.vllm_config = NS(compilation_config=NS(static_forward_context={"indexer": NS(hisparse_block_stride=2)}))
    indexer = torch.arange(16).reshape(4, 4)
    source_backing = torch.zeros(128, dtype=torch.uint8)
    source = source_backing[32:48].view(4, 4)
    with patch.object(MooncakePullConnectorWorker, "register_kv_caches") as register:
        worker.register_kv_caches({"source": source, "indexer": indexer})
    assert torch.equal(register.call_args.args[0]["indexer"][0], indexer[::2])
    assert register.call_args.args[0]["source"].data_ptr() == source.data_ptr()
    assert worker._host_pool_regions == {source_backing.data_ptr(): source_backing.nbytes}


def pull_layout():
    thread = _HiSparsePullThread.__new__(_HiSparsePullThread)
    thread.tp_size = 2
    thread.pcp_size = 1
    thread.group_indices = [0]
    thread.kv_cache_config = NS(kv_cache_groups=[NS(host_resident=True)])
    thread.kv_cache_specs = [NS(block_size=2)]
    thread.kv_caches_base_addr = [[100]]
    thread.block_lens = [[6]]
    thread.block_strides = [[6]]
    peer = NS(kv_caches_base_addr=[[1000, 2000]])
    remote = NS(
        block_lens=[[4, 2]], block_strides=[[4, 2]], metadata_by_pcp_rank={0: NS(metadata_by_tp_rank={0: peer})}
    )
    return thread, remote


def test_pull_reads_split_mla_components_into_one_host_page():
    thread, remote = pull_layout()
    local: list[int] = []
    peer: list[int] = []
    sizes: list[int] = []
    thread._append_spec_transfer_addresses(
        0, 0, 0, 1, 2, 1, {(0, 0): [("r", [1, 3], [1, 4])]}, remote, local, peer, sizes
    )
    assert local == [106, 110, 118, 122]
    assert peer == [1004, 2002, 1016, 2008]
    assert sizes == [4, 2, 4, 2]


@pytest.mark.parametrize("tokens,nope_bytes,rope_bytes", [(2, 2, 1), (128, 512, 144)])
@pytest.mark.parametrize("duplicate_page", [False, True])
def test_pull_reorders_split_mla_page_before_bucket_completes(tokens, nope_bytes, rope_bytes, duplicate_page):
    thread, remote = pull_layout()
    page_bytes = tokens * (nope_bytes + rope_bytes)
    nope_page_bytes = tokens * nope_bytes
    rope_page_bytes = tokens * rope_bytes
    local_pages = ctypes.create_string_buffer(2 * page_bytes)
    nope_data = bytes(i % 251 for i in range(2 * nope_page_bytes))
    rope_data = bytes((i + 17) % 251 for i in range(2 * rope_page_bytes))
    remote_nope = ctypes.create_string_buffer(nope_data)
    remote_rope = ctypes.create_string_buffer(rope_data)
    thread.kv_cache_specs[0].block_size = tokens
    thread.kv_caches_base_addr = [[ctypes.addressof(local_pages)]]
    thread.block_lens = [[page_bytes]]
    thread.block_strides = [[page_bytes]]
    remote.block_lens = [[nope_page_bytes, rope_page_bytes]]
    remote.block_strides = [[nope_page_bytes, rope_page_bytes]]
    peer = remote.metadata_by_pcp_rank[0].metadata_by_tp_rank[0]
    peer.kv_caches_base_addr = [[ctypes.addressof(remote_nope), ctypes.addressof(remote_rope)]]
    peer.local_ip, peer.te_rpc_port = "127.0.0.1", 9000
    thread.local_metadata = NS(local_ip="127.0.0.1")
    thread.tp_rank = 0

    def read(_session, local, remote_addrs, sizes):
        assert sizes == [nope_page_bytes, rope_page_bytes] * (3 if duplicate_page else 2)
        for dst, src, size in zip(local, remote_addrs, sizes, strict=True):
            ctypes.memmove(dst, src, size)
        return 0

    thread.engine = NS(batch_transfer_sync_read=read)
    blocks = [0, 1, 0] if duplicate_page else [0, 1]
    thread._execute_tp_transfer_bucket(0, 0, 0, 1, 2, 1, remote, {0: {(0, 0): [("req", blocks, blocks)]}})
    expected = b"".join(
        nope_data[page * nope_page_bytes + token * nope_bytes : page * nope_page_bytes + (token + 1) * nope_bytes]
        + rope_data[page * rope_page_bytes + token * rope_bytes : page * rope_page_bytes + (token + 1) * rope_bytes]
        for page in range(2)
        for token in range(tokens)
    )
    assert local_pages.raw[: 2 * page_bytes] == expected


def test_pull_reads_one_descriptor_per_packed_mla_page():
    thread, remote = pull_layout()
    thread.block_strides = [[8]]
    remote.block_lens = [[6]]
    remote.block_strides = [[10]]
    remote.block_shapes = [[(2, 3)]]
    remote.metadata_by_pcp_rank[0].metadata_by_tp_rank[0].kv_caches_base_addr = [[1000]]
    local: list[int] = []
    peer: list[int] = []
    sizes: list[int] = []
    thread._append_spec_transfer_addresses(
        0, 0, 0, 1, 2, 1, {(0, 0): [("r", [1, 3], [2, 5])]}, remote, local, peer, sizes
    )

    assert local == [108, 124]
    assert peer == [1020, 1050]
    assert sizes == [6, 6]


def test_pull_uses_selected_prefill_pcp_rank():
    thread, remote = pull_layout()
    remote.metadata_by_pcp_rank[1] = NS(metadata_by_tp_rank={0: NS(kv_caches_base_addr=[[3000, 4000]])})
    local: list[int] = []
    peer: list[int] = []
    sizes: list[int] = []
    thread._append_spec_transfer_addresses(0, 1, 0, 2, 2, 1, {(0, 0): [("r", [1], [1])]}, remote, local, peer, sizes)
    assert local == [106, 110]
    assert peer == [3004, 4002]
    assert sizes == [4, 2]


def test_pull_rejects_different_tp_and_component_sizes():
    thread, remote = pull_layout()
    with pytest.raises(ValueError, match="equal P/D TP"):
        thread._append_spec_transfer_addresses(0, 0, 0, 1, 4, 1, {}, remote, [], [], [])
    remote.block_lens = [[4, 4]]
    with pytest.raises(ValueError, match="component sizes"):
        thread._append_spec_transfer_addresses(0, 0, 0, 1, 2, 1, {(0, 0): [("r", [1], [1])]}, remote, [], [], [])


def test_pull_rejects_decode_pcp():
    thread, remote = pull_layout()
    thread.pcp_size = 2
    with pytest.raises(ValueError, match="D PCP=1"):
        thread._append_spec_transfer_addresses(0, 0, 0, 1, 2, 1, {}, remote, [], [], [])


def test_pull_registers_producer_sources_and_consumer_destinations():
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.kv_transfer_config = NS(is_kv_producer=False)
    with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.global_memfabric_te") as engine:
        worker._register_transfer_buffers([10], [20])
        engine.register_buffer.assert_called_once_with([10], [20])
        engine.register_buffer.reset_mock()
        worker.kv_transfer_config.is_kv_producer = True
        worker._register_transfer_buffers([10], [20])
        engine.register_buffer.assert_called_once_with([10], [20])


def test_memfabric_layout_validation_distinguishes_p_and_d():
    cache = NS(hisparse_host_num_blocks=None)
    validate_hisparse_pd_config(NS(is_kv_consumer=False), cache)
    with pytest.raises(ValueError, match="requires HiSparse Host Source"):
        validate_hisparse_pd_config(NS(is_kv_consumer=True), cache)
    cache.hisparse_host_num_blocks = 32
    cache.kv_cache_groups = [
        NS(enable_kv_transfer=True, host_resident=True, role="hisparse_source"),
        NS(enable_kv_transfer=True, host_resident=False, role="hisparse_indexer"),
    ]
    validate_hisparse_pd_config(NS(is_kv_consumer=True), cache)
    with pytest.raises(ValueError, match="ordinary Prefill KV"):
        validate_hisparse_pd_config(NS(is_kv_consumer=False), cache)
    cache.kv_cache_groups.pop()
    with pytest.raises(ValueError, match="transfer groups"):
        validate_hisparse_pd_config(NS(is_kv_consumer=True), cache)


@pytest.mark.parametrize(
    "extra,protocol",
    [
        ({}, "device_urma"),
        ({"memfabric_transfer_protocol": "device_rdma"}, "device_rdma"),
        ({"memfabric_transfer_protocol": "sdma"}, "sdma"),
    ],
)
@pytest.mark.parametrize("is_consumer,role", [(False, "Prefill"), (True, "Decode")])
def test_memfabric_worker_configures_direct_engine(monkeypatch, extra, protocol, is_consumer, role):
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.kv_transfer_config = NS(kv_connector_extra_config=extra, is_kv_consumer=is_consumer)
    worker.side_channel_host = "192.0.2.1"
    monkeypatch.setattr(torch.npu, "current_device", lambda: 3)
    with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.global_memfabric_te") as engine:
        assert worker._get_transfer_engine(None) is engine.get_transfer_engine.return_value
        engine.configure.assert_called_once_with(role=role, device_id=3, transfer_protocol=protocol)
        engine.get_transfer_engine.assert_called_once_with("192.0.2.1")


def test_hisparse_pd_registers_full_host_pool_once_and_preserves_indexer():
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker._host_pool_regions = {1000: 1000}
    with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.global_memfabric_te") as engine:
        worker._register_transfer_buffers([1100, 1500, 3000], [100, 200, 400])
        engine.register_buffer.assert_called_once_with([1000, 3000], [1000, 400])


def test_hisparse_pd_rejects_registration_crossing_host_pool_boundary():
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker._host_pool_regions = {1000: 1000}
    with pytest.raises(ValueError, match="exceeds its Host pool"):
        worker._register_transfer_buffers([1900], [101])


def test_pull_unpacks_flat_component_major_mla_slot():
    thread, remote = pull_layout()
    thread.model_config = NS(hf_text_config=NS(kv_lora_rank=2, qk_rope_head_dim=1))
    thread.kv_cache_specs[0].dtype = torch.uint8
    remote.block_lens = [[6]]
    remote.block_strides = [[6]]
    remote.block_shapes = [[(6,)]]
    remote.metadata_by_pcp_rank[0].metadata_by_tp_rank[0].kv_caches_base_addr = [[1000]]
    local: list[int] = []
    peer: list[int] = []
    sizes: list[int] = []
    thread._append_spec_transfer_addresses(0, 0, 0, 1, 2, 1, {(0, 0): [("r", [1], [1])]}, remote, local, peer, sizes)
    assert local == [106, 110]
    assert peer == [1006, 1010]
    assert sizes == [4, 2]
