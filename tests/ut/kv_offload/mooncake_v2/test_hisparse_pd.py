# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.hisparse_pd import (
    _HiSparsePDWorker,
    _HiSparsePullThread,
    validate_hisparse_pd_config,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullConnectorWorker


def test_hisparse_pd_worker_selects_source_and_indexer():
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.num_blocks = 8
    worker.kv_cache_config = NS(
        hisparse_host_num_blocks=32,
        kv_cache_groups=[
            NS(enable_kv_transfer=True, host_resident=True),
            NS(enable_kv_transfer=True, host_resident=False),
            NS(enable_kv_transfer=False, host_resident=False),
        ],
    )
    assert [worker._is_transfer_group(i) for i in range(3)] == [True, True, False]
    assert [worker._logical_num_blocks(i) for i in range(2)] == [32, 8]


@pytest.mark.parametrize("rank", [0, 1])
def test_each_rank_imports_its_private_source_and_indexer(rank):
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.tp_rank = rank
    worker.kv_cache_config = NS(
        kv_cache_groups=[
            NS(enable_kv_transfer=True, host_resident=True),
            NS(enable_kv_transfer=True, host_resident=False),
        ]
    )
    assert [worker._is_transfer_group(i) for i in range(2)] == [True, True]


def test_hisparse_pd_worker_slices_indexer_and_keeps_host_address():
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.kv_transfer_config = NS(is_kv_consumer=True)
    worker.kv_cache_config = NS(
        kv_cache_groups=[
            NS(role="hisparse_source", layer_names=["source"], host_resident=True, enable_kv_transfer=True),
            NS(role="hisparse_indexer", layer_names=["indexer"], host_resident=False, enable_kv_transfer=True),
        ]
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


def test_pull_interleaves_mla_local_destinations_and_remote_sources():
    thread, remote = pull_layout()
    local, peer, sizes = [], [], []
    thread._append_spec_transfer_addresses(0, 0, 0, 1, 2, 1, {(0, 0): [("r", [1], [1])]}, remote, local, peer, sizes)
    assert local == [106, 108, 109, 111]
    assert peer == [1004, 2002, 1006, 2003]
    assert sizes == [2, 1, 2, 1]


def test_pull_rejects_different_tp_and_component_sizes():
    thread, remote = pull_layout()
    with pytest.raises(ValueError, match="equal P/D TP"):
        thread._append_spec_transfer_addresses(0, 0, 0, 1, 4, 1, {}, remote, [], [], [])
    remote.block_lens = [[4, 4]]
    with pytest.raises(ValueError, match="component sizes"):
        thread._append_spec_transfer_addresses(0, 0, 0, 1, 2, 1, {(0, 0): [("r", [1], [1])]}, remote, [], [], [])


def test_pull_registers_producer_sources_and_consumer_destinations():
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.kv_transfer_config = NS(is_kv_producer=False)
    with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.hisparse_pd.global_hisparse_te") as engine:
        worker._register_transfer_buffers([10], [20])
        engine.register_buffer.assert_called_once_with([10], [20])
        engine.register_buffer.reset_mock()
        worker.kv_transfer_config.is_kv_producer = True
        worker._register_transfer_buffers([10], [20])
        engine.register_buffer.assert_called_once_with([10], [20])


def test_pull_uses_hisparse_receiving_thread():
    assert _HiSparsePDWorker.receiving_thread_cls is _HiSparsePullThread


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


def test_memfabric_worker_rejects_sdma_before_engine_initialization():
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.kv_transfer_config = NS(kv_connector_extra_config={"memfabric_transfer_protocol": "sdma"})
    with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.hisparse_pd.global_hisparse_te") as engine:
        with pytest.raises(ValueError, match="SDMA is disabled"):
            worker._get_transfer_engine(None)
        engine.get_transfer_engine.assert_not_called()


def test_hisparse_pd_registers_full_host_pool_once_and_preserves_indexer():
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker._host_pool_regions = {1000: 1000}
    with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.hisparse_pd.global_hisparse_te") as engine:
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
    local, peer, sizes = [], [], []
    thread._append_spec_transfer_addresses(0, 0, 0, 1, 2, 1, {(0, 0): [("r", [1], [1])]}, remote, local, peer, sizes)
    assert local == [106, 108, 109, 111]
    assert peer == [1006, 1010, 1008, 1011]
    assert sizes == [2, 1, 2, 1]
