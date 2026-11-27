# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Own the shared Host lifetime without replacing Core worker methods."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend.hisparse import host_pool


@pytest.fixture
def core_hisparse_backend():
    return pytest.importorskip("vllm.v1.hisparse.backend")


@pytest.fixture
def config():
    return NS(
        kv_transfer_config=NS(
            kv_connector="MultiConnector",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "HiSparseConnector"},
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_consumer",
                        "kv_connector_extra_config": {"transfer_backend": "memfabric"},
                    },
                ]
            },
        ),
        parallel_config=NS(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            world_size=2,
            distributed_executor_backend="mp",
            nnodes_within_dp=1,
        ),
    )


@pytest.mark.usefixtures("core_hisparse_backend")
def test_registered_connector_factory_does_not_replace_core_methods(monkeypatch):
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import worker
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    from vllm_ascend.distributed.kv_transfer import register_connector

    before = (
        worker._create_hisparse_host_events,
        worker.HiSparseConnectorWorker.start_step,
        worker.HiSparseConnectorWorker.finish_forward,
    )
    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    KVConnectorFactory.register_connector(
        "HiSparseConnector", "vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector", "HiSparseConnector"
    )
    monkeypatch.setattr(OffloadingSpecFactory, "_registry", {})
    register_connector()
    cls = KVConnectorFactory.get_connector_class(
        KVTransferConfig(
            kv_connector="HiSparseConnector",
            kv_role="kv_consumer",
            kv_connector_extra_config={"host_pool_gib": 1},
        )
    )
    assert cls.__module__ == "vllm_ascend.hisparse.connector"
    assert (
        worker._create_hisparse_host_events,
        worker.HiSparseConnectorWorker.start_step,
        worker.HiSparseConnectorWorker.finish_forward,
    ) == before


@pytest.mark.parametrize("tp_size,rank", [(1, 0), (2, 0), (2, 1)])
def test_manager_shared_source_retains_one_backing(monkeypatch, tp_size, rank):
    import vllm.distributed as distributed

    from vllm_ascend.hisparse.host_pool import AscendHiSparseOffloadManager

    backing = torch.full((256,), -1, dtype=torch.int8)
    registration = NS(tensor=backing, nbytes=256)
    backing._hisparse_registration = registration
    region = NS(base_tensor=backing)
    tp_group = NS(rank_in_group=rank, world_size=tp_size)
    expected_group = tp_group if tp_size > 1 else None

    def allocate(sizes, *, group):
        assert sizes == [64] and group is expected_group
        return region

    monkeypatch.setattr(host_pool, "MemFabricSharedHostRegion", allocate)
    monkeypatch.setattr(distributed, "get_tp_group", lambda: tp_group)
    manager = AscendHiSparseOffloadManager(64, shared=tp_size > 1)
    assert manager.shared_region is (region if tp_size > 1 else None)
    assert manager.is_host_writer is (rank == 0)
    assert manager.source.data_ptr() == backing.data_ptr() and manager.source.numel() == 64
    assert manager.source._hisparse_registration is registration
    assert registration.offload_manager is manager


@pytest.mark.usefixtures("core_hisparse_backend")
def test_existing_private_allocator_does_not_replace_core_shared_allocator(monkeypatch):
    from vllm.v1.hisparse import runtime

    shared = runtime.allocate_hisparse_host_pools
    monkeypatch.setattr(runtime.HiSparseHostPool, "allocate", runtime.HiSparseHostPool.allocate)
    monkeypatch.setattr(runtime, "allocate_pinned_host_pool", runtime.allocate_pinned_host_pool)
    host_pool.install()
    assert runtime.allocate_hisparse_host_pools is shared


@pytest.mark.parametrize(
    "connector, tp_size, shared",
    [
        ("HiSparseConnector", 2, True),
        ("HiSparseConnector", 1, False),
        ("MultiConnector", 2, True),
        ("MultiConnector", 1, False),
    ],
)
def test_core_pool_plan_selects_memfabric_allocator(
    monkeypatch, config, core_hisparse_backend, connector, tp_size, shared
):
    import vllm.config as core_config
    import vllm.distributed as distributed
    from vllm.v1.hisparse import layout, runtime
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from vllm_ascend.hisparse.backend import AscendHiSparseBackend
    from vllm_ascend.hisparse.host_pool import AscendHiSparseOffloadManager

    config.kv_transfer_config.kv_connector = connector
    config.parallel_config.tensor_parallel_size = config.parallel_config.world_size = tp_size
    config.model_config = NS(max_model_len=128)
    backend = AscendHiSparseBackend()
    monkeypatch.setattr(core_config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(core_hisparse_backend, "get_hisparse_backend", lambda: backend)
    monkeypatch.setattr(runtime, "get_hisparse_backend", lambda: backend)
    alignment = runtime.SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    page_bytes = alignment + alignment // 2
    source_group = NS(host_resident=True, kv_cache_spec=NS(page_size_bytes=page_bytes, block_size=128))
    plan = layout.create_hisparse_layout(config, [source_group], host_budget=7 * alignment)
    assert plan.shared_host_pool is shared
    assert plan.host_block_stride == (2 * alignment if shared else page_bytes)
    assert plan.host_num_blocks == (3 if shared else 4)

    backing = torch.full((256,), -1, dtype=torch.int8)
    backing._hisparse_registration = NS(tensor=backing, nbytes=256)
    region = NS(base_tensor=backing)
    tp_group = NS(rank_in_group=0)

    def allocate_shared(sizes, *, group):
        assert shared or connector == "MultiConnector"
        assert sizes == [64]
        assert group is (tp_group if shared else None)
        return region

    def allocate_private(shape, dtype, *, memfabric):
        assert connector == "HiSparseConnector"
        assert shape == (64,) and dtype == torch.int8 and not memfabric
        return backing

    monkeypatch.setattr(host_pool, "MemFabricSharedHostRegion", allocate_shared)
    monkeypatch.setattr(host_pool, "allocate_mapped_host", allocate_private)
    monkeypatch.setattr(runtime, "check_hisparse_host_memory", lambda size: None)
    monkeypatch.setattr(distributed, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(runtime.HiSparseHostPool, "allocate", runtime.HiSparseHostPool.allocate)
    monkeypatch.setattr(runtime, "allocate_pinned_host_pool", runtime.allocate_pinned_host_pool)
    host_pool.install()
    cache_config = KVCacheConfig(
        num_blocks=4, kv_cache_tensors=[], kv_cache_groups=[], hisparse_shared_host_pool=plan.shared_host_pool
    )
    pool = runtime.HiSparseHostPool(config, cache_config)
    source = pool.allocate(64)
    if shared or connector == "MultiConnector":
        manager = AscendHiSparseOffloadManager.from_host_pool(pool)
        assert manager.shared_region is pool.shared_region is (region if shared else None)
    else:
        assert pool.shared_region is None
        assert not hasattr(pool.registered._hisparse_registration, "offload_manager")
    assert cache_config.hisparse_shared_host_pool is shared
    assert pool.allocate(64) is source
    assert pool.registered.data_ptr() == backing.data_ptr()
    assert pool.registered.numel() == 256 and source.numel() == 64
    source.fill_(7)
    assert backing[:64].eq(7).all() and backing[64:].eq(-1).all()


def test_manager_rejects_foreign_host_pool():
    from vllm_ascend.hisparse.host_pool import AscendHiSparseOffloadManager

    with pytest.raises(RuntimeError, match="manager-owned"):
        AscendHiSparseOffloadManager.from_host_pool(NS(registered=torch.empty(1)))


@pytest.mark.usefixtures("core_hisparse_backend")
def test_memfabric_requires_hisparse_connector(monkeypatch, config):
    import vllm.config as core_config

    from vllm_ascend.hisparse.backend import AscendHiSparseBackend

    config.kv_transfer_config.kv_connector_extra_config["connectors"].pop(0)
    monkeypatch.setattr(core_config, "get_current_vllm_config_or_none", lambda: config)
    with pytest.raises(ValueError, match="requires Ascend's HiSparseConnector"):
        AscendHiSparseBackend().check_available()


@pytest.mark.usefixtures("core_hisparse_backend")
def test_external_connector_constructs_own_worker(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorRole
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import worker as core_worker

    from vllm_ascend.hisparse.connector import HiSparseConnector
    from vllm_ascend.hisparse.worker import AscendHiSparseConnectorWorker

    monkeypatch.setattr(KVConnectorBase_V1, "__init__", lambda *args: None)
    monkeypatch.setattr(core_worker, "get_hisparse_backend", lambda: object())
    cache_config = NS(hisparse_host_num_blocks=64)
    connector = HiSparseConnector(NS(), KVConnectorRole.WORKER, cache_config)
    assert type(connector.connector_worker) is AscendHiSparseConnectorWorker
    assert connector.connector_worker.kv_cache_config is cache_config
    assert connector.connector_scheduler is None


@pytest.mark.usefixtures("core_hisparse_backend")
def test_manager_backend_check_preserves_all_core_worker_entries(monkeypatch, config):
    import vllm.config as core_config
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import worker
    from vllm.v1.hisparse import runtime

    from vllm_ascend.hisparse import cache_ops
    from vllm_ascend.hisparse.backend import AscendHiSparseBackend

    names = ("start_step", "finish_forward", "_submit_dma_descriptors", "_init_dma", "invalidate_blocks")
    before = [getattr(worker.HiSparseConnectorWorker, name) for name in names]
    events, shared_allocator = worker._create_hisparse_host_events, runtime.allocate_hisparse_host_pools
    monkeypatch.setattr(core_config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(cache_ops, "_load_resolver", lambda: None)
    monkeypatch.setattr(torch.ops._C_ascend, "swap_blocks_batch", lambda *args: None, raising=False)
    monkeypatch.setattr(runtime.HiSparseHostPool, "allocate", runtime.HiSparseHostPool.allocate)
    monkeypatch.setattr(runtime, "allocate_pinned_host_pool", runtime.allocate_pinned_host_pool)
    AscendHiSparseBackend().check_available()
    assert [getattr(worker.HiSparseConnectorWorker, name) for name in names] == before
    assert worker._create_hisparse_host_events is events
    assert runtime.allocate_hisparse_host_pools is shared_allocator


def test_hisparse_pd_requires_hisparse_before_mooncake(config):
    config.kv_transfer_config.kv_connector_extra_config["connectors"].reverse()
    with pytest.raises(ValueError, match="must precede MooncakeConnectorV2"):
        host_pool.validate_hisparse_pd_order(config)
