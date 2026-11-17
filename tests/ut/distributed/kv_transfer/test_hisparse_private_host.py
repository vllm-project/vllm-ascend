# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Per-rank Offload storage keeps the original Core worker and local events."""

from types import SimpleNamespace as NS

import torch


def test_private_pool_uses_core_connector_and_event_methods(monkeypatch):
    import vllm.config as config
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import worker
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector
    from vllm.v1.hisparse import runtime

    from vllm_ascend.hisparse import cache_ops
    from vllm_ascend.hisparse.backend import AscendHiSparseBackend

    transfer = NS(
        kv_connector="MultiConnector",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "HiSparseConnector"},
                {
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_connector_extra_config": {"mooncake_transport": "memfabric"},
                },
            ]
        },
    )
    monkeypatch.setattr(config, "get_current_vllm_config_or_none", lambda: NS(kv_transfer_config=transfer))
    for name in ("_init_dma", "invalidate_blocks"):
        monkeypatch.setattr(worker.HiSparseConnectorWorker, name, getattr(worker.HiSparseConnectorWorker, name))
    monkeypatch.setattr(runtime, "allocate_pinned_host_pool", runtime.allocate_pinned_host_pool)
    monkeypatch.setattr(cache_ops, "_load_resolver", lambda: None)
    monkeypatch.setattr(torch.ops._C_ascend, "swap_blocks_batch", lambda *args: None, raising=False)
    before = [
        getattr(worker.HiSparseConnectorWorker, name)
        for name in ("start_step", "_submit_dma_descriptors", "finish_forward")
    ]
    backend = AscendHiSparseBackend()
    backend.check_available()
    assert not backend.supports_shared_host_pool
    assert [
        getattr(worker.HiSparseConnectorWorker, name)
        for name in ("start_step", "_submit_dma_descriptors", "finish_forward")
    ] == before
    assert (
        KVConnectorFactory.get_connector_class(
            KVTransferConfig(
                kv_connector="HiSparseConnector", kv_role="kv_consumer", kv_connector_extra_config={"host_pool_gib": 1}
            )
        )
        is HiSparseConnector
    )


def test_each_rank_allocates_an_independent_local_offload_pool(monkeypatch):
    import sys

    from vllm_ascend.hisparse.host_pool import allocate_mapped_host

    configs = []
    offload = NS(
        OffloadConfig=NS,
        Scene=NS(LOCAL="local"),
        OFFLOAD_FLAG_GIANT_PAGE=1,
        initialize=lambda config: configs.append(config) or 0,
        empty=lambda shape, **kwargs: torch.empty(shape, dtype=kwargs["dtype"]),
        get_dva=lambda ptr: ptr,
    )
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", NS(offload=offload))
    monkeypatch.setattr(torch, "npu", NS(current_device=lambda: 0), raising=False)
    sources = [allocate_mapped_host((64,), torch.int8, memfabric=True) for _ in range(2)]
    assert sources[0].data_ptr() != sources[1].data_ptr()
    assert all(config.scene == "local" and config.world_size == 1 and config.rank_id == 0 for config in configs)
    assert all(config.alloc_size == config.reserve_size == 1024**3 for config in configs)
    assert all(not hasattr(source._hisparse_registration, "offload_manager") for source in sources)
