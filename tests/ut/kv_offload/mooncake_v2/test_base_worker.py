# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import (
    MooncakeBaseConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.stats import MooncakeKVConnectorStats

from .helpers import make_full_spec, make_sfa_indexer_spec, make_sliding_spec


def test_build_spec_mappings_expands_uniform_group_by_layer_spec() -> None:
    full = make_full_spec()
    sliding = make_sliding_spec()
    uniform = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"layer.0": full, "layer.1": sliding},
    )
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.kv_cache_config = MagicMock(
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0", "layer.1"], kv_cache_spec=uniform)]
    )

    worker._build_kv_cache_spec_mappings()

    assert worker.kv_cache_specs == [full, sliding]
    assert worker.layer_name_to_group_index == {"layer.0": 0, "layer.1": 0}
    assert worker.layer_name_to_spec_index == {"layer.0": 0, "layer.1": 1}


def test_register_kv_caches_uses_config_order_and_publishes_tensor_metadata(monkeypatch) -> None:
    spec = make_full_spec()
    k_cache = torch.empty((4, 1, 16, 8), dtype=torch.float16)
    v_cache = torch.empty((4, 1, 16, 8), dtype=torch.float16)
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                size=k_cache.nbytes + v_cache.nbytes,
                shared_by=["layer.0"],
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)],
    )
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.kv_cache_config = config
    worker.engine_id = "engine-d"
    worker.te_rpc_port = 9000
    worker.block_size = 16
    worker.side_channel_host = "10.0.0.1"
    worker.handshake_port = 5000
    transfer_engine = MagicMock()
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker.global_te",
        transfer_engine,
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker.validate_register_region_count",
        MagicMock(),
    )

    worker.register_kv_caches({"layer.0": [k_cache, v_cache]})

    metadata = worker.xfer_handshake_metadata
    assert metadata is not None
    assert metadata.layer_names == ["layer.0"]
    assert metadata.layer_block_sizes == [spec.block_size]
    assert metadata.kv_caches_base_addr == [[k_cache.data_ptr(), v_cache.data_ptr()]]
    assert metadata.block_shapes == [[(1, 16, 8), (1, 16, 8)]]
    assert metadata.block_size_scales == [[2, 2]]
    assert metadata.block_strides == [[k_cache.stride(0) * 2, v_cache.stride(0) * 2]]
    transfer_engine.register_buffer.assert_called_once()


def test_register_kv_caches_publishes_sfa_indexer_virtual_block_size(monkeypatch) -> None:
    spec = make_sfa_indexer_spec(block_size=16, replication_size=2)
    cache = torch.empty((4, 16, 1, 8), dtype=torch.float16)
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[KVCacheTensor(size=cache.nbytes, shared_by=["layer.0.indexer"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0.indexer"], kv_cache_spec=spec)],
    )
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.kv_cache_config = config
    worker.engine_id = "engine-d"
    worker.te_rpc_port = 9000
    worker.block_size = 16
    worker.side_channel_host = "10.0.0.1"
    worker.handshake_port = 5000
    transfer_engine = MagicMock()
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker.global_te",
        transfer_engine,
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker.validate_register_region_count",
        MagicMock(),
    )

    worker.register_kv_caches({"layer.0.indexer": cache})

    metadata = worker.xfer_handshake_metadata
    assert metadata is not None
    assert metadata.layer_block_sizes == [32]
    assert metadata.block_size_scales == [[2]]
    assert metadata.layer_block_sizes[0] // metadata.block_size_scales[0][0] == spec.block_size


def test_register_kv_caches_rejects_missing_and_unconfigured_layers() -> None:
    spec = make_full_spec()
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[KVCacheTensor(size=64, shared_by=["layer.0"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)],
    )
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.kv_cache_config = config
    worker.engine_id = "engine"
    worker.te_rpc_port = 9000
    worker.block_size = 16
    worker.side_channel_host = "host"
    worker.handshake_port = 5000

    try:
        worker.register_kv_caches({})
    except ValueError as error:
        assert "No KV cache" in str(error)
    else:
        raise AssertionError("missing configured layer must fail")

    cache = torch.empty((2, 1, 16, 8), dtype=torch.float16)
    try:
        worker.register_kv_caches({"layer.0": cache, "unexpected": cache})
    except ValueError as error:
        assert "absent from kv_cache_tensors" in str(error)
    else:
        raise AssertionError("unexpected layer must fail")


def make_worker_config(
    *,
    is_consumer: bool = True,
    is_producer: bool = False,
    local_dp_rank: int | None = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            is_kv_consumer=is_consumer,
            is_kv_producer=is_producer,
            kv_role="kv_consumer" if is_consumer else "kv_producer",
            kv_port=6000,
        ),
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            data_parallel_rank_local=local_dp_rank,
            data_parallel_size_local=3,
            data_parallel_rank=2,
        ),
    )


def patch_worker_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pcp_size: int = 1,
) -> tuple[MagicMock, object]:
    monkeypatch.setenv("ASCEND_TRANSFER_TIMEOUT", "original")
    tp_group = object()
    transfer_engine = MagicMock()
    transfer_engine.get_rpc_port.return_value = 9000
    global_te = MagicMock()
    global_te.get_transfer_engine.return_value = transfer_engine
    patches = {
        "init_ascend_config": MagicMock(),
        "get_ascend_config": MagicMock(return_value=object()),
        "get_transfer_timeout_value": MagicMock(return_value=30),
        "get_tensor_model_parallel_rank": MagicMock(return_value=2),
        "get_tp_group": MagicMock(return_value=tp_group),
        "get_pp_group": MagicMock(return_value=SimpleNamespace(rank_in_group=1)),
        "get_pcp_group": MagicMock(return_value=SimpleNamespace(rank_in_group=0, world_size=pcp_size)),
        "get_decode_context_model_parallel_world_size": MagicMock(return_value=2),
        "get_decode_context_model_parallel_rank": MagicMock(return_value=1),
        "get_ip": MagicMock(return_value="10.0.0.2"),
        "global_te": global_te,
    }
    for name, value in patches.items():
        monkeypatch.setattr(
            f"vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker.{name}",
            value,
        )
    monkeypatch.setattr(torch.npu, "current_device", MagicMock(return_value=7))
    return global_te, tp_group


def test_base_worker_initializes_parallel_ranks_and_handshake_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_worker_config()
    kv_cache_config = SimpleNamespace(num_blocks=32)
    global_te, tp_group = patch_worker_runtime(monkeypatch)

    worker = MooncakeBaseConnectorWorker(config, "engine-d", kv_cache_config)  # type: ignore[arg-type]

    assert worker.engine_id == "engine-d"
    assert worker.block_size == 16
    assert worker.num_blocks == 32
    assert worker.tp_rank == 2
    assert worker.tp_size == 4
    assert worker.tp_group is tp_group
    assert worker.pp_rank == 1
    assert worker.pp_size == 2
    assert worker.dp_rank == 1
    assert worker.dp_size == 3
    assert worker.pcp_rank == 0
    assert worker.pcp_size == 1
    assert worker.dcp_rank == 1
    assert worker.dcp_size == 2
    assert worker.max_device_id == 24
    assert worker.side_channel_host == "10.0.0.2"
    assert worker.side_channel_port == 6016
    assert worker.handshake_port == 6022
    assert worker.te_rpc_port == 9000
    assert worker.xfer_handshake_metadata is None
    assert worker.xfer_stats.is_empty()
    global_te.get_transfer_engine.assert_called_once_with("10.0.0.2", device_name="7")


@pytest.mark.parametrize(("is_consumer", "is_producer"), [(False, False), (True, True)])
def test_base_worker_rejects_invalid_transfer_roles(
    is_consumer: bool,
    is_producer: bool,
) -> None:
    config = make_worker_config(is_consumer=is_consumer, is_producer=is_producer)

    with pytest.raises(ValueError, match="exactly one KV transfer role"):
        MooncakeBaseConnectorWorker(config, "engine", SimpleNamespace(num_blocks=1))  # type: ignore[arg-type]


def test_base_worker_requires_local_dp_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_worker_config(local_dp_rank=None)
    patch_worker_runtime(monkeypatch)

    with pytest.raises(ValueError, match="requires a local DP rank"):
        MooncakeBaseConnectorWorker(config, "engine", SimpleNamespace(num_blocks=1))  # type: ignore[arg-type]


def test_base_worker_rejects_unsupported_pcp(monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_worker_config()
    patch_worker_runtime(monkeypatch, pcp_size=2)

    with pytest.raises(AssertionError, match="prefill context parallel size 1"):
        MooncakeBaseConnectorWorker(config, "engine", SimpleNamespace(num_blocks=1))  # type: ignore[arg-type]


def test_base_worker_stats_are_returned_and_reset() -> None:
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.xfer_stats = MooncakeKVConnectorStats(data={})

    assert worker.get_kv_connector_stats() is None

    worker.xfer_stats.record_transfer(0.1, 1024)
    result = worker.get_kv_connector_stats()

    assert result is not None
    assert result.data["transfer_duration"] == [0.1]
    assert worker.xfer_stats.is_empty()


def test_base_worker_abstract_contract() -> None:
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)

    with pytest.raises(NotImplementedError):
        worker.get_finished()
    with pytest.raises(NotImplementedError):
        worker.get_block_ids_with_load_errors()
    with pytest.raises(NotImplementedError):
        worker.start_load_kv(MagicMock())
