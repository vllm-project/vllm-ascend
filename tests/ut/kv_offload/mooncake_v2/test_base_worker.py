# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

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

from .helpers import make_full_spec, make_sliding_spec


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
    assert metadata.kv_caches_base_addr == [[k_cache.data_ptr(), v_cache.data_ptr()]]
    assert metadata.block_shapes == [[(1, 16, 8), (1, 16, 8)]]
    assert metadata.block_size_scales == [[2, 2]]
    assert metadata.block_strides == [[k_cache.stride(0) * 2, v_cache.stride(0) * 2]]
    transfer_engine.register_buffer.assert_called_once()


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
