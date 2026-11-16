# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real Store range I/O; set VLLM_ASCEND_LAYERWISE_TEST_BACKEND to opt in.

Start the selected service and provide its normal MMC_* / MOONCAKE_CONFIG_PATH
configuration. These tests use real NPU buffers and production backend/address
builders. They cover transfer layouts, not model attention kernels or multi-rank
execution. A missing service is an error when the backend is explicitly selected.
"""

import os
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import ParallelConfig

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.config import (
    parse_layerwise_prefetch_layers,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import LayerBatchBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import SharedBlockData
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

BACKEND = os.getenv("VLLM_ASCEND_LAYERWISE_TEST_BACKEND", "")
pytestmark = pytest.mark.skipif(not BACKEND, reason="requires an explicitly configured real Store service")


@pytest.fixture(scope="module")
def backend():
    torch.npu.set_device(0)
    if BACKEND == "memcache":
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend

        store = MemcacheBackend(ParallelConfig(), device_id=0)
    elif BACKEND == "mooncake":
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

        store = MooncakeBackend(ParallelConfig())
        store.validate_layerwise_support()
    else:
        raise ValueError(f"Unknown VLLM_ASCEND_LAYERWISE_TEST_BACKEND={BACKEND!r}")
    yield store
    if BACKEND == "memcache":
        store.close()


def make_caches(layout, dtype):
    groups = [["model.layers.0.attn", "model.layers.1.attn", "model.layers.2.attn"]]
    if layout == "hybrid_mtp":
        groups = [
            ["model.layers.0.attn", "model.layers.2.attn", "model.mtp.0.attn"],
            ["model.layers.1.attn", "model.layers.2.attn", "model.layers.2.indexer", "model.mtp.0.attn"],
        ]
    caches, backing = {}, []
    for group, names in enumerate(groups):
        for index, name in enumerate(names):
            components = []
            for component in range(2 if layout == "hybrid_mtp" else 1):
                width = 4 + group * 2 + component
                padding = 7 if layout != "dense" else 0
                values = torch.arange(9 * (2 * width + padding), dtype=torch.float32)
                values = values.reshape(9, 2 * width + padding) + group * 71 + index * 13 + component * 5
                source = values.to(device="npu", dtype=dtype)
                target = torch.full_like(source, -1)
                components.append((source[:, : 2 * width].view(9, 2, width), target[:, : 2 * width].view(9, 2, width)))
                backing.append((source, target, 2 * width))
            caches[(group, name)] = components
    return groups, caches, backing


def make_database(groups, caches, side):
    worker = object.__new__(KVPoolWorker)
    worker.num_blocks = 3
    worker.use_layerwise = True
    worker.hf_config = SimpleNamespace(num_hidden_layers=3)
    worker.num_layers = 3
    worker._global_to_local_layer = {}
    for name in (
        "group_kv_caches_base_addr",
        "group_block_len",
        "group_block_stride",
        "group_layer_cache_entry_offsets",
        "group_cache_entry_segments",
        "group_num_layers",
    ):
        setattr(worker, name, {})
    for group, names in enumerate(groups):
        worker.kv_caches = {name: tuple(pair[side] for pair in caches[(group, name)]) for name in names}
        worker._infer_cache_group_metadata(group, names)
    database = ChunkedTokenDatabase([], [6] * len(groups), None)
    database.set_group_buffers(
        worker.group_kv_caches_base_addr,
        worker.group_block_len,
        worker.group_block_stride,
        group_num_layers=worker.group_num_layers,
        group_layer_cache_entry_offsets=worker.group_layer_cache_entry_offsets,
        group_cache_entry_segments=worker.group_cache_entry_segments,
    )
    return database


@pytest.mark.parametrize("layout", ["dense", "strided", "hybrid_mtp"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("prefetch", [1, "2", 8])
def test_real_backend_layer_ranges_preserve_cache_guards(backend, layout, dtype, prefetch):
    window = parse_layerwise_prefetch_layers({"layerwise_prefetch_layers": prefetch}, 2)
    groups, caches, backing = make_caches(layout, dtype)
    source = make_database(groups, caches, 0)
    target = make_database(groups, caches, 1)
    buffers = [tensor for pair in backing for tensor in pair[:2]]
    pointers = [tensor.data_ptr() for tensor in buffers]
    lengths = [tensor.numel() * tensor.element_size() for tensor in buffers]
    torch.npu.synchronize()
    backend.register_buffer(pointers, lengths)
    try:
        for group in range(len(groups)):
            keys = [f"layerwise-contract-{uuid.uuid4().hex}" for _ in range(2)]
            object_size = sum(source.group_block_len[group])
            if BACKEND == "memcache":
                gvas = backend.batch_alloc(keys, [object_size] * 2, 30000)
                assert all(0 < address < 2**64 - 1 for address in gvas)
            else:
                gvas = None
                assert backend.batch_put_start(keys, [object_size] * 2) == [0, 0]
            shared = SharedBlockData(
                block_ids_arr=np.array([0, 2], dtype=np.int64),
                block_gvas_arr=np.array(gvas, dtype=np.int64) if gvas is not None else None,
                block_keys=None if gvas is not None else keys,
                req_ids=["real-backend-roundtrip"],
                is_last_chunks=[True],
            )
            num_layers = source.group_num_layers["kv"][group]
            for database, saving in ((source, True), (target, False)):
                if not saving:
                    assert backend.batch_is_readable(keys) == [True, True]
                    if BACKEND == "memcache":
                        assert backend.batch_add_lease(keys, 30000) == [0, 0]
                    else:
                        assert backend.batch_get_start(keys) == [0, 0]
                builder = LayerBatchBuilder(database, object_size, num_layers, group_id=group)
                for first in range(0, num_layers, window):
                    for layer in range(first, min(first + window, num_layers)):
                        request = builder.build_addrs(shared, layer)
                        if BACKEND == "memcache":
                            result = backend.store.batch_copy(
                                request.gvas_array.tolist(),
                                request.addr_array.tolist(),
                                request.size_array.tolist(),
                                0 if saving else 1,
                            )
                            assert result == 0
                        else:
                            operation = backend.batch_copy_put if saving else backend.batch_copy_get
                            result = operation(keys, request.all_buffers, request.all_sizes, request.all_offsets)
                            assert result == [sum(sizes) for sizes in request.all_sizes]
                if saving:
                    if BACKEND == "memcache":
                        assert backend.batch_write_finish(keys, [0, 0]) == [0, 0]
                    else:
                        assert backend.batch_commit(keys) == [0, 0]
                elif BACKEND == "memcache":
                    assert backend.batch_remove_lease(keys) == 0
                else:
                    assert backend.batch_get_end(keys) == 0
        torch.npu.synchronize()
        for original, actual, width in backing:
            expected = torch.full_like(original.cpu(), -1)
            expected[:3, :width] = original.cpu()[:3, :width]
            expected[6:9, :width] = original.cpu()[6:9, :width]
            assert torch.equal(actual.cpu(), expected), "untouched blocks and physical-page padding must retain guards"
    finally:
        if BACKEND == "memcache":
            backend.unregister_buffer(pointers, lengths)
        else:
            from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

            global_te.unregister_buffer()
