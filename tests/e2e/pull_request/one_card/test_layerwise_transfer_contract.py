# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real-device descriptor round trips, independent of external Store services.

These tests validate device addresses, segment sizes and cache integrity. They
do not substitute for model inference against real Memcache/Mooncake services.
"""

import ctypes
from types import SimpleNamespace

import acl  # type: ignore[import-not-found]  # Provided by CANN on the NPU host.
import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.config import (
    parse_layerwise_prefetch_layers,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import LayerBatchBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import SharedBlockData
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

DEVICE_TO_HOST = 2
HOST_TO_DEVICE = 1
NUM_PHYSICAL_BLOCKS = 9
TOKENS_PER_BLOCK = 2
HEAD_SIZE = 8
NUM_LOGICAL_BLOCKS = 3
PHYSICAL_BLOCKS_PER_CHUNK = 3


def make_database(cache):
    # Build only the address metadata; no external backend or transfer thread.
    worker = object.__new__(KVPoolWorker)
    worker.num_blocks = NUM_LOGICAL_BLOCKS
    worker.use_layerwise = True
    worker.kv_caches = {"model.layers.0.attn": (cache,)}
    worker.hf_config = SimpleNamespace(num_hidden_layers=2)
    worker.num_layers = 2
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
    worker._infer_cache_group_metadata(0, list(worker.kv_caches))
    database = ChunkedTokenDatabase([], [PHYSICAL_BLOCKS_PER_CHUNK * TOKENS_PER_BLOCK], None)
    database.set_group_buffers(
        worker.group_kv_caches_base_addr,
        worker.group_block_len,
        worker.group_block_stride,
        group_num_layers=worker.group_num_layers,
        group_layer_cache_entry_offsets=worker.group_layer_cache_entry_offsets,
        group_cache_entry_segments=worker.group_cache_entry_segments,
    )
    return database


def addresses(database, mode):
    builder = LayerBatchBuilder(database, sum(database.group_block_len[0]), database.group_num_layers["kv"][0])
    shared = SharedBlockData(
        block_ids_arr=np.array([1], dtype=np.int64),
        block_gvas_arr=np.array([4096], dtype=np.int64) if mode == "gva" else None,
        req_ids=["npu-roundtrip"],
        is_last_chunks=[True],
        block_keys=["npu-roundtrip-key"] if mode == "block_key" else None,
    )
    result = builder.build_addrs(shared, 0)
    if mode == "gva":
        return result.addr_array.tolist(), result.size_array.tolist(), result.gvas_array.tolist()
    return result.all_buffers[0], result.all_sizes[0], result.all_offsets[0]


@pytest.mark.parametrize("mode", ["gva", "block_key"])
@pytest.mark.parametrize("prefetch", [1, "2", 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_layerwise_descriptors_roundtrip_real_npu_memory(mode, prefetch, dtype):
    window = parse_layerwise_prefetch_layers({"layerwise_prefetch_layers": prefetch}, 2)
    cpu = torch.arange(NUM_PHYSICAL_BLOCKS * TOKENS_PER_BLOCK * HEAD_SIZE, dtype=torch.float32)
    source = cpu.reshape(NUM_PHYSICAL_BLOCKS, TOKENS_PER_BLOCK, HEAD_SIZE).to(dtype=dtype, device="npu")
    restored = torch.full_like(source, -1)
    # Retain per-layer storage so each prefetch window operates on live buffers.
    sources = [source.clone() + layer for layer in range(3)]
    targets = [restored.clone() for _ in sources]
    torch.npu.synchronize()
    for first in range(0, len(sources), window):
        for layer in range(first, min(first + window, len(sources))):
            source_db = make_database(sources[layer])
            target_db = make_database(targets[layer])
            src_addresses, sizes, offsets = addresses(source_db, mode)
            dst_addresses, dst_sizes, dst_offsets = addresses(target_db, mode)
            assert sizes == dst_sizes
            assert offsets == dst_offsets
            for src, dst, size in zip(src_addresses, dst_addresses, sizes):
                host = ctypes.create_string_buffer(size)
                host_address = ctypes.addressof(host)
                assert acl.rt.memcpy(host_address, size, src, size, DEVICE_TO_HOST) == 0
                assert acl.rt.memcpy(dst, size, host_address, size, HOST_TO_DEVICE) == 0
    torch.npu.synchronize()
    for original, actual in zip(sources, targets):
        expected = torch.full_like(original.cpu(), -1)
        expected[3:6].copy_(original.cpu()[3:6])
        assert torch.equal(actual.cpu(), expected), "Unselected NPU blocks must retain their guard values"
