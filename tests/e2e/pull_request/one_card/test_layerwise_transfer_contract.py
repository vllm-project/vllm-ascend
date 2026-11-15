# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real-device descriptor round trips, independent of external Store services.

These tests validate device addresses, segment sizes and cache integrity. They
do not substitute for model inference against real Memcache/Mooncake services.
"""

import ctypes

import acl
import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.distributed.ascend_store.test_layerwise_strided_roundtrip import addresses
from tests.ut.distributed.ascend_store.test_segmented_cache_transfer import make_database, make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.config import (
    parse_layerwise_prefetch_layers,
)

DEVICE_TO_HOST = 2
HOST_TO_DEVICE = 1
NUM_PHYSICAL_BLOCKS = 9
TOKENS_PER_BLOCK = 2
HEAD_SIZE = 8


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
            source_db = make_database(make_worker({"model.layers.0.attn": (sources[layer],)}, layerwise=True), 3)
            target_db = make_database(make_worker({"model.layers.0.attn": (targets[layer],)}, layerwise=True), 3)
            src_addresses, sizes, offsets = addresses(source_db, mode, 3, TOKENS_PER_BLOCK)
            dst_addresses, dst_sizes, dst_offsets = addresses(target_db, mode, 3, TOKENS_PER_BLOCK)
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
