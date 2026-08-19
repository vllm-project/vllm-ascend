# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

import os
import uuid

import pytest
import torch
from vllm_ascend.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    AscendECSharedRegion,
)
from vllm_ascend.utils import enable_custom_op


def test_pinned_mmap_bf16_round_trip():
    assert enable_custom_op(), "Failed to load vllm-ascend custom ops"
    assert torch.ops._C_ascend.supports_eccpu_offload(), (
        "A2 ECCPUConnector test requires aclrtMemcpyBatchAsync and aclrtHostRegisterV2(ACL_HOST_REG_PINNED)"
    )

    block_size = 64
    block_ids = [7, 2, 5, 1]
    region = AscendECSharedRegion(
        engine_id=f"ut_{os.getpid()}_{uuid.uuid4().hex}",
        num_blocks=8,
        block_size_bytes=block_size,
    )
    stream = torch.npu.Stream()
    registered = False

    try:
        torch.ops._C_ascend.register_pinned_host_mmap(region.blocks)
        registered = True
        with pytest.raises(RuntimeError, match="already registered"):
            torch.ops._C_ascend.register_pinned_host_mmap(region.blocks)

        source_cpu = torch.randn((len(block_ids), block_size // 2), dtype=torch.bfloat16)
        source = source_cpu.npu()
        restored = torch.empty_like(source)
        sizes = torch.full((len(block_ids),), block_size, dtype=torch.int64)

        d2h_src = torch.tensor(
            [source.data_ptr() + i * block_size for i in range(len(block_ids))],
            dtype=torch.int64,
        )
        d2h_dst = torch.tensor(
            [region.blocks.data_ptr() + block_id * block_size for block_id in block_ids],
            dtype=torch.int64,
        )
        with torch.npu.stream(stream):
            torch.ops._C_ascend.swap_blocks_batch(d2h_src, d2h_dst, sizes, 1)
        stream.synchronize()

        h2d_src = d2h_dst
        h2d_dst = torch.tensor(
            [restored.data_ptr() + i * block_size for i in range(len(block_ids))],
            dtype=torch.int64,
        )
        with torch.npu.stream(stream):
            torch.ops._C_ascend.swap_blocks_batch(h2d_src, h2d_dst, sizes, 0)
        stream.synchronize()

        assert torch.equal(restored.cpu(), source_cpu)
    finally:
        stream.synchronize()
        if registered:
            torch.ops._C_ascend.unregister_pinned_host_mmap(region.blocks)
        region.cleanup()
