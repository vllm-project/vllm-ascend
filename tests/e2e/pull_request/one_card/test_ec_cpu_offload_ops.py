# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

"""Single-NPU ECCPU pinned-mmap transfer test."""

import os
import uuid
from collections import deque

import pytest
import torch
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.worker import (
    ECCPUTransferDirection,
    Transfer,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.worker.descriptor_buffers import (
    DescriptorBufferPool,
)

from vllm_ascend.distributed.ec_transfer.ec_connector.cpu.worker import (
    AscendECCPUWorker,
)
from vllm_ascend.utils import enable_custom_op


def test_pinned_mmap_bf16_round_trip():
    assert enable_custom_op(), "Failed to load vllm-ascend custom ops"
    assert torch.ops._C_ascend.supports_eccpu_offload(), (
        "A2 ECCPUConnector test requires aclrtMemcpyBatchAsync and aclrtHostRegisterV2(ACL_HOST_REG_PINNED)"
    )

    block_size = 64
    block_ids = [7, 2, 5, 1]
    region = ECSharedRegion(
        engine_id=f"ut_{os.getpid()}_{uuid.uuid4().hex}",
        num_blocks=8,
        block_size_bytes=block_size,
    )
    stream = torch.npu.Stream()
    registered = False
    worker = AscendECCPUWorker.__new__(AscendECCPUWorker)
    worker._buf_pool = DescriptorBufferPool()
    worker._stream_pool = [stream]
    worker._event_pool = []

    def transfer(src_ptrs, dst_ptrs, direction, completion):
        transfer_stream = worker._acquire_stream()
        bufs = worker._buf_pool.acquire(len(src_ptrs))
        bufs.src_np[:] = src_ptrs
        bufs.dst_np[:] = dst_ptrs
        bufs.sizes_np[:] = block_size
        start_event = worker._acquire_event()
        end_event = worker._acquire_event()
        with torch.npu.stream(transfer_stream):
            start_event.record(transfer_stream)
            worker._submit_transfer(bufs, len(src_ptrs), direction)
            end_event.record(transfer_stream)
        inflight = deque(
            [
                Transfer(
                    start_event=start_event,
                    end_event=end_event,
                    completions=[completion],
                    bufs=bufs,
                    stream=transfer_stream,
                    num_bytes=len(src_ptrs) * block_size,
                )
            ]
        )
        transfer_stream.synchronize()
        assert worker._collect_finished(inflight, "test") == [completion]
        assert not inflight

    try:
        torch.ops._C_ascend.register_pinned_host_mmap(region.blocks)
        registered = True
        with pytest.raises(RuntimeError, match="already registered"):
            torch.ops._C_ascend.register_pinned_host_mmap(region.blocks)

        source_cpu = torch.randn((len(block_ids), block_size // 2), dtype=torch.bfloat16)
        source = source_cpu.npu()
        restored = torch.empty_like(source)
        d2h_src = [source.data_ptr() + i * block_size for i in range(len(block_ids))]
        d2h_dst = [region.blocks.data_ptr() + block_id * block_size for block_id in block_ids]
        transfer(
            d2h_src,
            d2h_dst,
            ECCPUTransferDirection.DEVICE_TO_HOST,
            "d2h",
        )

        h2d_dst = [restored.data_ptr() + i * block_size for i in range(len(block_ids))]
        transfer(
            d2h_dst,
            h2d_dst,
            ECCPUTransferDirection.HOST_TO_DEVICE,
            "h2d",
        )

        assert torch.equal(restored.cpu(), source_cpu)
        assert len(worker._buf_pool._pool) == 1
        assert len(worker._event_pool) == 2
    finally:
        stream.synchronize()
        if registered:
            torch.ops._C_ascend.unregister_pinned_host_mmap(region.blocks)
        region.cleanup()
