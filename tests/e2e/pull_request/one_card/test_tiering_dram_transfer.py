# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Raw A3 SDMA round trips, including the 16-bit FFTS context boundary."""

import mmap

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend.vllm_ascend_C")


@pytest.mark.parametrize("copies", [17, 65536])
def test_tiering_sdma_scatter_round_trip(copies):
    if not int(torch.ops._C_ascend.kv_offload_dram_capabilities()) & 2:
        pytest.skip("A3 FFTS SDMA was not enabled in this build")
    page_size = 64
    size = copies * page_size
    allocation = ((size + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
    mapping = mmap.mmap(-1, allocation)
    host = torch.frombuffer(mapping, dtype=torch.uint8)
    memory = None
    batches = []
    try:
        memory = torch.classes._C_ascend.KVOffloadHostMemory(host, True)
        reference = (torch.arange(size, dtype=torch.int64) % 251).to(torch.uint8).view(copies, page_size)
        host[:size].copy_(reference.flatten())
        device = torch.empty((copies, page_size), dtype=torch.uint8, device="npu")
        offsets = torch.arange(copies, dtype=torch.int64) * page_size
        sources = offsets + host.data_ptr()
        destinations = offsets.flip(0) + device.data_ptr()
        sizes = torch.full((copies,), page_size, dtype=torch.int64)
        copy_stream = torch.npu.Stream()
        with torch.npu.stream(copy_stream):
            batch = torch.classes._C_ascend.KVOffloadSdmaBatch(memory, sources, destinations, sizes, 0)
            batches.append(batch)
            batch.launch()
            complete = torch.npu.Event()
            complete.record()
        complete.synchronize()
        batch.release()
        assert torch.equal(device.cpu(), reference.flip(0))

        device.fill_(73)
        host.zero_()
        copy_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(copy_stream):
            batch = torch.classes._C_ascend.KVOffloadSdmaBatch(memory, destinations, sources, sizes, 1)
            batches.append(batch)
            batch.launch()
            complete.record()
        complete.synchronize()
        batch.release()
        assert torch.all(host[:size] == 73)
    finally:
        torch.npu.synchronize()
        for batch in batches:
            batch.release()
        if memory is not None:
            memory.close()
        del host
        mapping.close()
