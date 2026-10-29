# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Native discard/remap and shutdown while a weights pool is asleep."""

import gc

import pytest
import torch

from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.distributed.fasteep.memory import PeerMemory


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="Requires an NPU")
def test_discard_remap_and_sleeping_cleanup():
    torch.npu.set_device(0)
    allocator = CaMemAllocator.get_instance()
    with allocator.use_memory_pool("weights"):
        model = torch.nn.Linear(1024, 1024, bias=False, device="npu", dtype=torch.bfloat16)
        model.register_buffer("metadata", torch.arange(16, device="npu"))
    state = PeerMemory(model)
    pointer = model.weight.data_ptr()
    for _ in range(2):
        state.sleep()
        assert state.sleeping
        assert all(data.cpu_backup_tensor is None for data in allocator.pointer_to_data.values())
        state.map()
        # The separate peer-transfer test checks how values arrive. Here write
        # into the actual remapped allocation and verify its local lifecycle.
        model.weight.data.fill_(3)
        state.finish()
        assert model.weight.data_ptr() == pointer
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 3))
        torch.testing.assert_close(model.metadata, torch.arange(16, device="npu"))
    state.sleep()
    del state, model
    gc.collect()
    torch.npu.empty_cache()
    allocator.release_pools()
