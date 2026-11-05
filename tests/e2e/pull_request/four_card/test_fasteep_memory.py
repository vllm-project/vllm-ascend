# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real pooled allocations, discard, peer refill and stable tensor addresses."""

import socket
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.distributed.fasteep.hccl import RetainedCommunicators
from vllm_ascend.distributed.fasteep.memory import PeerMemory


def _worker(rank, port):
    torch.npu.set_device(rank)
    torch.empty(1, device=f"npu:{rank}")
    torch.npu.synchronize()
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4, timeout=timedelta(seconds=90)
    )
    parent = SimpleNamespace(
        rank_in_group=rank, world_size=4, ranks=list(range(4)), local_rank=rank, cpu_group=dist.group.WORLD
    )
    owner = RetainedCommunicators(parent)
    allocator = CaMemAllocator.get_instance()
    with allocator.use_memory_pool("weights"):
        model = torch.nn.Linear(1024, 1024, bias=False, dtype=torch.bfloat16, device=f"npu:{rank}")
        model.weight.data.fill_(rank % 2 + 1)
        model.register_buffer("metadata", torch.full((16,), rank, device=f"npu:{rank}"))
    with allocator.use_memory_pool("kv_cache"):
        cache = torch.ones((1024, 1024), device=f"npu:{rank}", dtype=torch.bfloat16)
    state = PeerMemory(model)
    addresses = (model.weight.data_ptr(), model.metadata.data_ptr(), cache.data_ptr())
    try:
        for _ in range(2):
            if rank >= 2:
                state.sleep()
                assert state.sleeping
                assert all(
                    d.is_asleep and d.cpu_backup_tensor is None
                    for d in allocator.pointer_to_data.values()
                    if d.tag in ("weights", "kv_cache")
                )
            owner.barrier()
            if rank >= 2:
                state.map()
            manifests = owner.gather_objects(state.metadata())
            assert manifests[rank] == manifests[rank % 2]
            for receiver in (2, 3):
                donor = receiver - 2
                if rank in (donor, receiver):
                    for storage in state.storages():
                        owner.transfer(
                            storage["pointer"], storage["bytes"], receiver if rank == donor else donor, rank == donor
                        )
            torch.npu.synchronize()
            if rank >= 2:
                state.finish()
                cache.fill_(7)
                torch.testing.assert_close(cache, torch.full_like(cache, 7))
            torch.testing.assert_close(model.weight, torch.full_like(model.weight, rank % 2 + 1))
            torch.testing.assert_close(model.metadata, torch.full_like(model.metadata, rank))
            assert addresses == (model.weight.data_ptr(), model.metadata.data_ptr(), cache.data_ptr())
            owner.barrier()
    finally:
        owner.close()
        dist.destroy_process_group()


@pytest.mark.skipif(not hasattr(torch, "npu") or torch.npu.device_count() < 4, reason="Requires four NPUs")
def test_peer_memory_refill():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=4, join=True)
