# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Non-replica expert migration with captured graph and memory recovery."""

import socket
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.distributed.fasteep.communication import ReclaimableCommunicators
from vllm_ascend.distributed.fasteep.memory import PeerMemory
from vllm_ascend.distributed.fasteep.migration import ExpertMigration
from vllm_ascend.distributed.fasteep.placement import plan_active_placement


def _exercise(rank, port, host="127.0.0.1", device=None, world_size=4):
    device = rank if device is None else device
    torch.npu.set_device(device)
    torch.empty(1, device=f"npu:{device}")
    torch.npu.synchronize()
    dist.init_process_group(
        "gloo", init_method=f"tcp://{host}:{port}", rank=rank, world_size=world_size, timeout=timedelta(seconds=180)
    )
    parent = SimpleNamespace(
        rank_in_group=rank,
        world_size=world_size,
        ranks=list(range(world_size)),
        local_rank=device,
        cpu_group=dist.group.WORLD,
    )
    comms = ReclaimableCommunicators(parent)
    allocator = CaMemAllocator.get_instance()
    # No two ranks have identical shards. EP4 -> EP2 must redistribute experts.
    template = [[0, 1, -1, -1], [0, -1, 1, -1], [-1, 0, -1, 1], [-1, -1, 0, 1]][:world_size]
    with allocator.use_memory_pool("weights"):
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(torch.empty((2, 256, 256), device="npu", dtype=torch.bfloat16))
        layer.w2_weight = torch.nn.Parameter(torch.empty_like(layer.w13_weight))
        for expert, slot in enumerate(template[rank]):
            if slot >= 0:
                layer.w13_weight.data[slot].fill_(expert + 1)
                layer.w2_weight.data[slot].fill_(expert + 10)
    memory = PeerMemory(layer)
    parameter_addresses = [p.data_ptr() for p in layer.parameters()]
    graph = torch.npu.NPUGraph()
    with torch.inference_mode():
        for _ in range(3):
            _ = layer.w13_weight.float().sum(dim=(1, 2)) + layer.w2_weight.float().sum(dim=(1, 2))
        torch.npu.synchronize()
        with allocator.capture_graph_memory() as pool, torch.npu.graph(graph, pool=pool):
            output = layer.w13_weight.float().sum(dim=(1, 2)) + layer.w2_weight.float().sum(dim=(1, 2))
    current = [template]
    active = world_size
    observations = []
    try:
        for target in (2, 3, 4, 2, 4) if world_size == 4 else (2, 3, 2, 3):
            released = 0
            comms.begin_transition()
            after = [plan_active_placement(current[0], template, target)]
            if target > active:
                if active <= rank < target:
                    memory.map()
                for joining in range(active, target):
                    if rank in (0, joining):
                        for storage in memory.storages():
                            comms.transfer(storage["pointer"], storage["bytes"], joining if rank == 0 else 0, rank == 0)
                torch.npu.synchronize()
                if active <= rank < target:
                    memory.finish()
                comms.barrier()
            with torch.inference_mode():
                ExpertMigration([layer], rank, comms, verify=True).execute(current, after)
            group = comms.activate(target)
            assert comms.root is None
            if rank >= target:
                assert comms.active_native is None
                assert comms.last_memory_delta > 0
            if target <= rank < active:
                graph_bytes = sum(d.handle[1] for d in allocator.pointer_to_data.values() if d.tag == "graphs")
                assert graph_bytes > 0
                free_before = torch.npu.mem_get_info()[0]
                memory.sleep()
                released = torch.npu.mem_get_info()[0] - free_before
                assert released >= graph_bytes
            if rank < target:
                metadata = torch.tensor([rank + 1], dtype=torch.int64)
                assert group.cpu_all_reduce_(metadata) is metadata
                assert metadata.item() == target * (target + 1) // 2
                graph.replay()
                torch.npu.synchronize()
                expected = torch.empty(2, dtype=torch.float32)
                for expert, slot in enumerate(after[0][rank]):
                    if slot >= 0:
                        expected[slot] = (2 * expert + 11) * 256 * 256
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
                value = torch.full((16,), rank + 1, device="npu", dtype=torch.float32)
                torch.testing.assert_close(group.all_reduce(value), torch.full_like(value, target * (target + 1) // 2))
                sends = [(rank + peer) % 3 for peer in range(target)]
                receives = [(peer + rank) % 3 for peer in range(target)]
                source = torch.cat(
                    [
                        torch.full((count, 2), rank * 100 + peer, device="npu", dtype=torch.float32)
                        for peer, count in enumerate(sends)
                    ]
                )
                received = torch.empty((sum(receives), 2), device="npu", dtype=torch.float32)
                group.all_to_all(received, source, receives, sends)
                expected_exchange = torch.cat(
                    [
                        torch.full((count, 2), peer * 100 + rank, dtype=torch.float32)
                        for peer, count in enumerate(receives)
                    ]
                )
                torch.testing.assert_close(received.cpu(), expected_exchange, rtol=0, atol=0)
                # Rank zero sends no rows, then receives no rows in the reverse
                # exchange. Both sides must retain valid native buffer pointers.
                sends = [int(rank != 0)] * target
                receives = [int(peer != 0) for peer in range(target)]
                source = torch.full((sum(sends), 2), rank, device="npu", dtype=torch.float32)
                received = torch.empty((sum(receives), 2), device="npu", dtype=torch.float32)
                group.all_to_all(received, source, receives, sends)
                expected_empty = torch.arange(1, target, dtype=torch.float32)[:, None].expand(-1, 2)
                torch.testing.assert_close(received.cpu(), expected_empty, rtol=0, atol=0)
                returned = torch.empty_like(source)
                group.all_to_all(returned, received, sends, receives)
                torch.testing.assert_close(returned, source, rtol=0, atol=0)
                assert [p.data_ptr() for p in layer.parameters()] == parameter_addresses
            comms.barrier()
            observations.append(
                {"active": target, "released_bytes": released, "communication_delta_bytes": comms.last_memory_delta}
            )
            current, active = after, target
    finally:
        if memory.sleeping:
            memory.map()
            memory.finish()
        graph.reset()
        comms.close()
        dist.destroy_process_group()
    return {"rank": rank, "captures": 1, "transitions": observations}


@pytest.mark.skipif(not hasattr(torch, "npu") or torch.npu.device_count() < 4, reason="Requires four NPUs")
def test_nonreplica_expert_migration_and_graph_recovery():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_exercise, args=(port,), nprocs=4, join=True)
