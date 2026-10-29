# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Four-NPU retained HCCL lifecycle, excluded ranks, and peer refill values."""

import socket
import traceback
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_ascend.distributed.fasteep.hccl import RetainedCommunicators


def _worker(rank, port):
    try:
        _exercise(rank, port)
    except BaseException:
        traceback.print_exc()
        raise


def _exercise(rank, port):
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
    try:
        baseline = torch.full((16,), rank + 1, device=f"npu:{rank}", dtype=torch.float32)
        torch.testing.assert_close(owner.root.all_reduce(baseline), torch.full_like(baseline, 10))
        handles = {}
        for size in (2, 3, 2, 3):
            group = owner.child(size)
            owner.barrier()
            if rank < size:
                previous = handles.setdefault(size, group.comm.value)
                assert group.comm.value == previous
                expected = size * (size + 1) // 2
                torch.testing.assert_close(group.all_reduce(baseline), torch.full_like(baseline, expected))
                source = torch.full((2, 4), rank, device=f"npu:{rank}", dtype=torch.bfloat16).T
                gathered = group.all_gather(source, dim=1)
                expected_gather = torch.cat([torch.full_like(source, peer) for peer in range(size)], dim=1)
                torch.testing.assert_close(gathered, expected_gather)
                source = torch.full((2, size * 4), rank + 1, device=f"npu:{rank}", dtype=torch.float32)
                torch.testing.assert_close(
                    group.reduce_scatter(source, dim=1), torch.full_like(source[:, :4], expected)
                )
            else:
                assert group is None
            owner.barrier()
            # Returning to the parent must work with children still resident.
            torch.testing.assert_close(owner.root.all_reduce(baseline), torch.full_like(baseline, 10))
        payload = (
            torch.arange(1024, device=f"npu:{rank}", dtype=torch.float32)
            if rank == 0
            else torch.zeros(1024, device=f"npu:{rank}")
        )
        if rank in (0, 3):
            owner.transfer(
                payload.data_ptr(), payload.numel() * payload.element_size(), 3 if rank == 0 else 0, rank == 0
            )
        torch.npu.synchronize()
        owner.barrier()
        if rank == 3:
            torch.testing.assert_close(payload, torch.arange(1024, device=f"npu:{rank}", dtype=torch.float32))
    finally:
        owner.close()
        dist.destroy_process_group()


@pytest.mark.skipif(not hasattr(torch, "npu") or torch.npu.device_count() < 4, reason="Requires four NPUs")
def test_retained_hccl_lifecycle():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=4, join=True)
