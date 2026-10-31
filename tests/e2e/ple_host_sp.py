# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --nproc-per-node=4 to verify PLE TP/SP routing."""

import os

import torch
import torch.distributed as dist

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.deepseek_v41.engram.npu import HostUvaBuffer
from vllm_ascend.ops.triton.qwen4_exp.ple_lookup import lookup


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    world = dist.get_world_size()
    rows = 64
    table = HostUvaBuffer((rows, 160), torch.bfloat16, f"npu:{rank}")
    values = torch.arange(rows * world * 160).reshape(rows * world, 160).remainder(127).bfloat16()
    table.tensor.copy_(values[rank * rows : (rank + 1) * rows])
    try:
        for shift in [0, 31, 75]:
            host_ids = (torch.arange(32 * 16).reshape(32, 16) + shift) % (rows * world)
            ids = host_ids.npu()
            partial = torch.empty((32, 16, 160), device=ids.device, dtype=torch.bfloat16)
            lookup(table.ptrs, ids, partial, rank * rows, (rank + 1) * rows, 1 << 22)
            full = partial.clone()
            dist.all_reduce(full)
            torch.testing.assert_close(full.cpu(), values[host_ids], rtol=0, atol=0)
            shard = torch.empty((32 // world, 16, 160), device=ids.device, dtype=torch.bfloat16)
            dist.reduce_scatter_tensor(shard, partial)
            expected = values[host_ids].chunk(world)[rank]
            torch.testing.assert_close(shard.cpu(), expected, rtol=0, atol=0)
        dist.barrier()
        print(f"rank={rank} PLE all-reduce and SP reduce-scatter PASS", flush=True)
    finally:
        table.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
