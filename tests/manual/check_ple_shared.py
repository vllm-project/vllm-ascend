# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun on four idle NPUs to validate shared PLE tables."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.qwen4_exp.ple_embedding import AscendPLEHostEmbedding


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("gloo")
    obj = object.__new__(AscendPLEHostEmbedding)
    torch.nn.Module.__init__(obj)
    obj._shared_group = SimpleNamespace(cpu_group=dist.group.WORLD, rank_in_group=rank, world_size=4)
    obj._host_table = None
    obj.embedding_dim = 160
    obj.etp_data_parallel_size = obj.tp_size = 1
    obj.reduce_results = True
    obj.shard_indices = SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=32)
    obj.weight = torch.nn.Parameter(obj.allocate_embedding_weight(32, 160, torch.bfloat16), requires_grad=False)
    source = torch.arange(32 * 160).reshape(32, 160).remainder(127).bfloat16()
    obj.weight_loader(obj.weight, source, checkpoint_start=0)
    ids = torch.tensor([[rank, 31, -1]], device="npu")
    for _ in range(3):
        out = obj(ids)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = obj(ids)
    for step in range(3):
        if rank == 0:
            obj.weight.data.copy_(source + step)
        dist.barrier()
        ids.copy_(torch.tensor([[rank + step, 31, -1]]))
        graph.replay()
        expected = torch.stack([source[rank + step] + step, source[31] + step, torch.zeros(160)]).bfloat16()[None]
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    print(f"RANK {rank}: shared loading and changing-weight graph replay passed", flush=True)
    torch.npu.synchronize()
    graph.reset()
    dist.barrier()
    obj.close_host_offload()
    obj.close_host_offload()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
