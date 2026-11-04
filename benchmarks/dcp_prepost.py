# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check and time Kimi DCP preprocessing and postprocessing over real HCCL.

Run the same script against the baseline and candidate source environments:
    torchrun --standalone --nproc-per-node=8 benchmarks/dcp_prepost.py \
        --variant candidate --output result.json
Use AB/BA order on idle devices. Times exclude compilation, capture and warmup;
all samples contain the maximum device-event duration across ranks. Query
lengths 1/2/4/8 cover target verification shapes, not speculative acceptance.
Native LI/SFA and model projections are excluded from this microbenchmark.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.attention.context_parallel.sharded_indexer import (
    distributed_topk,
    local_to_global,
    local_visible_lengths,
    merge_topk_candidates,
    query_visible_lengths,
)
from vllm_ascend.ops.triton.dcp.dcp_a2a import dcp_a2a_fused_combine
from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton


class Group:
    def __init__(self, rank, world):
        self.rank_in_group, self.world_size = rank, world

    def all_gather(self, x, dim=0):
        y = torch.empty((self.world_size * x.shape[0], *x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(y, x)
        if dim == 0:
            return y
        return torch.cat(y.chunk(self.world_size, 0), dim=dim)


def measure(fn, graph_mode, iterations=20):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    graph = None
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(iterations):
                fn()
        execute = graph.replay
        for _ in range(3):
            execute()
        torch.npu.synchronize()
    else:

        def execute():
            for _ in range(iterations):
                fn()

    samples = []
    for _ in range(7):
        dist.barrier()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        execute()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    values = torch.tensor(samples, device="npu")
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    if graph is not None:
        graph.reset()
    return values.cpu().tolist()


def run_case(args, group, batch, qlen, results):
    rank, world = group.rank_in_group, group.world_size
    rows, topk, interleave = batch * qlen, 2048, 768
    torch.manual_seed(2026 + rank)
    ids_cpu = torch.randint(0, 131072 // world, (rows, 1, topk), dtype=torch.int32)
    scores_cpu = torch.randn(rows, 1, topk).bfloat16().float()
    ids_cpu[..., -17:] = -1
    scores_cpu[..., :9] = 1.0
    ids, scores = ids_cpu.npu(), scores_cpu.npu()
    globals_cpu = group.all_gather(local_to_global(ids, world, rank, interleave), dim=-1).cpu()
    all_scores_cpu = group.all_gather(scores, dim=-1).cpu()
    expected = merge_topk_candidates(globals_cpu, all_scores_cpu, topk)

    def select():
        return distributed_topk(ids, scores, group, topk, interleave)

    selected = select()
    assert torch.equal(selected.cpu(), expected), (batch, qlen, "topk")

    def remap():
        return remap_sparse_indices_triton(selected, world, rank, interleave)

    expected_remap = torch.full_like(expected, -1)
    for row in range(rows):
        x = expected[row, 0]
        own = (x >= 0) & ((x // interleave) % world == rank)
        local = x[own] // (world * interleave) * interleave + x[own] % interleave
        expected_remap[row, 0, : local.numel()] = local
    assert torch.equal(remap().cpu(), expected_remap)
    # DCP gathers all 96 Kimi attention heads before the sparse kernel.
    partial = torch.randn(rows, 96, 512, device="npu", dtype=torch.bfloat16)
    maxima = torch.randn(96, rows, 1, device="npu").transpose(0, 1)
    sums = torch.rand(96, rows, 1, device="npu").transpose(0, 1) + 0.25

    def merge():
        if args.variant == "candidate":
            return dcp_a2a_fused_combine(partial, maxima, world, 1, dist.group.WORLD, softmax_sum=sums)
        return dcp_a2a_fused_combine(partial, maxima + sums.log(), world, 1, dist.group.WORLD)

    all_outputs = group.all_gather(partial).reshape(world, rows, 96, 512).float()
    all_lse = group.all_gather(maxima + sums.log()).reshape(world, rows, 96, 1)
    expected_out = (all_outputs * all_lse.softmax(0)).sum(0)[:, rank * (96 // world) : (rank + 1) * (96 // world)]
    torch.testing.assert_close(merge().float(), expected_out, rtol=0.008, atol=0.008)
    seq = torch.full((batch,), 131072, dtype=torch.int32, device="npu")
    ends = torch.arange(1, batch + 1, dtype=torch.int32, device="npu") * qlen
    keys = torch.empty_like(seq)
    visible = torch.empty((rows, 1), dtype=torch.int32, device="npu")
    row_ids = torch.arange(rows, dtype=torch.int32, device="npu")
    requests = torch.arange(batch, dtype=torch.long, device="npu").repeat_interleave(qlen)

    def visibility():
        if args.variant == "candidate":
            from vllm_ascend.ops.triton.sharded_indexer import build_local_lengths

            build_local_lengths(seq, ends, keys, visible, world, rank, interleave)
        else:
            keys.copy_(local_visible_lengths(seq, world, rank, interleave))
            vis = query_visible_lengths(seq, ends, row_ids, requests)
            visible.copy_(local_visible_lengths(vis, world, rank, interleave).view(-1, 1))
        return visible

    visibility()
    assert torch.equal(
        visible.cpu().flatten(),
        local_visible_lengths(
            query_visible_lengths(seq.cpu(), ends.cpu(), row_ids.cpu(), requests.cpu()), world, rank, interleave
        ),
    )

    def chain():
        visibility()
        selected = select()
        remap_sparse_indices_triton(selected, world, rank, interleave)
        return merge()

    for mode in ["eager", "graph"]:
        # Changed graph inputs test communication layout, not just static replay.
        if mode == "graph":
            g = torch.npu.NPUGraph()
            with torch.npu.graph(g):
                graph_selected = select()
            scores.fill_(1)
            g.replay()
            same_scores = torch.ones_like(all_scores_cpu)
            expect_ties = merge_topk_candidates(globals_cpu, same_scores, topk)
            assert torch.equal(graph_selected.cpu(), expect_ties)
            scores.copy_(scores_cpu.npu())
            g.replay()
            assert torch.equal(graph_selected.cpu(), expected)
            g.reset()
        for name, fn in [
            ("visible", visibility),
            ("candidate_exchange_topk", select),
            ("remap", remap),
            ("output_merge", merge),
            ("prepost_chain", chain),
        ]:
            samples = measure(fn, mode == "graph")
            result = dict(
                batch=batch,
                qlen=qlen,
                rows=rows,
                topk=topk,
                interleave=interleave,
                world=world,
                mode=mode,
                stage=name,
                max_rank_us=samples,
                variant=args.variant,
            )
            results.append(result)
            if rank == 0:
                print(json.dumps(result), flush=True)
                Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    if args.profile:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.profile + f"/rank{rank}"),
            record_shapes=True,
        ) as prof:
            for _ in range(3):
                chain()
                torch.npu.synchronize()
                prof.step()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["base", "candidate"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--profile")
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    group = Group(rank, world)
    cases = [(1, 1), (4, 1), (16, 1), (32, 1), (16, 2), (16, 4), (16, 8), (32, 8)]
    if args.quick:
        cases = [(16, 1)]
    results = []
    for batch, qlen in cases:
        run_case(args, group, batch, qlen, results)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
