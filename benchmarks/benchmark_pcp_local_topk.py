# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Compare PCP routing paths without loading a model.

Example: torchrun --standalone --nproc-per-node=8 \
    benchmarks/benchmark_pcp_local_topk.py --dp-size 2 --tokens 2000

Use an otherwise idle, reserved host. Latencies include DP preparation, routing,
hidden-state gathering and packed-route gathering, excluding expert execution.
The reference uses the same checkout's unchanged prepare-then-route path.
"""

import argparse
import json
import os
import statistics
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm_ascend.ops.fused_moe import prepare_finalize
from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter


class Group:
    def __init__(self, process_group, size):
        self.process_group = process_group
        self.size = size

    def all_gather(self, tensor, dim=0):
        assert dim == 0
        output = tensor.new_empty((self.size * tensor.shape[0], *tensor.shape[1:]))
        dist.all_gather_into_tensor(output, tensor.contiguous(), group=self.process_group)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=2000, help="Maximum local tokens before DP preparation")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert args.dp_size > 0 and world % args.dp_size == 0
    pcp_size = world // args.dp_size
    assert pcp_size > 1 and args.tokens > 0
    assert args.repetitions > 0 and args.iterations > 0
    torch.set_num_threads(2)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    # Every rank creates all groups in the same order.
    for d in range(args.dp_size):
        ranks = list(range(d * pcp_size, (d + 1) * pcp_size))
        process_group = dist.new_group(ranks)
        if rank in ranks:
            pcp = Group(process_group, pcp_size)
    for p in range(pcp_size):
        ranks = [d * pcp_size + p for d in range(args.dp_size)]
        process_group = dist.new_group(ranks)
        if rank in ranks:
            dp = Group(process_group, args.dp_size)
    prepare_finalize.get_pcp_group = lambda: pcp
    prepare_finalize._EXTRA_CTX = SimpleNamespace(
        max_tokens_across_dp=args.tokens, max_tokens_across_pcp=args.tokens * args.dp_size
    )
    config = SimpleNamespace(dp_size=args.dp_size, dp_group=dp, pcp_size=pcp_size, is_sequence_parallel=False)
    preparer = object.__new__(prepare_finalize.PrepareAndFinalizeWithAllGather)
    preparer.moe_config = config
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    bias = torch.linspace(-0.1, 0.1, 288, device=device)
    router = AscendFusedTopKRouter(
        top_k=8,
        global_num_experts=288,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
        e_score_correction_bias=bias,
    )

    def run(name, hidden, logits):
        if name == "baseline":
            prepared = preparer.prepare(hidden, logits)
            weights, ids = router._select_experts(prepared.hidden_states, prepared.router_logits)
            return prepared.hidden_states, weights, ids
        hidden, weights, ids = preparer.prepare_with_local_topk(hidden, logits, router.select_experts_locally)
        weights, ids = router.finalize_local_routing(weights, ids)
        return hidden, weights, ids

    # Distinct rank data, ties, uneven lengths and a completely empty rank.
    torch.manual_seed(1024 + rank)
    hidden = torch.randn(args.tokens, 4096, dtype=torch.bfloat16, device=device)
    logits = torch.randn(args.tokens, 288, device=device)
    for kind in ("random", "ties", "uneven"):
        count = args.tokens if kind != "uneven" else (0 if rank == 0 else max(0, args.tokens - rank))
        local_logits = torch.zeros_like(logits[:count]) if kind == "ties" else logits[:count]
        expected = run("baseline", hidden[:count], local_logits)
        actual = run("candidate", hidden[:count], local_logits)
        same = all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(expected, actual))
        passed = torch.tensor(int(same), dtype=torch.int32, device=device)
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        assert bool(passed.cpu()), (rank, kind)

    for _ in range(5):
        run("baseline", hidden, logits)
        run("candidate", hidden, logits)
    pairs = []
    for repeat in range(args.repetitions):
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        pair = {"repeat": repeat, "order": order}
        for name in order:
            dist.barrier()
            torch.npu.synchronize()
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                run(name, hidden, logits)
            end.record()
            torch.npu.synchronize()
            elapsed = torch.tensor(start.elapsed_time(end) / args.iterations, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            pair[name] = float(elapsed.cpu())
        pairs.append(pair)
    if rank == 0:
        print(
            json.dumps(
                {
                    "dp_size": args.dp_size,
                    "pcp_size": pcp_size,
                    "local_tokens": args.tokens,
                    "experts": 288,
                    "top_k": 8,
                    "hidden_size": 4096,
                    "repetitions": args.repetitions,
                    "iterations": args.iterations,
                    "correctness": "bitwise_equal_random_ties_uneven_empty",
                    "metric": "max_rank_device_ms",
                    "pairs": pairs,
                    "median_ms": {
                        name: statistics.median(pair[name] for pair in pairs) for name in ("baseline", "candidate")
                    },
                },
                indent=2,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
