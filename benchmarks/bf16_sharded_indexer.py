# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time one rank's BF16 indexer scorer on an Ascend 950 with LightningIndexerV2.

    python benchmarks/bf16_sharded_indexer.py --requests 16 --local-tokens 16384

Each NPU graph contains one scorer, through sorted local IDs and scores.
DCP communication and shared per-step metadata preparation are outside both
timings. The inputs and the native window plan are prepared before capture.
"""

import argparse
import json
import statistics
from functools import partial

import torch


def scorer_latency_us(scorer, warmup, repeats):
    for _ in range(warmup):
        scorer()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        outputs = scorer()
    for _ in range(warmup):
        graph.replay()
    torch.npu.synchronize()
    samples = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    del outputs
    graph.reset()
    return statistics.median(samples)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--tokens-per-request", type=int, default=4)
    parser.add_argument("--local-tokens", type=int, default=4096, help="maximum visible cache tokens on this rank")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10, help="replays per timing sample")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if min(args.requests, args.tokens_per_request, args.local_tokens, args.warmup, args.repeats) < 1:
        parser.error("requests, token counts, warmup and repeats must be positive")

    # Load the NPU runtime only when executing the benchmark.
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.bf16_sharded_indexer import bf16_paged_indexer_candidates
    from vllm_ascend.ops.lightning_indexer_v2 import (
        build_lightning_indexer_v2_plan,
        has_lightning_indexer_v2,
        lightning_indexer_v2_candidates,
    )

    torch.npu.set_device(args.device)
    if not has_lightning_indexer_v2():
        raise RuntimeError("The installed Ascend extension and CANN runtime must provide LightningIndexerV2.")
    torch.manual_seed(37)
    rows = args.requests * args.tokens_per_request
    head_dim, heads, page_size, topk = 128, 32, 128, 2048
    cols = (args.local_tokens + page_size - 1) // page_size
    options = dict(dtype=torch.int32, device="npu")
    table = torch.arange(1, args.requests * cols + 1, **options).reshape(args.requests, cols)
    row = torch.arange(rows, **options)
    owners = row // args.tokens_per_request
    ends = torch.arange(1, args.requests + 1, **options) * args.tokens_per_request
    lengths = (args.local_tokens - args.tokens_per_request + row % args.tokens_per_request + 1).clamp_min(0)
    keys = torch.randn(args.requests * cols + 1, page_size, 1, head_dim, dtype=torch.bfloat16, device="npu")
    query = torch.randn(rows, heads, head_dim, dtype=torch.bfloat16, device="npu")
    weights = torch.randn(rows, heads, dtype=torch.bfloat16, device="npu")
    plan = build_lightning_indexer_v2_plan(table, owners, lengths, keys.size(0), page_size, 1, heads)
    reference = partial(
        bf16_paged_indexer_candidates,
        query,
        weights,
        keys,
        table,
        ends,
        lengths,
        topk,
        request_rows=owners,
        full_graph=True,
    )
    native = partial(lightning_indexer_v2_candidates, query, weights, keys, plan, topk)
    reference_us = scorer_latency_us(reference, args.warmup, args.repeats)
    native_us = scorer_latency_us(native, args.warmup, args.repeats)
    print(
        json.dumps(
            {
                **vars(args),
                "rows": rows,
                "heads": heads,
                "topk": topk,
                "windows": len(plan.windows),
                "shared_metadata_timed": False,
                "reference_graph_us": reference_us,
                "li_v2_graph_us": native_us,
                "speedup": reference_us / native_us,
            }
        )
    )


if __name__ == "__main__":
    main()
