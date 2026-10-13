# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure dense GLM KPool selection in fixed-shape NPU graphs.

Run: python benchmarks/glm5next_dense_topk.py --tokens 1 60
Compilation, capture, host allocation are outside timing.
"""

import argparse
import json
import statistics
from functools import partial
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton import glm5_next_lightning_indexer as indexer


def graph_latency_us(fn):
    fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fn()
    graph.replay()
    torch.npu.synchronize()

    def sample(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / repeats

    estimate = max(sample(1), 1)
    repeats = min(100, max(1, int(100_000 / estimate)))
    latency = statistics.median(sample(repeats) for _ in range(5))
    del output
    return latency


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 60])
    parser.add_argument("--capacities", type=int, nargs="+", default=[33728, 262208])
    parser.add_argument("--visible-pools", type=int, nargs="+", default=[1024, 32768, 262144])
    parser.add_argument("--json-output", type=Path)
    options = parser.parse_args()
    if min(options.tokens) <= 0 or min(options.capacities) <= 2048 or min(options.visible_pools) < 512:
        parser.error("tokens must be positive, capacities > 2048, and visible-pools >= 512")
    torch.npu.set_device(0)
    torch.manual_seed(41)
    results = []
    for capacity in options.capacities:
        pages = (capacity + 31) // 32
        cache = torch.randn(pages, 32, 1, 128, dtype=torch.bfloat16, device="npu")
        for tokens in options.tokens:
            query = torch.randn(tokens, 32, 128, dtype=torch.bfloat16, device="npu")
            weights = torch.randn(tokens, 32, dtype=torch.bfloat16, device="npu")
            ends = torch.arange(1, tokens + 1, dtype=torch.int32, device="npu")
            table = torch.arange(pages, dtype=torch.int32, device="npu").repeat(tokens, 1)
            for visible in options.visible_pools:
                if visible > capacity:
                    continue
                lengths = torch.full((tokens,), visible, dtype=torch.int32, device="npu")
                positions = torch.full((tokens,), visible * 4 - 1, dtype=torch.int64, device="npu")
                fn = partial(
                    indexer.glm5_next_lightning_indexer_triton,
                    query,
                    cache,
                    weights,
                    ends,
                    lengths,
                    table,
                    positions,
                    index_topk=2048,
                    index_kpool=4,
                    max_pool_seq_len=capacity,
                    allow_cache_packing=False,
                )
                dense_us = graph_latency_us(fn)
                record = dict(
                    tokens=tokens,
                    capacity_pools=capacity,
                    visible_pools=visible,
                    actual_context_tokens=visible * 4,
                    dense_us=dense_us,
                    dense_score_bytes=tokens * capacity * 4,
                )
                results.append(record)
                print(json.dumps(record), flush=True)
                if options.json_output:
                    options.json_output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
