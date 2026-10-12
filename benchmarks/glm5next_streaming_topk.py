# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare dense and streaming GLM KPool selection in fixed-shape NPU graphs.

Run: python benchmarks/glm5next_streaming_topk.py --tokens 1 60
Compilation, capture, host allocation and correctness checks are outside timing.
"""

import argparse
import json
import statistics
from functools import partial

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.glm5_next_lightning_indexer import glm5_next_lightning_indexer_triton


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


def assert_selection(actual, expected, query, cache, weights, visible):
    actual, expected = actual.cpu()[:, 0], expected.cpu()[:, 0]
    torch.testing.assert_close(actual[:, 2048:], expected[:, 2048:], rtol=0, atol=0)
    qbar = (query.float() * weights.float().unsqueeze(-1)).sum(1).cpu()
    keys = cache.cpu().reshape(-1, 128)
    score_sets = []
    for result in (actual, expected):
        pools = result[:, :2048].reshape(-1, 512, 4)
        assert ((pools[:, :, 0] >= 0) & (pools[:, :, 0] < visible * 4)).all()
        torch.testing.assert_close(pools, pools[:, :, :1] + torch.arange(4, dtype=pools.dtype), rtol=0, atol=0)
        ids = (pools[:, :, 0] // 4).long()
        assert all(torch.unique(row).numel() == 512 for row in ids)
        scores = (keys[ids].float() * qbar[:, None, :]).sum(-1)
        score_sets.append(scores.sort(dim=1).values)
    # Boundary ties may select different IDs with identical scores.
    torch.testing.assert_close(*score_sets, rtol=1e-5, atol=1e-4)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 60])
    parser.add_argument("--capacities", type=int, nargs="+", default=[33728, 262208])
    parser.add_argument("--visible-pools", type=int, nargs="+", default=[1024, 8192, 262144])
    options = parser.parse_args()
    if min(options.tokens) <= 0 or min(options.capacities) <= 2048 or min(options.visible_pools) < 512:
        parser.error("tokens must be positive, capacities > 2048, and visible-pools >= 512")
    torch.npu.set_device(0)
    torch.manual_seed(41)
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
                    glm5_next_lightning_indexer_triton,
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
                dense = partial(fn, use_streaming_topk=False)
                streaming = partial(fn, use_streaming_topk=True)
                assert_selection(streaming(), dense(), query, cache, weights, visible)
                dense_us, streaming_us = graph_latency_us(dense), graph_latency_us(streaming)
                print(
                    json.dumps(
                        dict(
                            tokens=tokens,
                            capacity_pools=capacity,
                            visible_pools=visible,
                            dense_us=dense_us,
                            streaming_us=streaming_us,
                            speedup=dense_us / streaming_us,
                            dense_score_bytes=tokens * capacity * 4,
                            streaming_score_and_id_bytes=tokens * 2048 * 8,
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
