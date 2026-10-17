# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the original cat-based KPool guard with its fused implementation.

Run in a vllm-ascend environment on one idle NPU. Times include the guard's
allocations in eager mode; graph results amortize replay dispatch over five calls.
No performance threshold is enforced across different devices or CANN versions.
"""

import argparse
import json
import statistics
import time
from functools import partial

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.glm5_next_cann_keypool import prepare_nonempty_keypool_inputs
from vllm_ascend.utils import bootstrap_custom_op_env


def cat_reference(query, weights, ends, lengths, tails, table):
    return (
        torch.cat((query, torch.zeros_like(query[:1]))),
        torch.cat((weights, torch.zeros_like(weights[:1]))),
        torch.cat((ends, ends.new_full((1,), query.shape[0] + 1))),
        torch.cat((lengths.clamp_min(1), lengths.new_ones((1,)))),
        torch.cat((tails, tails.new_zeros((1,)))),
        torch.cat((table, torch.zeros_like(table[:1]))),
    )


def measure(fn, mode, iterations, repeats):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    graph = None
    calls = 1
    if mode == "graph":
        calls = 5
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = [fn() for _ in range(calls)]
        run = graph.replay
    else:
        run = fn
    samples = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            run()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / (iterations * calls))
    result = {"median_us": statistics.median(samples), "samples_us": samples}
    if graph is not None:
        del outputs, graph
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[3, 16, 64, 384, 2048])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    bootstrap_custom_op_env(include_vendor_lib=True)
    import cann_ops_transformer  # noqa: F401

    torch.manual_seed(1024)
    torch.npu.set_device(0)
    device = "npu:0"
    records = []
    for tokens in args.tokens:
        requests = min(tokens, 64)
        query = torch.randn(tokens, 32, 128, dtype=torch.bfloat16, device=device)
        weights = torch.ones(tokens, 32, dtype=torch.bfloat16, device=device)
        ends = torch.arange(1, requests + 1, dtype=torch.int64, device=device) * tokens // requests
        lengths = torch.full((requests,), 700, dtype=torch.int64, device=device)
        tails = torch.zeros(requests, dtype=torch.int64, device=device)
        table = (torch.arange(1025, dtype=torch.int32, device=device) % 32).expand(requests, -1).contiguous()
        cache = torch.randn(32, 32, 1, 128, dtype=torch.bfloat16, device=device)
        inputs = (query, weights, ends, lengths, tails, table)
        for actual, expected in zip(prepare_nonempty_keypool_inputs(*inputs), cat_reference(*inputs)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        def with_indexer(prepare, inputs=inputs, cache=cache, tokens=tokens):
            q, w, e, k, t, b = prepare(*inputs)
            result, _ = torch.ops.cann_ops_transformer.pool_key_indexer(
                q,
                cache,
                w,
                t,
                actual_seq_q=e,
                actual_seq_k=k,
                block_table=b,
                layout_q="TND",
                layout_k="PA_BBND",
                topk=2048,
                pool_size=4,
                mask_mode=3,
            )
            return result[:tokens]

        for scope in ("prepare", "prepare_and_indexer"):
            functions = {
                "cat": partial(cat_reference, *inputs),
                "fused": partial(prepare_nonempty_keypool_inputs, *inputs),
            }
            if scope == "prepare_and_indexer":
                functions = {
                    "cat": partial(with_indexer, cat_reference),
                    "fused": partial(with_indexer, prepare_nonempty_keypool_inputs),
                }
                torch.testing.assert_close(functions["fused"](), functions["cat"](), rtol=0, atol=0)
            for mode in ("eager", "graph"):
                record = {"tokens": tokens, "requests": requests, "scope": scope, "mode": mode}
                # Alternate order across shapes to reduce systematic order bias.
                order = ("fused", "cat") if tokens % 2 else ("cat", "fused")
                for name in order:
                    record[name] = measure(functions[name], mode, args.iterations, args.repeats)
                record["speedup"] = record["cat"]["median_us"] / record["fused"]["median_us"]
                records.append(record)
                print(json.dumps(record), flush=True)
    if args.output:
        with open(args.output, "w") as out:
            json.dump(records, out, indent=2)


if __name__ == "__main__":
    main()
