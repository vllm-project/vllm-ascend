# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Controlled legacy-normalization/fused-ordering microbenchmarks on one NPU.

Run from an installed checkout. Compilation and correctness are outside timing.
This does not launch a model or claim end-to-end performance improvement.
"""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import time
from pathlib import Path


def measure_pair(functions, torch, warmup, iterations, samples, mode="eager"):
    results = {name: {"device_us": [], "host_us": []} for name in functions}
    for function in functions.values():
        for _ in range(warmup):
            function()
    torch.npu.synchronize()
    # Keep graph outputs alive for the entire timing interval. Allocations and
    # compilation are excluded, just as they are in a warmed serving graph.
    graphs, graph_outputs = {}, {}
    if mode == "graph":
        for name, function in functions.items():
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                graph_outputs[name] = function()
            graphs[name] = graph
        functions = {name: graph.replay for name, graph in graphs.items()}
        for function in functions.values():
            for _ in range(warmup):
                function()
        torch.npu.synchronize()
        for actual, expected in zip(graph_outputs["fused"], graph_outputs["baseline"]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    names = list(functions)
    for sample in range(samples):
        for name in names[:: 1 if sample % 2 == 0 else -1]:
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            torch.npu.synchronize()
            host_start = time.perf_counter()
            start.record()
            for _ in range(iterations):
                functions[name]()
            end.record()
            end.synchronize()
            results[name]["host_us"].append((time.perf_counter() - host_start) * 1e6 / iterations)
            results[name]["device_us"].append(start.elapsed_time(end) * 1000 / iterations)
    for values in results.values():
        for key in ["device_us", "host_us"]:
            ordered = sorted(values[key])
            values[key + "_median"] = statistics.median(ordered)
            values[key + "_p90"] = ordered[int(0.9 * (len(ordered) - 1))]
    results["device_speedup"] = results["baseline"]["device_us_median"] / results["fused"]["device_us_median"]
    return results


def build_postprocess_pair(kind, raw, context, query_len, logical_blocks, torch):
    from vllm.triton_utils import triton

    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.models.minimax_m3.ops import msa_m3_triton_a5 as a5
    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _sort_valid_gbsa_prefix
    from vllm_ascend.ops.triton.msa import msa_m3_index_postprocess as cann
    from vllm_ascend.ops.triton.msa.constants import FP32_INDEX_SENTINEL

    heads, tokens, topk = raw.shape
    indices = torch.empty(raw.shape, dtype=torch.int32, device=raw.device)
    counts = torch.empty(raw.shape[:2], dtype=torch.int32, device=raw.device)

    def legacy():
        if kind == "cann":
            cann._ascendc_decode_topk_postprocess_kernel[(tokens, heads)](
                raw,
                indices,
                counts,
                context,
                *raw.stride(),
                *indices.stride(),
                *counts.stride(),
                context.stride(0),
                logical_blocks,
                query_len,
                TOPK=topk,
                BLOCK_SIZE=128,
                TILE=triton.next_power_of_2(topk),
                SORT_INDICES=False,
                SENTINEL=float(FP32_INDEX_SENTINEL),
            )
        else:
            a5._index_topk_postprocess_kernel[(tokens, heads)](
                raw,
                indices,
                counts,
                context,
                128,
                topk,
                query_len,
                *raw.stride(),
                *indices.stride(),
                *counts.stride(),
                BLOCK_SIZE_T=triton.next_power_of_2(topk),
            )
        return _sort_valid_gbsa_prefix(indices, counts), counts

    def fused():
        if kind == "a5":
            a5._decode_index_postprocess_sorted(raw, context, indices, counts, query_len)
        elif topk == 16:
            cann._ascendc_decode_topk_sorted_batched_kernel[(triton.cdiv(tokens, 4), heads)](
                raw,
                indices,
                counts,
                context,
                tokens,
                logical_blocks,
                query_len,
                TOPK=topk,
                BLOCK_SIZE=128,
                ROWS=4,
                TILE=32,
                SENTINEL=float(FP32_INDEX_SENTINEL),
            )
        else:
            cann._ascendc_decode_topk_postprocess_kernel[(tokens, heads)](
                raw,
                indices,
                counts,
                context,
                *raw.stride(),
                *indices.stride(),
                *counts.stride(),
                context.stride(0),
                logical_blocks,
                query_len,
                TOPK=topk,
                BLOCK_SIZE=128,
                TILE=max(32, triton.next_power_of_2(topk)),
                SORT_INDICES=True,
                SENTINEL=float(FP32_INDEX_SENTINEL),
            )
        return indices, counts

    return {"baseline": legacy, "fused": fused}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--paths", nargs="+", choices=["cann", "a5"], default=["cann"])
    parser.add_argument("--mode", choices=["eager", "graph"], default="eager")
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.samples) < 1:
        parser.error("warmup, iterations and samples must be positive")
    # --help remains usable on a workstation without the NPU runtime.
    import torch
    import torch_npu  # noqa: F401

    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _sort_valid_gbsa_prefix
    from vllm_ascend.ops.triton.msa.msa_m3_index_postprocess import index_decode_topk_fused

    torch.npu.set_device(args.device)
    generator = torch.Generator().manual_seed(18426)
    cases = [(1, 1, 1, 16, 512), (1, 5, 8, 16, 1024), (1, 11, 4, 16, 512), (4, 2, 9, 3, 33), (1, 2, 16, 32, 2055)]
    rows = []
    for heads, requests, query_len, topk, width in cases:
        tokens = requests * query_len
        raw_cpu = torch.stack(
            [torch.randperm(width, generator=generator)[:topk] for _ in range(heads * tokens)]
        ).reshape(heads, tokens, topk)
        expected = raw_cpu.sort(-1).values.to(torch.int32)
        expected_counts = torch.full((heads, tokens), topk, dtype=torch.int32)
        raw = raw_cpu.to(args.device)
        context = torch.full((requests,), (width - 1) * 128, dtype=torch.int32, device=args.device)
        for kind in args.paths:
            lengths = context if kind == "cann" else context + query_len
            functions = build_postprocess_pair(kind, raw, lengths, query_len, width, torch)
            for function in functions.values():
                actual, counts = function()
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)
            measured = measure_pair(functions, torch, args.warmup, args.iterations, args.samples, args.mode)
            rows.append(
                dict(
                    scope=kind + "_postprocess",
                    shape=list(raw.shape),
                    logical_blocks=width,
                    requests=requests,
                    query_len=query_len,
                    exact_integer_match=True,
                    **measured,
                )
            )

        # Complete shared CANN producer: identical score values and forced windows.
        # Score preparation is idempotent; it is an explicit reusable workspace.
        score_cpu = torch.rand((heads, tokens, width), generator=generator)
        scores = {name: score_cpu.to(args.device) for name in ["baseline", "fused"]}

        def baseline_producer(score=scores["baseline"], lengths=context, q=query_len, k=topk, w=width):
            ids, counts = index_decode_topk_fused(score, lengths, q, k, w, 1, 2)
            return _sort_valid_gbsa_prefix(ids, counts), counts

        def fused_producer(score=scores["fused"], lengths=context, q=query_len, k=topk, w=width):
            return index_decode_topk_fused(score, lengths, q, k, w, 1, 2, sort_indices=True)

        baseline_ids, baseline_counts = baseline_producer()
        actual, counts = fused_producer()
        torch.testing.assert_close(actual, baseline_ids, rtol=0, atol=0)
        torch.testing.assert_close(counts, baseline_counts, rtol=0, atol=0)
        measured = measure_pair(
            {"baseline": baseline_producer, "fused": fused_producer},
            torch,
            args.warmup,
            args.iterations,
            args.samples,
            args.mode,
        )
        rows.append(
            dict(
                scope="cann_complete_topk",
                shape=list(scores["fused"].shape),
                requests=requests,
                query_len=query_len,
                topk=topk,
                exact_integer_match=True,
                **measured,
            )
        )
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "vllm_ascend" / name
        for name in [
            "ops/triton/msa/constants.py",
            "ops/triton/msa/msa_m3_index_postprocess.py",
            "models/minimax_m3/ops/msa_m3_triton_a5.py",
        ]
    ]
    packages = {}
    for name in ["torch-npu", "triton-ascend", "triton"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None  # Source archives retain the per-file hashes below.
    report = dict(
        device=torch.npu.get_device_name(),
        torch=torch.__version__,
        packages=packages,
        git_head=git_head,
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        warmup=args.warmup,
        iterations=args.iterations,
        samples=args.samples,
        cases=rows,
        timing_scope=args.mode + " component only; no model or end-to-end speedup claim",
    )
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
