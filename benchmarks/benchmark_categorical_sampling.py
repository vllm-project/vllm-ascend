# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native categorical sampling with the unchanged Ascend Triton path.

Example: python benchmarks/benchmark_categorical_sampling.py --output sampling.csv
Graph timings amortize launch/event overhead over repeated captured calls. Eager
timings include Python dispatch and allocations. Inputs and compilation are not
timed. Different algorithms need not produce identical random token streams.
"""

import argparse
import csv
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_categorical_sample_op
from vllm_ascend.worker.v2.sample.gumbel import _fallback_gumbel_sample, gumbel_sample

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def capture(fn: Callable[[], torch.Tensor], repeats: int) -> tuple[torch.npu.NPUGraph, list[torch.Tensor]]:
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [fn() for _ in range(repeats)]
    for _ in range(5):
        graph.replay()
    torch.npu.synchronize()
    return graph, outputs


def measure(functions: dict[str, Callable[[], torch.Tensor]], repeats: int, rounds: int) -> dict[str, float]:
    captured = {name: capture(fn, repeats) for name, fn in functions.items()}
    graph_samples: dict[str, list[float]] = {name: [] for name in functions}
    eager_samples: dict[str, list[float]] = {name: [] for name in functions}
    for round_idx in range(rounds):
        names = list(functions)
        if round_idx % 2:
            names.reverse()
        for name in names:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            captured[name][0].replay()
            end.record()
            end.synchronize()
            graph_samples[name].append(start.elapsed_time(end) * 1000 / repeats)
            torch.npu.synchronize()
            begin = time.perf_counter()
            for _ in range(repeats):
                functions[name]()
            torch.npu.synchronize()
            eager_samples[name].append((time.perf_counter() - begin) * 1e6 / repeats)
    result = {}
    for name in functions:
        result[f"{name}_graph_us"] = statistics.median(graph_samples[name])
        quartiles = statistics.quantiles(graph_samples[name], n=4)
        result[f"{name}_graph_q25_us"] = quartiles[0]
        result[f"{name}_graph_q75_us"] = quartiles[2]
        result[f"{name}_eager_us"] = statistics.median(eager_samples[name])
    result["graph_speedup"] = result["triton_graph_us"] / result["categorical_graph_us"]
    result["eager_speedup"] = result["triton_eager_us"] / result["categorical_eager_us"]
    return result


def benchmark_case(batch: int, vocab: int, dtype: torch.dtype, scenario: str, args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    logits = torch.randn(batch, vocab, dtype=dtype, device="npu")
    mapping_dtype = torch.int64 if scenario == "int64_mapping" else torch.int32
    mapping = torch.arange(batch, dtype=mapping_dtype, device="npu")
    temperature = torch.full((batch,), 0.0 if scenario == "greedy" else 0.7, device="npu")
    seeds = torch.arange(batch, dtype=torch.int64, device="npu") + args.seed
    positions = torch.arange(batch, dtype=torch.int64, device="npu")
    cache = None
    columns = None
    if scenario in ("cache", "strided_cache", "draft"):
        cache = torch.empty(batch, 4, vocab + 16, dtype=dtype, device="npu")
        if scenario == "strided_cache":
            cache = cache[:, ::2, :]
        columns = torch.ones(batch, dtype=torch.int32, device="npu")
    kwargs = dict(
        apply_temperature=scenario != "no_temperature",
        is_drafting=scenario == "draft",
        logits_cache=cache,
        logits_cache_col=columns,
    )
    tensors = (logits, mapping, temperature, seeds, positions)
    functions = {
        "triton": lambda: _fallback_gumbel_sample(*tensors, **kwargs),
        "categorical": lambda: gumbel_sample(*tensors, **kwargs),
    }
    if args.include_fp64:
        functions["categorical_fp64"] = lambda: gumbel_sample(*tensors, **kwargs, use_fp64=True)
    for fn in functions.values():
        output = fn()
        assert output.shape == (batch,)
        assert bool(((output >= 0) & (output < vocab)).all())
        if scenario == "greedy":
            torch.testing.assert_close(output.long(), logits.argmax(dim=-1))
        if cache is not None:
            torch.testing.assert_close(cache[:, 1, :vocab], logits)
    return measure(functions, args.repeats, args.rounds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16, 64, 256, 512])
    parser.add_argument("--vocabs", nargs="+", type=int, default=[4096, 32000, 65536, 128256, 151936, 262144])
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["random", "greedy", "cache", "strided_cache", "draft", "int64_mapping", "no_temperature"],
        default=["random"],
    )
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--include-fp64", action="store_true")
    parser.add_argument("--reverse", action="store_true", help="Reverse case order for a second independent run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.rounds < 2 or min(args.batches + args.vocabs) < 1:
        parser.error("positive shapes/repeats and at least two rounds are required")
    if not enable_categorical_sample_op():
        raise RuntimeError("the native categorical operator is unavailable; benchmarking fallback is invalid")
    cases = [(b, v, d, s) for s in args.scenarios for d in args.dtypes for b in args.batches for v in args.vocabs]
    if args.reverse:
        cases.reverse()
    with args.output.open("w", newline="") as output_file:
        writer = None
        for index, (batch, vocab, dtype, scenario) in enumerate(cases, 1):
            row = {"batch": batch, "vocab": vocab, "dtype": dtype, "scenario": scenario, "seed": args.seed}
            row.update(benchmark_case(batch, vocab, DTYPES[dtype], scenario, args))
            if writer is None:
                writer = csv.DictWriter(output_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            output_file.flush()
            print(f"[{index}/{len(cases)}] {row}", flush=True)


if __name__ == "__main__":
    main()
