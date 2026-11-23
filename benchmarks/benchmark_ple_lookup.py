# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare compact PLE host lookup with the same rows expanded to BF16.

Run in a fresh process and TRITON_CACHE_DIR for cold-start compiler counts.
This synthetic single-rank benchmark does not measure model serving.
"""

import argparse
import json
import statistics
from collections import Counter
from importlib.metadata import version

import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401  # Register ops before importing DeviceOperator through Engram.
from vllm_ascend.models.deepseek_v41.engram.npu import CHUNK_ROWS, HostUvaBuffer
from vllm_ascend.models.qwen4_exp.ple_quant import AscendPLEQuantizedEmbeddingMethod
from vllm_ascend.ops.triton.qwen4_exp.ple_lookup import _lookup, lookup
from vllm_ascend.ops.triton.qwen4_exp.ple_lookup_quantized import _lookup_quantized, lookup_quantized


def graph_latency(run, samples, repeats):
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            # Put repetitions inside the graph so Python replay submission
            # gaps are not divided into the reported operator latency.
            for _ in range(repeats):
                run()
        for _ in range(5):
            graph.replay()
        timings = []
        for _ in range(samples):
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            timings.append(begin.elapsed_time(end) * 1000 / repeats)
        return statistics.median(timings), timings
    finally:
        torch.npu.synchronize()
        graph.reset()


def main(args):
    torch.npu.set_device(args.device)
    torch.manual_seed(0)
    device = torch.device("npu", args.device)
    method = AscendPLEQuantizedEmbeddingMethod(mxfp8=args.format == "mxfp8")
    buffers: list[HostUvaBuffer] = []
    owner = torch.nn.Module()
    compile_entries: Counter[str] = Counter()
    original_compilers = [(kernel, kernel._do_compile) for kernel in (_lookup, _lookup_quantized)]

    # Count the actual JIT dispatch entry, including persistent-cache lookups.
    # Only a fresh cache can make this a cold compilation count.
    for kernel, original_compile in original_compilers:

        def counted_compile(*positional, _compile=original_compile, _name=kernel.__name__, **keywords):
            compile_entries[_name] += 1
            return _compile(*positional, **keywords)

        kernel._do_compile = counted_compile
    try:
        for columns, dtype in (
            (args.width, method.weight_dtype),
            (args.width // 32, method.scale_dtype),
            (args.width, torch.bfloat16),
        ):
            buffers.append(HostUvaBuffer((args.rows, columns), dtype, device, f"table_{len(buffers)}", owner))
        codes, scales, bf16 = buffers
        # Production tables contain tens of millions of rows. Bound temporary
        # CPU allocations while keeping all three registered tables resident.
        for begin in range(0, args.rows, 65536):
            end = min(begin + 65536, args.rows)
            count = end - begin
            values = torch.randint(-7, 8, (count, args.width)).to(method.weight_dtype)
            decoded = torch.linspace(0.25, 2, args.width // 32).expand(count, -1)
            if method.mxfp8:
                exponents = torch.arange(args.width // 32).remainder(4).add(125).to(torch.uint8)
                scales.tensor[begin:end].copy_(exponents.expand(count, -1))
                decoded = torch.exp2(exponents.float() - 127).expand(count, -1)
            else:
                scales.tensor[begin:end].copy_(decoded)
            codes.tensor[begin:end].copy_(values)
            bf16.tensor[begin:end].copy_(
                (values.float().unflatten(-1, (-1, 32)) * decoded.unsqueeze(-1)).flatten(-2).bfloat16()
            )

        def runners(ids, compact_output, baseline_output):
            def compact():
                lookup_quantized(
                    codes.ptrs,
                    scales.ptrs,
                    ids,
                    compact_output,
                    args.shard_start,
                    args.shard_start + args.rows,
                    CHUNK_ROWS,
                    is_mxfp8=method.mxfp8,
                )

            def baseline():
                lookup(bf16.ptrs, ids, baseline_output, args.shard_start, args.shard_start + args.rows, CHUNK_ROWS)

            return compact, baseline

        ids = torch.full((1,), args.shard_start, dtype=torch.int64, device=device)
        output = torch.empty((1, args.width), dtype=torch.bfloat16, device=device)
        compact, baseline = runners(ids, output, torch.empty_like(output))
        compact()
        baseline()
        torch.npu.synchronize()
        startup_entries = dict(compile_entries)
        print(
            json.dumps(
                dict(
                    environment={name: version(name) for name in ("torch", "torch-npu", "triton-ascend")},
                    device=torch.npu.get_device_name(args.device),
                    format=args.format,
                    rows=args.rows,
                    width=args.width,
                    id_dtype="int64",
                    start=args.shard_start,
                    end=args.shard_start + args.rows,
                    ownership_shards=args.ownership_shards,
                    chunk=CHUNK_ROWS,
                    block=1 << (args.width - 1).bit_length(),
                    samples=args.samples,
                    repeats=args.repeats,
                    compact_storage_bytes=sum(b.tensor.numel() * b.tensor.element_size() for b in (codes, scales)),
                    bf16_storage_bytes=bf16.tensor.numel() * bf16.tensor.element_size(),
                    startup_jit_compile_entries=startup_entries,
                )
            ),
            flush=True,
        )
        for tokens in args.tokens:
            ids = torch.randint(args.rows * args.ownership_shards, (tokens,), dtype=torch.int64, device=device)
            ids += args.shard_start
            output = torch.empty((tokens, args.width), dtype=torch.bfloat16, device=device)
            baseline_output = torch.empty_like(output)
            compact, baseline = runners(ids, output, baseline_output)
            compact()
            baseline()
            torch.testing.assert_close(output, baseline_output, rtol=0, atol=0)
            owned_rows = int(((ids >= args.shard_start) & (ids < args.shard_start + args.rows)).sum().cpu())
            compact_us, compact_samples = graph_latency(compact, args.samples, args.repeats)
            bf16_us, bf16_samples = graph_latency(baseline, args.samples, args.repeats)
            print(
                json.dumps(
                    dict(
                        tokens=tokens,
                        owned_rows=owned_rows,
                        compact_us=compact_us,
                        bf16_us=bf16_us,
                        compact_over_bf16=compact_us / bf16_us,
                        compact_samples_us=compact_samples,
                        bf16_samples_us=bf16_samples,
                    )
                ),
                flush=True,
            )
        serving_entries = {key: count - startup_entries.get(key, 0) for key, count in compile_entries.items()}
        print(json.dumps(dict(synthetic_runtime_additional_jit_compile_entries=serving_entries)), flush=True)
    finally:
        for kernel, original_compile in original_compilers:
            kernel._do_compile = original_compile
        torch.npu.synchronize()
        for buffer in reversed(buffers):
            buffer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("int8", "mxfp8"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=262144)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--ownership-shards", type=int, default=1, help="Uniform IDs over this many equal shards")
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 256, 4096])
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if (
        args.width <= 0
        or args.width % 32
        or args.shard_start < 0
        or min(args.rows, args.samples, args.repeats, args.ownership_shards, *args.tokens) <= 0
    ):
        parser.error("width must be a positive multiple of 32; sizes, samples and repeats must be positive")
    main(args)
