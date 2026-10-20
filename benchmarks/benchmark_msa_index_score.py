# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure MSA IndexScore decode through the registered framework operator.

Run this script with each installed build and compare the JSON records. Timings
exclude input allocation, compilation and graph capture. Numerical regression
coverage lives in tests/e2e/pull_request/one_card/test_msa_index_score.py.
"""

import argparse
import json
import math
import statistics

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401

BLOCK_SIZE = 128
NUM_HEADS = 8
HEAD_DIM = 128


def graph_latency_us(invoke, calls: int, replays: int, samples: int) -> list[float]:
    for _ in range(3):
        invoke()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    outputs = []
    with torch.npu.graph(graph):
        for _ in range(calls):
            outputs.append(invoke())
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / (calls * replays))
    # Keep every captured output allocation alive throughout measurement.
    del outputs
    return timings


def benchmark(args, batch: int, max_kv_len: int, distribution: str) -> dict:
    torch.manual_seed(2026)
    dtype = getattr(torch, args.dtype)
    num_pages = math.ceil(max_kv_len / BLOCK_SIZE)
    table_width = max(args.table_width, num_pages)
    query = torch.randn((batch, NUM_HEADS, HEAD_DIM), dtype=dtype, device="npu")
    key = torch.randn((batch * num_pages, BLOCK_SIZE, 1, HEAD_DIM), dtype=dtype, device="npu")
    block_table = torch.zeros((batch, table_width), dtype=torch.int32, device="npu")
    # Give requests disjoint physical pages, as in a serving KV cache.
    block_table[:, :num_pages] = torch.arange(batch * num_pages, dtype=torch.int32, device="npu").view(batch, num_pages)
    if distribution == "uniform":
        lengths = [max_kv_len] * batch
    else:
        lengths = [max(1, max_kv_len * (i + 1) // batch) for i in range(batch)]
    k_lens = torch.tensor(lengths, dtype=torch.int32, device="npu")
    q_lens = torch.arange(batch + 1, dtype=torch.int32, device="npu")
    start_loc = torch.tensor([(length - 1) // BLOCK_SIZE for length in lengths], dtype=torch.int32, device="npu")
    atten_mask = torch.zeros((2048, 2048), dtype=torch.int8, device="npu")

    def invoke():
        return torch.ops._C_ascend.npu_msa_index_score(
            query,
            key,
            block_table,
            start_loc,
            atten_mask=atten_mask,
            actual_seq_qlen=q_lens,
            actual_seq_klen=k_lens,
            layout_key="BBND",
            sparse_mode=3,
            init_blocks=0,
            local_blocks=0,
        )

    timings = graph_latency_us(invoke, args.calls, args.replays, args.samples)
    return {
        "device": torch.npu.get_device_name(0),
        "dtype": args.dtype,
        "batch": batch,
        "max_kv_len": max_kv_len,
        "table_width": table_width,
        "distribution": distribution,
        "calls_per_graph": args.calls,
        "replays_per_sample": args.replays,
        "median_us": statistics.median(timings),
        "samples_us": timings,
    }


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=positive_int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--kv-len", type=positive_int, nargs="+", default=[4096, 65536, 131072])
    parser.add_argument("--distribution", choices=["uniform", "ramp"], nargs="+", default=["uniform", "ramp"])
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--table-width", type=positive_int, default=1, help="minimum block-table width")
    parser.add_argument("--calls", type=positive_int, default=10)
    parser.add_argument("--replays", type=positive_int, default=20)
    parser.add_argument("--samples", type=positive_int, default=5)
    args = parser.parse_args()
    torch.npu.set_device(0)
    for batch in args.batch:
        for kv_len in args.kv_len:
            for distribution in args.distribution:
                print(json.dumps(benchmark(args, batch, kv_len, distribution)), flush=True)


if __name__ == "__main__":
    main()
