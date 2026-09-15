# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare fixed-shape dense attention costs on Ascend (not model throughput).

Example: python benchmarks/benchmark_fa3.py --num-layers 4 --output fa3.json
FA3 graph timing includes refreshing device-side tiling before each replay.
FIA is measured in eager mode; this is not a comparison to FIA full graphs.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata

BLOCK_SIZE = 128
HEAD_SIZE = 128
NUM_HEADS = 16
NUM_KV_HEADS = 1
SCALE = HEAD_SIZE**-0.5


def measure(operation, iterations):
    for _ in range(3):
        operation()
    torch.npu.synchronize()
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start = time.perf_counter()
    start_event.record()
    for _ in range(iterations):
        operation()
    end_event.record()
    torch.npu.synchronize()
    return {
        "wall_ms": (time.perf_counter() - start) * 1000 / iterations,
        "device_ms": start_event.elapsed_time(end_event) / iterations,
    }


@torch.inference_mode()
def benchmark(batch_size, query_len, kv_len, layers, iterations):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    query = torch.randn(batch_size * query_len, NUM_HEADS, HEAD_SIZE, device="npu", dtype=dtype)
    blocks_per_req = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    key = torch.randn(batch_size * blocks_per_req, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, device="npu", dtype=dtype)
    value = torch.randn_like(key)
    pages = torch.arange(batch_size * blocks_per_req, device="npu", dtype=torch.int32).view(batch_size, -1)
    offsets = torch.arange(batch_size + 1, device="npu", dtype=torch.int32) * query_len
    lengths = torch.full((batch_size,), kv_len, device="npu", dtype=torch.int32)
    tiling_args = dict(
        batch_size=batch_size,
        max_seqlen_q=query_len,
        max_seqlen_k=blocks_per_req * BLOCK_SIZE,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=lengths,
        qkv_dtype=dtype,
        cu_seqlens_q=offsets,
        page_size=BLOCK_SIZE,
        causal=True,
        softmax_scale=SCALE,
        num_splits=1,
    )
    tiling = get_scheduler_metadata(**tiling_args)
    attention_args = dict(
        cache_seqlens=lengths,
        page_table=pages,
        cu_seqlens_q=offsets,
        max_seqlen_q=query_len,
        softmax_scale=SCALE,
        causal=True,
        num_splits=1,
    )
    # Match the compressed causal mask used by the standard Ascend backend.
    mask_size = 2048
    mask = torch.triu(torch.ones(mask_size, mask_size, device="npu", dtype=torch.bool), diagonal=1)
    fia_args = dict(
        query=query,
        key=key.flatten(2),
        value=value.flatten(2),
        atten_mask=mask,
        block_table=pages,
        block_size=BLOCK_SIZE,
        input_layout="TND",
        actual_seq_lengths=[query_len * (i + 1) for i in range(batch_size)],
        actual_seq_lengths_kv=[kv_len] * batch_size,
        num_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        scale=SCALE,
        sparse_mode=3,
    )
    expected = torch_npu.npu_fused_infer_attention_score(**fia_args)[0]
    actual = flash_attn_with_kvcache(query, key, value, scheduler_metadata=tiling, **attention_args)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    def fia_eager():
        for _ in range(layers):
            torch_npu.npu_fused_infer_attention_score(**fia_args)

    def fa3_eager():
        metadata = get_scheduler_metadata(**tiling_args)
        for _ in range(layers):
            flash_attn_with_kvcache(query, key, value, scheduler_metadata=metadata, **attention_args)

    fa3_eager()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(layers):
            flash_attn_with_kvcache(query, key, value, scheduler_metadata=tiling, **attention_args)

    def fa3_graph():
        tiling.copy_(get_scheduler_metadata(**tiling_args))
        graph.replay()

    return {
        "batch_size": batch_size,
        "query_len": query_len,
        "kv_len": kv_len,
        "layers": layers,
        "fia_eager": measure(fia_eager, iterations),
        "fa3_eager": measure(fa3_eager, iterations),
        "fa3_graph": measure(fa3_graph, iterations),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_layers < 1 or args.iterations < 1:
        parser.error("num-layers and iterations must be positive")
    results = [
        benchmark(*case, args.num_layers, args.iterations)
        for case in [(1, 1, 1024), (8, 1, 8192), (8, 4, 8192), (2, 128, 2048)]
    ]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
