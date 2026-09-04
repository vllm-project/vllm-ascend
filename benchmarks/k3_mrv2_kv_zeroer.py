# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark the MRV2 Ascend KV block zeroing hot path."""

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.utils import AscendKVBlockZeroer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=24, help="Attention layers; each contributes K and V segments")
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--blocks-per-step", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--padding-elements", type=int, default=64)
    parser.add_argument("--max-p95-us", type=float, default=None, help="Optional regression threshold")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if min(args.layers, args.num_blocks, args.blocks_per_step, args.iterations) <= 0:
        raise ValueError("layers, blocks, and iterations must be positive")
    if args.blocks_per_step > args.num_blocks:
        raise ValueError("blocks-per-step cannot exceed num-blocks")

    device = torch.device("npu")
    init_device_properties_triton()
    block_size = 128
    head_size = 64
    dtype = torch.bfloat16
    payload_elements = block_size * head_size
    stride_elements = payload_elements + args.padding_elements
    layer_names = [f"layer.{idx}" for idx in range(args.layers)]
    raw_buffers: list[torch.Tensor] = []
    static_forward_context = {}

    for layer_name in layer_names:
        components = []
        for _ in range(2):
            raw = torch.empty(
                64 + args.num_blocks * stride_elements,
                dtype=dtype,
                device=device,
            )
            component = torch.as_strided(
                raw,
                (args.num_blocks, block_size, 1, head_size),
                (stride_elements, head_size, head_size, 1),
                storage_offset=64,
            )
            raw_buffers.append(raw)
            components.append(component)
        static_forward_context[layer_name] = SimpleNamespace(kv_cache=tuple(components))

    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
    )
    group = SimpleNamespace(
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        layer_names=layer_names,
    )
    zeroer = AscendKVBlockZeroer(device, pin_memory=True)
    zeroer.init_meta(
        [group],
        [[block_size]],
        "auto",
        set(),
        static_forward_context,
    )

    block_ids = list(range(args.blocks_per_step))
    for _ in range(args.warmup_iterations):
        zeroer.zero_block_ids(block_ids)
    torch.npu.synchronize()

    samples_us = []
    for iteration in range(args.iterations):
        start_block = iteration % (args.num_blocks - args.blocks_per_step + 1)
        block_ids = list(range(start_block, start_block + args.blocks_per_step))
        start = time.perf_counter_ns()
        zeroer.zero_block_ids(block_ids)
        torch.npu.synchronize()
        samples_us.append((time.perf_counter_ns() - start) / 1000)

    sorted_samples = sorted(samples_us)
    p95_index = min(len(sorted_samples) - 1, int(len(sorted_samples) * 0.95))
    result = {
        "layers": args.layers,
        "segments": args.layers * 2,
        "num_blocks": args.num_blocks,
        "blocks_per_step": args.blocks_per_step,
        "iterations": args.iterations,
        "mean_us": statistics.fmean(samples_us),
        "p50_us": statistics.median(samples_us),
        "p95_us": sorted_samples[p95_index],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.max_p95_us is not None and result["p95_us"] > args.max_p95_us:
        raise SystemExit(f"p95 {result['p95_us']:.2f} us exceeds {args.max_p95_us:.2f} us")


if __name__ == "__main__":
    main()
