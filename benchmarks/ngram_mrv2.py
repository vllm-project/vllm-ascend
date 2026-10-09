# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Measure the MRV2 NPU proposer without model loading or JIT compilation time.

Run: python benchmarks/ngram_mrv2.py --batch-sizes 1 32 128
Reports proposal latency, not serving throughput or model accuracy.
"""

import argparse
import json
import time
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from vllm_ascend.worker.v2.spec_decode import init_speculator

PATTERN_LENGTH = 32


@torch.inference_mode()
def benchmark(batch_size, seq_len, capacity, k, iterations, workload):
    device = torch.device("npu")
    row = torch.arange(seq_len, dtype=torch.int32, device=device)
    if workload == "repeat":
        row.remainder_(PATTERN_LENGTH)
    history = torch.zeros((batch_size, capacity), dtype=torch.int32, device=device)
    history[:, :seq_len] = row
    lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    states = SimpleNamespace(all_token_ids=SimpleNamespace(gpu=history), total_len=SimpleNamespace(gpu=lengths))
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="ngram_gpu", prompt_lookup_min=2, prompt_lookup_max=5, num_speculative_tokens=k
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=batch_size),
        model_config=SimpleNamespace(max_model_len=capacity),
    )
    speculator = init_speculator(config, device, states)
    batch = SimpleNamespace(num_reqs=batch_size, idx_mapping=torch.arange(batch_size, device=device))
    empty = torch.empty(0, device=device)
    args = dict(
        input_batch=batch,
        attn_metadata=None,
        slot_mappings=None,
        last_hidden_states=empty,
        aux_hidden_states=None,
        num_sampled=torch.ones(batch_size, dtype=torch.int32, device=device),
        num_rejected=torch.zeros(batch_size, dtype=torch.int32, device=device),
        last_sampled=row[-1:].expand(batch_size, 1).contiguous().to(torch.int64),
        next_prefill_tokens=empty,
        temperature=empty,
        seeds=empty,
        num_speculative_tokens=k,
    )
    expected = (
        [(seq_len - PATTERN_LENGTH + i) % PATTERN_LENGTH for i in range(k)]
        if workload == "repeat"
        else [seq_len - 1] * k
    )
    output = speculator.propose(**args)
    torch.testing.assert_close(output, torch.tensor([expected] * batch_size, dtype=torch.int64, device=device))
    for _ in range(10):
        speculator.propose(**args)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        speculator.propose(**args)
    end.record()
    end.synchronize()
    wall_us = (time.perf_counter() - wall_start) * 1e6 / iterations
    print(
        json.dumps(
            dict(
                batch_size=batch_size,
                seq_len=seq_len,
                capacity=capacity,
                k=k,
                workload=workload,
                device_us=start.elapsed_time(end) * 1000 / iterations,
                wall_us=wall_us,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 1024, 21000])
    parser.add_argument("--capacity", type=int, default=32768)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1 or not 1 <= args.k <= 15 or min(args.batch_sizes) < 1:
        parser.error("iterations and batch sizes must be positive; k must be in [1, 15]")
    if min(args.seq_lens) < PATTERN_LENGTH + 5 or max(args.seq_lens) > args.capacity:
        parser.error("sequence lengths must be in [37, capacity]")
    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lens:
            for workload in ("repeat", "unique"):
                benchmark(batch_size, seq_len, args.capacity, args.k, args.iterations, workload)


if __name__ == "__main__":
    main()
