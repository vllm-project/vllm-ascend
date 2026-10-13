# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Measure real-model MRV2 dynamic-K latency on Ascend.

Run the same command on the baseline and adapted revisions with the same
upstream vLLM and hardware. Compare latency, output tokens/s, and graph memory.
"""

import argparse
import json
import os
import statistics
import time

import torch
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--method", choices=["eagle", "eagle3", "mtp"], default="eagle")
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--runtime-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--graph-mode", choices=["NONE", "FULL_DECODE_ONLY"], default="FULL_DECODE_ONLY")
    args = parser.parse_args()
    if not 0 <= args.runtime_k <= args.max_k or args.max_k <= 0:
        parser.error("require max-k > 0 and 0 <= runtime-k <= max-k")
    if min(args.batch_size, args.input_len, args.output_len, args.iterations, args.tp_size) <= 0 or args.warmup < 0:
        parser.error("sizes and iterations must be positive; warmup must be nonnegative")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.input_len + args.output_len + args.max_k + 1,
        max_num_seqs=args.batch_size,
        enable_prefix_caching=False,
        async_scheduling=True,
        speculative_config={
            "model": args.draft_model,
            "method": args.method,
            "num_speculative_tokens": args.max_k,
            "num_speculative_tokens_per_batch_size": [(1, args.batch_size, args.runtime_k)],
        },
        compilation_config={
            "cudagraph_mode": args.graph_mode,
            "cudagraph_capture_sizes": sorted({args.batch_size, args.batch_size * (args.max_k + 1)}),
        },
    )
    tokens = llm.get_tokenizer().encode("Explain how speculative decoding improves language model inference.")
    prompts = [
        {"prompt_token_ids": (tokens * (args.input_len // len(tokens) + 1))[: args.input_len]}
        for _ in range(args.batch_size)
    ]
    sampling = SamplingParams(temperature=0, max_tokens=args.output_len, min_tokens=args.output_len, ignore_eos=True)
    times = []
    generated_tokens = []
    for iteration in range(args.warmup + args.iterations):
        torch.npu.synchronize()
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start
        count = sum(len(output.outputs[0].token_ids) for output in outputs)
        assert count == args.batch_size * args.output_len
        if iteration >= args.warmup:
            times.append(elapsed)
            generated_tokens.append(count)
    print(
        json.dumps(
            {
                "config": vars(args),
                "latency_seconds": times,
                "median_latency_seconds": statistics.median(times),
                "mean_latency_seconds": statistics.mean(times),
                "output_tokens_per_second": sum(generated_tokens) / sum(times),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
