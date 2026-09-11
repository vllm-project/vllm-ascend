# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Manual NPU smoke/benchmark for an already running Mooncake store.

Run on the same host as the model workers. See mooncake_hybrid_attention.md.
This is opt-in: it requires a hybrid model and external Mooncake services.
"""

import argparse
import json
import time
import uuid

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=32768)
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--prefetch-layers", type=int, default=1)
    parser.add_argument("--profile-dir")
    parser.add_argument(
        "--non-layerwise", action="store_true", help="Run the existing non-layerwise Mooncake baseline."
    )
    parser.add_argument("--engine-args", type=json.loads, default={}, help="Additional LLM arguments as a JSON object.")
    args = parser.parse_args()
    for name in ("tp", "prompt_tokens", "chunk_tokens", "max_new_tokens", "warm_runs", "prefetch_layers"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.chunk_tokens >= args.prompt_tokens:
        parser.error("--chunk-tokens must be smaller than --prompt-tokens to exercise chunked prefill")
    if not isinstance(args.engine_args, dict):
        parser.error("--engine-args must be a JSON object")
    reserved = {
        "model",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
        "kv_transfer_config",
        "profiler_config",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "enforce_eager",
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_model_len",
    }
    conflicting_args = sorted(reserved.intersection(args.engine_args))
    if conflicting_args:
        parser.error(f"Use the dedicated CLI options; reserved engine arguments: {conflicting_args}")
    return args


def main():
    args = parse_args()
    engine_args = dict(args.engine_args)
    if args.profile_dir:
        engine_args["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": args.profile_dir}
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=1,
        max_num_batched_tokens=args.chunk_tokens,
        max_model_len=args.prompt_tokens + args.max_new_tokens,
        kv_transfer_config=KVTransferConfig(
            kv_connector="AscendStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "backend": "mooncake",
                "use_layerwise": not args.non_layerwise,
                "layerwise_prefetch_layers": args.prefetch_layers,
                "layerwise_max_transfer_blocks": 64,
                "layerwise_max_transfer_bytes": 16777216,
            },
        ),
        **engine_args,
    )
    tokenizer = llm.get_tokenizer()
    # A fresh prefix avoids accidentally using objects from an earlier run.
    salt = tokenizer.encode(f"Mooncake hybrid validation {uuid.uuid4()}. ", add_special_tokens=False)
    body = tokenizer.encode(
        "Explain how a sliding window and compressed attention retain context. ", add_special_tokens=False
    )
    token_ids = (salt + body * (args.prompt_tokens // max(1, len(body)) + 1))[: args.prompt_tokens]
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, ignore_eos=True)
    reference = None
    results = []
    for run in range(args.warm_runs + 1):
        if run and not llm.reset_prefix_cache(reset_connector=False):
            raise RuntimeError("Could not clear the local prefix cache; cannot validate a remote hit")
        phase = "cold" if run == 0 else f"warm-{run}"
        if args.profile_dir:
            llm.start_profile(profile_prefix=f"mooncake-hybrid-{phase}")
        try:
            started = time.perf_counter()
            output = llm.generate([{"prompt_token_ids": token_ids}], sampling, use_tqdm=False)[0]
            elapsed = time.perf_counter() - started
        finally:
            if args.profile_dir:
                llm.stop_profile()
        generated = list(output.outputs[0].token_ids)
        if reference is None:
            reference = generated
        elif generated != reference:
            raise AssertionError(f"{phase}: remote-cache output differs from the cold-compute token sequence")
        cached_tokens = output.num_cached_tokens
        if run and (cached_tokens is None or cached_tokens <= 0):
            raise AssertionError(f"{phase}: no cache hit after clearing the local prefix cache")
        metrics = output.metrics
        ttft = None
        if metrics is not None and metrics.first_token_time is not None and metrics.arrival_time is not None:
            ttft = metrics.first_token_time - metrics.arrival_time
        row = {"phase": phase, "wall_seconds": elapsed, "ttft_seconds": ttft, "cached_tokens": cached_tokens}
        results.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"mode": "non-layerwise" if args.non_layerwise else "layerwise", "results": results}), flush=True)


if __name__ == "__main__":
    main()
