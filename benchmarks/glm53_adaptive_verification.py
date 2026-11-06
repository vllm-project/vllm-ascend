# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare full-checkpoint plain, fixed-MTP and adaptive-MTP serving offline.

Run each mode in a fresh process with identical arguments. Performance runs do
not request logprobs. Separate greedy accuracy runs retain tokens and logprobs.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import time
from pathlib import Path


def _revision():
    revision = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, check=True).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    return {
        "commit": revision,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_sha256": {
            name.decode(): hashlib.sha256(Path(name.decode()).read_bytes()).hexdigest() for name in untracked if name
        },
    }


class BenchmarkWorker:
    """Read-only per-rank memory measurements through the worker extension."""

    def adaptive_benchmark_memory(self):
        import torch

        return {
            "allocated_bytes": torch.npu.memory_allocated(),
            "peak_allocated_bytes": torch.npu.max_memory_allocated(),
        }


def compare_accuracy(reference, actual):
    left_settings = {k: v for k, v in reference["settings"].items() if k != "speculative_config"}
    right_settings = {k: v for k, v in actual["settings"].items() if k != "speculative_config"}
    if left_settings != right_settings:
        raise ValueError("Reference and candidate must use identical engine settings except speculative_config")
    differences = []
    max_logprob_delta = 0.0
    for expected, observed in zip(reference["cases"], actual["cases"], strict=True):
        if expected["batch_size"] != observed["batch_size"] or expected["prefix_len"] != observed["prefix_len"]:
            raise ValueError("Reference and candidate must use identical workload cases")
        for request, (left, right) in enumerate(zip(expected["accuracy"], observed["accuracy"], strict=True)):
            if left["token_ids"] != right["token_ids"]:
                differences.append(
                    {"batch_size": observed["batch_size"], "prefix_len": observed["prefix_len"], "request": request}
                )
            for a, b, a_lp, b_lp in zip(left["token_ids"], right["token_ids"], left["logprobs"], right["logprobs"]):
                if a != b:
                    break
                if not math.isfinite(a_lp) or not math.isfinite(b_lp):
                    raise ValueError("Accuracy logprobs must be finite")
                max_logprob_delta = max(max_logprob_delta, abs(a_lp - b_lp))
    return {"greedy_token_mismatches": differences, "max_shared_prefix_logprob_delta": max_logprob_delta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("plain", "fixed", "adaptive"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--prefix-lens", type=int, nargs="+", default=[512, 8192, 32768])
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--engine-args", type=Path, help="JSON with deployment-specific LLM arguments, e.g. quantization"
    )
    args = parser.parse_args()
    if min(*args.batch_sizes, *args.prefix_lens, args.output_len, args.repeats) < 1 or args.warmups < 0:
        parser.error("Workload sizes and repeats must be positive; warmups must be nonnegative")

    import torch
    import torch_npu  # noqa: F401
    from vllm import LLM, SamplingParams

    settings = {
        "model": args.model,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": True,
        "seed": 60229,
        "enable_prefix_caching": False,
        "block_size": 128,
        "max_num_seqs": max(args.batch_sizes),
        "max_model_len": max(args.prefix_lens) + 34 + args.output_len + args.num_speculative_tokens,
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "worker_extension_cls": "benchmarks.glm53_adaptive_verification.BenchmarkWorker",
    }
    if args.engine_args is not None:
        settings.update(json.loads(args.engine_args.read_text()))
    if args.mode != "plain":
        settings["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
            "enable_adaptive_verification": args.mode == "adaptive",
        }
    else:
        settings.pop("speculative_config", None)
    result = {
        "mode": args.mode,
        "settings": settings,
        "revision": _revision(),
        "versions": {name: importlib.metadata.version(name) for name in ("vllm", "vllm-ascend", "torch", "torch-npu")},
        "hardware": subprocess.run(["npu-smi", "info"], text=True, capture_output=True, check=True).stdout,
        "cases": [],
    }
    engine = LLM(**settings)
    tokenizer = engine.get_tokenizer()
    text = "Explain the reasoning behind the following scientific result and verify each step carefully. "
    source = tokenizer.encode(text, add_special_tokens=False)
    if not source:
        raise ValueError("The benchmark prompt must tokenize to at least one token")
    perf_params = SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True, detokenize=False)
    accuracy_params = SamplingParams(
        temperature=0, max_tokens=args.output_len, ignore_eos=True, detokenize=False, logprobs=1
    )
    for prefix in args.prefix_lens:
        for batch in args.batch_sizes:
            prompts = []
            for request in range(batch):
                length = prefix + 17 * (request % 3)
                rotated = source[request % len(source) :] + source[: request % len(source)]
                tokens = (rotated * ((length + len(rotated) - 1) // len(rotated)))[:length]
                prompts.append({"prompt_token_ids": tokens})
            accuracy = engine.generate(prompts, accuracy_params, use_tqdm=False)
            for _ in range(args.warmups):
                engine.generate(prompts, perf_params, use_tqdm=False)
            elapsed = []
            output_tokens = []
            for _ in range(args.repeats):
                torch.npu.synchronize()
                start = time.perf_counter()
                responses = engine.generate(prompts, perf_params, use_tqdm=False)
                torch.npu.synchronize()
                elapsed.append(time.perf_counter() - start)
                output_tokens.append(sum(len(response.outputs[0].token_ids) for response in responses))
            result["cases"].append(
                {
                    "batch_size": batch,
                    "prefix_len": prefix,
                    "elapsed_seconds": elapsed,
                    "output_tokens": output_tokens,
                    "median_output_tokens_per_second": statistics.median(n / t for n, t in zip(output_tokens, elapsed)),
                    "accuracy": [
                        {
                            "token_ids": list(response.outputs[0].token_ids),
                            "logprobs": [
                                row[token].logprob
                                for token, row in zip(
                                    response.outputs[0].token_ids, response.outputs[0].logprobs, strict=True
                                )
                            ],
                        }
                        for response in accuracy
                    ],
                }
            )
            result["memory_by_rank"] = engine.collective_rpc("adaptive_benchmark_memory")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
    if args.reference is not None:
        result["comparison"] = compare_accuracy(json.loads(args.reference.read_text()), result)
        args.output.write_text(json.dumps(result, indent=2))
        if result["comparison"]["greedy_token_mismatches"]:
            raise RuntimeError(f"Greedy token mismatches: see {args.output}")


if __name__ == "__main__":
    main()
