# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Manual Ascend/Mooncake PP acceptance: recompute, save-only, then load-only.

Each mode runs in a fresh process. Use identical engine args and prompt salt,
a running Mooncake Master and MOONCAKE_CONFIG_PATH. Reports retain token IDs,
external hit evidence (after clearing local prefix cache), and latency samples.
This is a correctness/latency smoke test, not a serving throughput benchmark.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


def reset_local_cache(llm, timeout=30):
    deadline = time.monotonic() + timeout
    while not llm.reset_prefix_cache(reset_connector=False):
        if time.monotonic() >= deadline:
            raise RuntimeError("Local prefix cache reset failed; remote-hit evidence would be ambiguous")
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--mode", choices=("baseline", "save", "load"), required=True)
    parser.add_argument("--prompt-salt", required=True, help="Use a fresh salt for each three-process acceptance run")
    parser.add_argument("--prompt-tokens", type=int, default=32769)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prefetch-layers", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--engine-args", default="{}", help="JSON engine args, including quantization/MTP if needed")
    parser.add_argument("--reference", type=Path, help="The no-connector baseline JSON; required for save/load")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("prompt_tokens", "output_tokens", "batch_size", "rounds", "prefetch_layers"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.mode != "baseline" and args.reference is None:
        parser.error("save/load require --reference from a separate no-connector run")
    extra = json.loads(args.engine_args)
    if not isinstance(extra, dict) or "kv_transfer_config" in extra:
        parser.error("--engine-args must be a JSON object without kv_transfer_config")

    # Import only after argument parsing, so --help works off-device and the
    # script can be launched in an isolated engine process for each control.
    from vllm import LLM, SamplingParams

    engine_args = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "pipeline_parallel_size": args.pp,
        "distributed_executor_backend": "mp",
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.prompt_tokens + args.output_tokens + 1024,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": False,
        "seed": 0,
        **extra,
    }
    comparison_config = dict(engine_args)
    if args.mode != "baseline":
        engine_args["kv_transfer_config"] = {
            "kv_connector": "AscendStoreConnector",
            "kv_role": "kv_producer" if args.mode == "save" else "kv_consumer",
            "kv_connector_extra_config": {
                "backend": "mooncake",
                "use_layerwise": True,
                "layerwise_prefetch_layers": args.prefetch_layers,
                "consumer_is_to_load": True,
            },
        }
    llm = LLM(**engine_args)
    tokenizer = llm.get_tokenizer()
    body = tokenizer.encode(
        "A pipeline assigns different transformer layers to different workers. "
        "A reusable prefix must restore the cache for every layer before attention reads it. ",
        add_special_tokens=False,
    )
    prompts = []
    for index in range(args.batch_size):
        # Differ in the first cache block, so another request in this batch
        # cannot supply a local-prefix hit that looks like a remote hit.
        prefix = tokenizer.encode(f"{index}: validation {args.prompt_salt}. ", add_special_tokens=False)
        suffix = tokenizer.encode("\nExplain why an incomplete cache cannot be used:", add_special_tokens=False)
        remaining = args.prompt_tokens - len(prefix) - len(suffix)
        if remaining < 1 or not body:
            raise ValueError("Increase --prompt-tokens")
        tokens = prefix + (body * ((remaining + len(body) - 1) // len(body)))[:remaining] + suffix
        prompts.append({"prompt_token_ids": tokens})
    prompt_digest = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference is not None:
        if reference["mode"] != "baseline" or not reference["passed"]:
            raise ValueError("Reference must be a successful no-connector baseline")
        if reference["prompt_digest"] != prompt_digest or reference["engine_args"] != comparison_config:
            raise ValueError("Reference prompt or engine configuration differs")
        if reference["output_tokens"] != args.output_tokens:
            raise ValueError("Reference output length differs")
    report = {
        "mode": args.mode,
        "engine_args": comparison_config,
        "prompt_digest": prompt_digest,
        "output_tokens": args.output_tokens,
        "prefetch_layers": args.prefetch_layers,
        "rounds": [],
        "passed": False,
    }
    sampling = SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True)
    expected = reference["rounds"][0]["token_ids"] if reference else None
    errors = []
    for round_id in range(args.rounds):
        reset_local_cache(llm)
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - started
        token_ids = [list(output.outputs[0].token_ids) for output in outputs]
        cached = [getattr(output, "num_cached_tokens", None) for output in outputs]
        ttft = []
        for output in outputs:
            metrics = output.metrics
            if metrics is not None and metrics.first_token_time is not None:
                ttft.append(metrics.first_token_time - metrics.arrival_time)
        if expected is None:
            expected = token_ids
        if token_ids != expected:
            errors.append(f"round {round_id}: generated token IDs differ from the no-connector baseline")
        if args.mode == "load" and any(value is None or value <= 0 for value in cached):
            errors.append(f"round {round_id}: no proven remote hit for every request: {cached}")
        if args.mode != "load" and any(value not in (None, 0) for value in cached):
            errors.append(f"round {round_id}: unexpected prefix hit in recompute control: {cached}")
        report["rounds"].append(
            {
                "token_ids": token_ids,
                "cached_tokens": cached,
                "elapsed_s": elapsed,
                "ttft_s": ttft,
                "output_tokens_per_s": sum(map(len, token_ids)) / elapsed,
            }
        )
        print(f"round={round_id} mode={args.mode} seconds={elapsed:.3f} cached_tokens={cached}", flush=True)
    # Exclude the first measured round from the summary: kernel warmup can
    # otherwise dominate the baseline while pooled runs appear faster.
    steady = report["rounds"][1:] or report["rounds"]
    report["steady_median_elapsed_s"] = statistics.median(row["elapsed_s"] for row in steady)
    report["errors"] = errors
    report["passed"] = not errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report={args.output} passed={report['passed']}", flush=True)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
