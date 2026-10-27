# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Record one reproducible Qwen3.5 dense MTP accuracy/performance run.

Run V1 and V2 in separate processes with VLLM_USE_V2_MODEL_RUNNER=0/1.
This script records observations; it does not declare regression thresholds.
Logprob collection is confined to accuracy runs, outside performance timing.
"""

import argparse
import dataclasses
import importlib.metadata
import json
import os
import platform
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

KV_CACHE_SHARED_PREFIX_REPETITIONS = 160
KV_CACHE_CHECK_TOKENS = 16


def qualified_name(value):
    if value is None:
        return None
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__name__}"


def worker_memory(worker, reset=False):
    """Collect worker-local memory outside the timed generation calls."""
    import torch

    torch.npu.synchronize()
    if reset:
        torch.npu.reset_peak_memory_stats()
    free, total = torch.npu.mem_get_info()
    return {
        "device": str(worker.device),
        "device_name": torch.npu.get_device_name(),
        "device_hbm_used_bytes": total - free,
        "torch_allocated_bytes": torch.npu.memory_allocated(),
        "torch_reserved_bytes": torch.npu.memory_reserved(),
        "torch_peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.npu.max_memory_reserved(),
    }


class QwenMtpTestWorkerExtension:
    """Expose named RPCs without enabling pickle-based RPC serialization."""

    model_runner: Any

    def comparison_memory_snapshot(self, reset=False):
        return worker_memory(self, reset=reset)

    def comparison_runtime_snapshot(self):
        runner = self.model_runner
        speculator = getattr(runner, "speculator", None)
        drafter = getattr(runner, "drafter", None)
        kv_cache_config = getattr(runner, "kv_cache_config", None)
        cache_config = getattr(runner, "cache_config", None)
        block_tables = getattr(runner, "block_tables", None)
        graph_managers = {}
        for owner_name, owner in (("runner", runner), ("speculator", speculator), ("drafter", drafter)):
            if owner is None:
                continue
            for name, value in vars(owner).items():
                if "graph" in name and value is not None:
                    graph_managers[f"{owner_name}.{name}"] = (
                        value if isinstance(value, (bool, int, float, str)) else qualified_name(value)
                    )
        groups = getattr(kv_cache_config, "kv_cache_groups", []) or []
        return {
            "use_v2_model_runner": bool(getattr(self, "use_v2_model_runner", False)),
            "runner_class": qualified_name(runner),
            "speculator_class": qualified_name(speculator),
            "drafter_class": qualified_name(drafter),
            "rejection_sampler_class": qualified_name(getattr(runner, "rejection_sampler", None)),
            "speculative_method": getattr(getattr(runner, "speculative_config", None), "method", None),
            "graph_managers": graph_managers,
            "kv_cache_groups": [
                {
                    "spec_class": qualified_name(getattr(group, "kv_cache_spec", None)),
                    "layer_count": len(getattr(group, "layer_names", []) or []),
                    "block_size": getattr(getattr(group, "kv_cache_spec", None), "block_size", None),
                }
                for group in groups
            ],
            "kv_cache_num_blocks": getattr(kv_cache_config, "num_blocks", None),
            "block_tables_class": qualified_name(block_tables),
            "kernel_block_sizes": [int(size) for size in (getattr(block_tables, "kernel_block_sizes", None) or [])],
            "cache_block_size": getattr(cache_config, "block_size", None),
            "cache_layout": getattr(cache_config, "kv_cache_layout", None),
            "enable_prefix_caching": getattr(cache_config, "enable_prefix_caching", None),
        }


def metric_snapshot(llm):
    return [dataclasses.asdict(metric) for metric in llm.get_metrics()]


def output_record(output, include_logprobs):
    completion = output.outputs[0]
    record = {
        "prompt_token_count": len(output.prompt_token_ids),
        "token_ids": list(completion.token_ids),
        "text": completion.text,
        "finish_reason": completion.finish_reason,
        "metrics": dataclasses.asdict(output.metrics) if output.metrics else None,
    }
    if include_logprobs:
        record["logprobs"] = [
            {str(token): {"logprob": value.logprob, "rank": value.rank} for token, value in step.items()}
            for step in completion.logprobs
        ]
    return record


def performance_request_record(output):
    completion = output.outputs[0]
    return {
        "output_token_count": len(completion.token_ids),
        "metrics": dataclasses.asdict(output.metrics) if output.metrics else None,
    }


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(args, result):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    questions = [
        ("english_short", "What is 6 plus 7? Give the final integer."),
        ("chinese_short", "中国的首都是哪里？请简短回答。"),
        ("code", "Write a Python function that adds two integers."),
        (
            "english_long",
            "A notebook contains observations about forests, rivers and mountains. " * 300
            + "\nSummarize the three types of places mentioned in one sentence.",
        ),
        (
            "chinese_long",
            "这是一份观察记录，内容涉及森林、河流与山脉。" * 140 + "\n请用一句话概括记录提到的三类自然景观。",
        ),
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for _, question in questions
    ]
    result["prompt_categories"] = [name for name, _ in questions]
    compilation: dict[str, Any] = {"cudagraph_mode": "NONE"}
    if args.graph:
        compilation = {
            "mode": 3,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [(args.spec_tokens + 1) * batch for batch in args.batches],
        }
    loaded_at = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=2,
        max_model_len=8192,
        max_num_seqs=max(args.batches),
        max_num_batched_tokens=8192,
        gpu_memory_utilization=args.gpu_memory_utilization,
        distributed_executor_backend="mp",
        worker_extension_cls="compare_qwen35_mtp.QwenMtpTestWorkerExtension",
        enable_prefix_caching=args.kv_cache_check,
        disable_log_stats=False,
        enforce_eager=not args.graph,
        seed=args.seed,
        compilation_config=compilation,
        speculative_config={
            "method": args.spec_method,
            "num_speculative_tokens": args.spec_tokens,
            "enforce_eager": True,
        },
        max_logprobs=args.probe_logprobs if args.probe_only else 20,
    )
    result["initialization_seconds"] = time.perf_counter() - loaded_at
    try:
        run_workloads(args, result, llm, tokenizer, prompts)
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=5)


def run_workloads(args, result, llm, tokenizer, prompts):
    result["memory_after_initialization"] = llm.collective_rpc("comparison_memory_snapshot")
    result["runtime_snapshot"] = llm.collective_rpc("comparison_runtime_snapshot")
    if args.kv_cache_check:
        run_kv_cache_check(args, result, llm, tokenizer)
        return
    if args.probe_only:
        probe_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 6 plus 7? Give the final integer."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        outputs = llm.generate(
            [probe_prompt],
            SamplingParams(temperature=0, max_tokens=8, logprobs=args.probe_logprobs, seed=args.seed),
            use_tqdm=False,
        )
        result["full_logits_probe"] = [output_record(outputs[0], True)]
        result["status"] = "complete"
        write_result(args.output, result)
        return
    result["status"] = "running"
    write_result(args.output, result)
    result["accuracy"] = []
    for temperature in (0.0, 0.8):
        before = metric_snapshot(llm)
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=temperature,
                top_p=1.0,
                top_k=-1,
                seed=args.seed,
                max_tokens=args.accuracy_tokens,
                logprobs=20,
            ),
            use_tqdm=False,
        )
        records = [output_record(output, True) for output in outputs]
        if len(records) != len(prompts) or not all(record["token_ids"] for record in records):
            raise RuntimeError("Missing accuracy output")
        result["accuracy"].append(
            {
                "temperature": temperature,
                "outputs": records,
                "metrics_before": before,
                "metrics_after": metric_snapshot(llm),
            }
        )
        write_result(args.output, result)

    # The same 128 input tokens and fixed output length are used for every batch.
    perf_ids = tokenizer.encode(
        "Summarize the following observations: " + "Forests, rivers and mountains are natural landscapes. " * 30
    )[:128]
    perf_prompt = {"prompt_token_ids": perf_ids}
    sampling = SamplingParams(temperature=0, max_tokens=args.performance_tokens, ignore_eos=True, seed=args.seed)
    result["performance"] = []
    for batch in args.batches:
        batch_prompts = [perf_prompt] * batch
        llm.generate(batch_prompts, sampling, use_tqdm=False)
        llm.collective_rpc("comparison_memory_snapshot", kwargs={"reset": True})
        before = metric_snapshot(llm)
        trials = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            outputs = llm.generate(batch_prompts, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - start
            records = [performance_request_record(output) for output in outputs]
            if len(records) != batch or any(
                record["output_token_count"] != args.performance_tokens for record in records
            ):
                raise RuntimeError("Performance trial did not produce the requested fixed output length")
            trials.append(
                {
                    "batch_wall_seconds": elapsed,
                    "output_tokens_per_second": batch * args.performance_tokens / elapsed,
                    "requests": records,
                }
            )
        result["performance"].append(
            {
                "batch_size": batch,
                "input_tokens": len(perf_ids),
                "trials": trials,
                "median_output_tokens_per_second": statistics.median(
                    trial["output_tokens_per_second"] for trial in trials
                ),
                "memory_after_trials": llm.collective_rpc("comparison_memory_snapshot"),
                "metrics_before": before,
                "metrics_after": metric_snapshot(llm),
            }
        )
        write_result(args.output, result)
    result["status"] = "complete"
    write_result(args.output, result)


def run_kv_cache_check(args, result, llm, tokenizer):
    shared_text = "A durable cache-prefix record contains forests, rivers, and mountains. " * (
        KV_CACHE_SHARED_PREFIX_REPETITIONS
    )
    shared_ids = tokenizer.encode(shared_text, add_special_tokens=False)
    prompt_a = {
        "prompt_token_ids": shared_ids + tokenizer.encode("\nAnswer the first suffix.", add_special_tokens=False)
    }
    prompt_b = {
        "prompt_token_ids": shared_ids + tokenizer.encode("\nAnswer the second suffix.", add_special_tokens=False)
    }
    cases = [("seed", prompt_a), ("repeat", prompt_a), ("shared_prefix", prompt_b)]
    sampling = SamplingParams(temperature=0, max_tokens=KV_CACHE_CHECK_TOKENS, seed=args.seed)
    records = []
    for name, prompt in cases:
        before = metric_snapshot(llm)
        output = llm.generate([prompt], sampling, use_tqdm=False)[0]
        after = metric_snapshot(llm)
        records.append(
            {
                "name": name,
                "prompt_token_count": len(prompt["prompt_token_ids"]),
                "output": output_record(output, False),
                "metrics_before": before,
                "metrics_after": after,
            }
        )
    result["kv_cache_check"] = {
        "shared_prefix_token_count": len(shared_ids),
        "tokens_per_case": KV_CACHE_CHECK_TOKENS,
        "cases": records,
    }
    result["status"] = "complete"
    write_result(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec-tokens", type=int, choices=(1, 3, 5), required=True)
    parser.add_argument("--spec-method", default="qwen3_5_mtp")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--accuracy-tokens", type=int, default=64)
    parser.add_argument("--performance-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--probe-only", action="store_true", help="Run only the short logits probe.")
    parser.add_argument("--kv-cache-check", action="store_true", help="Check exact and partial prefix-cache reuse.")
    parser.add_argument(
        "--probe-logprobs",
        type=int,
        default=1000,
        help="Number of ranked logprobs returned by --probe-only; use -1 to request all tokens.",
    )
    args = parser.parse_args()
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") not in ("0", "1"):
        parser.error("Set VLLM_USE_V2_MODEL_RUNNER explicitly to 0 or 1")
    if not args.batches or min(args.batches) < 1 or args.repeats < 1:
        parser.error("Batch sizes and repeat count must be positive")
    result = {
        "status": "initializing",
        "runner": int(os.environ["VLLM_USE_V2_MODEL_RUNNER"]) + 1,
        "spec_tokens": args.spec_tokens,
        "graph": args.graph,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "host": platform.node(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "triton-ascend", "transformers")
        },
        "measurement_scope": (
            "Offline engine latency and throughput; no HTTP transport. "
            "HBM snapshot and Torch allocator peaks are separate."
        ),
    }
    try:
        run(args, result)
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        write_result(args.output, result)
        raise


if __name__ == "__main__":
    main()
