#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Convenience wrapper for generation-based PIQA evaluation on vllm-ascend.

This intentionally avoids lm-eval's multiple-choice loglikelihood path because
prompt logprobs are not reliable on the current Ascend V1 backend. Instead it:

1. Formats each PIQA validation example as a short A/B instruction prompt.
2. Runs standard generation with vLLM.
3. Parses the model response into label A or B.
4. Reports exact answer accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable

DEFAULT_MODELS = [
    "dense=/data/weights/Qwen3-32B",
    "quarot=/data/weights/Qwen3-32B-QuaRot-W4A4-no-attn-quant",
]
DEFAULT_MODEL_QUANT = [
    "quarot=ascend",
]
DEFAULT_SYSTEM_PROMPT = (
    "You are solving a two-choice commonsense reasoning task. "
    "Answer with exactly one character: A or B. Do not output any explanation or other text."
)


def parse_model_specs(specs: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --model spec {spec!r}; expected name=path")
        name, path = spec.split("=", maxsplit=1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --model spec {spec!r}; name/path must not be empty")
        parsed.append((name, path))
    return parsed


def parse_model_overrides(specs: list[str], *, kind: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid {kind} override {spec!r}; expected name=value")
        name, value = spec.split("=", maxsplit=1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Invalid {kind} override {spec!r}; name/value must not be empty")
        parsed[name] = value
    return parsed


def parse_optional_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


def bytes_to_gib(value: int) -> float:
    return value / (1024 ** 3)


def build_model_args(
    *,
    model_path: str,
    tensor_parallel_size: int,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
    enforce_eager: bool,
    quantization: str | None,
    enable_chunked_prefill: bool | None,
    enable_thinking: bool | None,
) -> str:
    model_args: dict[str, Any] = {
        "pretrained": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
        "enforce_eager": enforce_eager,
    }
    if quantization is not None:
        model_args["quantization"] = quantization
    if enable_chunked_prefill is not None:
        model_args["enable_chunked_prefill"] = enable_chunked_prefill
    if enable_thinking is not None:
        model_args["enable_thinking"] = enable_thinking
    return ",".join(f"{key}={value}" for key, value in model_args.items())


def cleanup_vllm_state() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
    except Exception:
        pass


def sample_npu_memory_usage() -> dict[str, Any] | None:
    visible_devices_str = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    visible_devices: set[int] | None = None
    if visible_devices_str:
        try:
            visible_devices = {
                int(device.strip()) for device in visible_devices_str.split(",") if device.strip()
            }
        except ValueError:
            visible_devices = None

    try:
        output = subprocess.check_output(
            ["npu-smi", "info"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None

    per_device_used_bytes: list[int] = []
    per_device_total_bytes: list[int] = []
    current_device: int | None = None
    current_device_selected = False

    for line in output.splitlines():
        first_line_match = re.match(r"^\|\s*(\d+)\s+\S+", line)
        if first_line_match:
            current_device = int(first_line_match.group(1))
            current_device_selected = visible_devices is None or current_device in visible_devices
            continue

        if current_device is None or not current_device_selected:
            continue

        hbm_match = re.search(r"(\d+)\s*/\s*(\d+)\s*$", line)
        if hbm_match:
            used_mb = int(hbm_match.group(1))
            total_mb = int(hbm_match.group(2))
            per_device_used_bytes.append(used_mb * 1024 * 1024)
            per_device_total_bytes.append(total_mb * 1024 * 1024)
            current_device = None
            current_device_selected = False

    if not per_device_total_bytes:
        return None

    return {
        "device_count": len(per_device_total_bytes),
        "per_device_used_bytes": per_device_used_bytes,
        "per_device_total_bytes": per_device_total_bytes,
        "total_used_bytes": sum(per_device_used_bytes),
        "total_capacity_bytes": sum(per_device_total_bytes),
    }


class NpuMemoryMonitor:

    def __init__(self, poll_interval_s: float = 0.5) -> None:
        self.poll_interval_s = poll_interval_s
        self.baseline = sample_npu_memory_usage()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_loop(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            snapshot = sample_npu_memory_usage()
            if snapshot is None:
                continue
            if self.peak is None or snapshot["total_used_bytes"] > self.peak["total_used_bytes"]:
                self.peak = snapshot

    def start(self) -> None:
        if self.baseline is None:
            return
        self._thread = threading.Thread(target=self._poll_loop, name="npu-memory-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any] | None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(self.poll_interval_s * 2, 1.0))
        if self.baseline is None or self.peak is None:
            return None
        baseline_total_used_bytes = int(self.baseline["total_used_bytes"])
        peak_total_used_bytes = int(self.peak["total_used_bytes"])
        return {
            "device_count": int(self.peak["device_count"]),
            "baseline_total_used_gib": bytes_to_gib(baseline_total_used_bytes),
            "peak_total_used_gib": bytes_to_gib(peak_total_used_bytes),
            "peak_increment_gib": bytes_to_gib(peak_total_used_bytes - baseline_total_used_bytes),
            "per_device_peak_used_gib": [
                bytes_to_gib(int(value)) for value in self.peak["per_device_used_bytes"]
            ],
            "per_device_total_capacity_gib": [
                bytes_to_gib(int(value)) for value in self.peak["per_device_total_bytes"]
            ],
        }


def load_piqa_examples(*, limit: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("baber/piqa", split="validation")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(example) for example in dataset]


def build_piqa_prompt(example: dict[str, Any]) -> str:
    return (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        f"Question: {example['goal']}\n"
        f"A. {example['sol1']}\n"
        f"B. {example['sol2']}\n\n"
        "Respond with exactly one character: A or B.\nAnswer:"
    )


def apply_chat_template_if_needed(
    prompts: list[str],
    *,
    model_path: str,
    trust_remote_code: bool,
    apply_chat_template: bool | None,
    enable_thinking: bool | None,
) -> list[str]:
    if not apply_chat_template:
        return prompts

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    rendered = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        rendered.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking if enable_thinking is not None else False,
            )
        )
    return rendered


def parse_ab_prediction(text: str) -> str | None:
    cleaned = text.strip()
    if not cleaned:
        return None

    leading_match = re.match(r"^[\s:()\[\]\"'`<>{}\-–—,.!?;]*([ABab])(?:\b|[^A-Za-z])", cleaned)
    if leading_match:
        return leading_match.group(1).upper()

    token_match = re.search(r"\b([ABab])\b", cleaned)
    if token_match:
        return token_match.group(1).upper()
    return None


def batch_iter(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for begin in range(0, len(items), batch_size):
        yield items[begin : begin + batch_size]


def score_predictions(
    *,
    examples: list[dict[str, Any]],
    predictions: list[str],
) -> tuple[float, list[dict[str, Any]]]:
    if len(examples) != len(predictions):
        raise ValueError(f"Mismatched lengths: {len(examples)} examples vs {len(predictions)} predictions")

    scored_examples: list[dict[str, Any]] = []
    num_correct = 0
    for index, (example, prediction) in enumerate(zip(examples, predictions)):
        gold_choice = "A" if int(example["label"]) == 0 else "B"
        parsed_prediction = parse_ab_prediction(prediction)
        correct = parsed_prediction == gold_choice
        num_correct += int(correct)
        scored_examples.append(
            {
                "index": index,
                "goal": example["goal"],
                "gold_choice": gold_choice,
                "prediction": prediction,
                "parsed_prediction": parsed_prediction,
                "correct": correct,
            }
        )
    accuracy = num_correct / len(examples) if examples else 0.0
    return accuracy, scored_examples


def run_model(
    *,
    name: str,
    model_path: str,
    args: argparse.Namespace,
    output_root: Path,
    tensor_parallel_size: int,
    quantization: str | None,
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    model_output_dir = output_root / name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    prompts = [build_piqa_prompt(example) for example in examples]
    prompts = apply_chat_template_if_needed(
        prompts,
        model_path=model_path,
        trust_remote_code=args.trust_remote_code,
        apply_chat_template=args.apply_chat_template,
        enable_thinking=args.enable_thinking,
    )

    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": True,
    }
    if quantization is not None:
        llm_kwargs["quantization"] = quantization
    if args.enable_chunked_prefill is not None:
        llm_kwargs["enable_chunked_prefill"] = args.enable_chunked_prefill
    if args.enable_thinking is not None:
        llm_kwargs["enable_thinking"] = args.enable_thinking

    model_args = build_model_args(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        quantization=quantization,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_thinking=args.enable_thinking,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["\n"],
        skip_special_tokens=True,
    )

    memory_monitor = NpuMemoryMonitor()
    memory_monitor.start()
    load_s = 0.0
    eval_s = 0.0
    predictions: list[str] = []
    llm = None
    try:
        load_begin = time.perf_counter()
        llm = LLM(**llm_kwargs)
        load_s = time.perf_counter() - load_begin
        eval_begin = time.perf_counter()
        if args.batch_size == "auto":
            outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
            predictions = [output.outputs[0].text for output in outputs]
        else:
            batch_size = int(args.batch_size)
            for prompt_batch in batch_iter(prompts, batch_size):
                outputs = llm.generate(prompt_batch, sampling_params, use_tqdm=False)
                predictions.extend(output.outputs[0].text for output in outputs)
        eval_s = time.perf_counter() - eval_begin
    finally:
        memory_summary = memory_monitor.stop()
        if llm is not None:
            del llm
        cleanup_vllm_state()

    accuracy, scored_examples = score_predictions(examples=examples, predictions=predictions)
    task_summary = {
        "primary_metric": "accuracy",
        "primary_value": accuracy,
        "results": {
            "accuracy": accuracy,
            "num_examples": len(examples),
            "num_correct": sum(int(example["correct"]) for example in scored_examples),
        },
    }

    result = {
        "name": name,
        "model_path": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "quantization": quantization,
        "load_s": load_s,
        "eval_s": eval_s,
        "npu_memory": memory_summary,
        "tasks": {"piqa": task_summary},
        "examples": scored_examples,
        "model_args": model_args,
    }
    (model_output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def compact_model_result(model_result: dict[str, Any], output_root: Path) -> dict[str, Any]:
    detailed_result_path = output_root / model_result["name"] / "result.json"
    return {
        "name": model_result["name"],
        "model_path": model_result["model_path"],
        "tensor_parallel_size": model_result["tensor_parallel_size"],
        "quantization": model_result["quantization"],
        "load_s": model_result["load_s"],
        "eval_s": model_result["eval_s"],
        "npu_memory": model_result["npu_memory"],
        "tasks": model_result["tasks"],
        "model_args": model_result["model_args"],
        "detailed_result_path": str(detailed_result_path),
    }


def build_comparisons(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(models) < 2:
        return []
    baseline = models[0]
    comparisons = []
    for candidate in models[1:]:
        base_task = baseline["tasks"]["piqa"]
        cand_task = candidate["tasks"]["piqa"]
        comparisons.append(
            {
                "baseline": baseline["name"],
                "candidate": candidate["name"],
                "eval_speedup": baseline["eval_s"] / candidate["eval_s"] if candidate["eval_s"] > 0 else 0.0,
                "task": {
                    "metric": "accuracy",
                    "baseline": base_task["primary_value"],
                    "candidate": cand_task["primary_value"],
                    "delta": cand_task["primary_value"] - base_task["primary_value"],
                },
            }
        )
    return comparisons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[], help="Model spec name=path. Can be repeated.")
    parser.add_argument(
        "--model-tp",
        action="append",
        default=[],
        help="Optional tensor parallel override name=int. Can be repeated per model.",
    )
    parser.add_argument(
        "--model-quant",
        action="append",
        default=[],
        help="Optional quantization override name=value, e.g. quarot=ascend.",
    )
    parser.add_argument("--tasks", default="piqa", help="Comma-separated tasks. Only piqa is supported.")
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/lm_eval_piqa"))
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--apply-chat-template", type=parse_optional_bool, default=None)
    parser.add_argument("--enable-chunked-prefill", type=parse_optional_bool, default=None)
    parser.add_argument("--enable-thinking", type=parse_optional_bool, default=None)
    parser.add_argument("--max-tokens", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("USE_MODELSCOPE_HUB", "0")

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if tasks != ["piqa"]:
        raise ValueError(f"Only piqa is supported by this helper, got: {tasks}")

    model_specs = parse_model_specs(args.model or DEFAULT_MODELS)
    model_tps = {name: int(value) for name, value in parse_model_overrides(args.model_tp, kind="tp").items()}
    model_quants = parse_model_overrides(args.model_quant or DEFAULT_MODEL_QUANT, kind="quantization")
    examples = load_piqa_examples(limit=args.limit)

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "tasks": tasks,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "enforce_eager": args.enforce_eager,
            "trust_remote_code": args.trust_remote_code,
            "apply_chat_template": args.apply_chat_template,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "enable_thinking": args.enable_thinking,
            "max_tokens": args.max_tokens,
            "model_tps": model_tps,
            "model_quants": model_quants,
        },
        "models": [],
    }

    for name, model_path in model_specs:
        tensor_parallel_size = model_tps.get(name, 1)
        quantization = model_quants.get(name)
        print(f"[{name}] model={model_path}")
        print(f"[{name}] tensor_parallel_size={tensor_parallel_size}")
        print(f"[{name}] quantization={quantization}")
        model_result = run_model(
            name=name,
            model_path=model_path,
            args=args,
            output_root=args.output_root,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
            examples=examples,
        )
        results["models"].append(compact_model_result(model_result, args.output_root))
        task = model_result["tasks"]["piqa"]
        print(
            f"[{name}] piqa:{task['primary_metric']}={task['primary_value']:.4f}, "
            f"load_s={model_result['load_s']:.3f}, "
            f"eval_s={model_result['eval_s']:.3f}"
        )

    results["comparisons"] = build_comparisons(results["models"])
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"summary_path={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
