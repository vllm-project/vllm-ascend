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
"""Convenience wrapper around lm-eval for PIQA-style dense vs quantized runs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_MODELS = [
    "dense=/data/weights/Qwen3-32B",
    "quarot=/data/weights/Qwen3-32B-QuaRot-W4A4-no-attn-quant",
]
DEFAULT_MODEL_QUANT = [
    "quarot=ascend",
]


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


def pick_primary_metric(results: dict[str, Any], task_name: str) -> tuple[str, float]:
    task_result = results["results"][task_name]
    for metric_name in ("acc_norm,none", "acc,none", "exact_match,strict-match", "exact_match,flexible-extract"):
        value = task_result.get(metric_name)
        if isinstance(value, (int, float)):
            return metric_name, float(value)
    for metric_name, value in task_result.items():
        if metric_name.endswith("_stderr"):
            continue
        if isinstance(value, (int, float)):
            return metric_name, float(value)
    raise ValueError(f"Could not find a numeric primary metric for task {task_name}")


def cleanup_vllm_state() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
    except Exception:
        pass


def run_model(
    *,
    name: str,
    model_path: str,
    tasks: list[str],
    args: argparse.Namespace,
    output_root: Path,
    tensor_parallel_size: int,
    quantization: str | None,
) -> dict[str, Any]:
    import lm_eval

    model_output_dir = output_root / name
    model_output_dir.mkdir(parents=True, exist_ok=True)

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

    eval_params: dict[str, Any] = {
        "model": "vllm",
        "model_args": model_args,
        "tasks": tasks,
        "batch_size": args.batch_size,
    }
    if args.limit is not None:
        eval_params["limit"] = args.limit
    if args.num_fewshot is not None:
        eval_params["num_fewshot"] = args.num_fewshot
    if args.apply_chat_template is not None:
        eval_params["apply_chat_template"] = args.apply_chat_template
    if args.fewshot_as_multiturn is not None:
        eval_params["fewshot_as_multiturn"] = args.fewshot_as_multiturn

    begin = time.perf_counter()
    raw_results = lm_eval.simple_evaluate(**eval_params)
    eval_s = time.perf_counter() - begin

    task_summaries: dict[str, Any] = {}
    for task_name in tasks:
        primary_metric, primary_value = pick_primary_metric(raw_results, task_name)
        task_summaries[task_name] = {
            "primary_metric": primary_metric,
            "primary_value": primary_value,
            "results": raw_results["results"][task_name],
        }

    result = {
        "name": name,
        "model_path": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "quantization": quantization,
        "eval_s": eval_s,
        "tasks": task_summaries,
        "raw_results": raw_results,
        "model_args": model_args,
    }
    (model_output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    cleanup_vllm_state()
    return result


def build_comparisons(models: list[dict[str, Any]], tasks: list[str]) -> list[dict[str, Any]]:
    if len(models) < 2:
        return []
    baseline = models[0]
    comparisons = []
    for candidate in models[1:]:
        task_deltas = {}
        for task_name in tasks:
            base_task = baseline["tasks"][task_name]
            cand_task = candidate["tasks"][task_name]
            if base_task["primary_metric"] != cand_task["primary_metric"]:
                metric_name = f"{base_task['primary_metric']} vs {cand_task['primary_metric']}"
            else:
                metric_name = base_task["primary_metric"]
            task_deltas[task_name] = {
                "metric": metric_name,
                "baseline": base_task["primary_value"],
                "candidate": cand_task["primary_value"],
                "delta": cand_task["primary_value"] - base_task["primary_value"],
            }
        comparisons.append(
            {
                "baseline": baseline["name"],
                "candidate": candidate["name"],
                "eval_speedup": baseline["eval_s"] / candidate["eval_s"] if candidate["eval_s"] > 0 else 0.0,
                "tasks": task_deltas,
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
    parser.add_argument("--tasks", default="piqa", help="Comma-separated lm-eval tasks. Default: piqa")
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/lm_eval_piqa"))
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--apply-chat-template", type=parse_optional_bool, default=None)
    parser.add_argument("--fewshot-as-multiturn", type=parse_optional_bool, default=None)
    parser.add_argument("--enable-chunked-prefill", type=parse_optional_bool, default=None)
    parser.add_argument("--enable-thinking", type=parse_optional_bool, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("USE_MODELSCOPE_HUB", "0")

    model_specs = parse_model_specs(args.model or DEFAULT_MODELS)
    model_tps = {name: int(value) for name, value in parse_model_overrides(args.model_tp, kind="tp").items()}
    model_quants = parse_model_overrides(args.model_quant or DEFAULT_MODEL_QUANT, kind="quantization")
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise ValueError("At least one task must be provided.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "tasks": tasks,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "num_fewshot": args.num_fewshot,
            "enforce_eager": args.enforce_eager,
            "trust_remote_code": args.trust_remote_code,
            "apply_chat_template": args.apply_chat_template,
            "fewshot_as_multiturn": args.fewshot_as_multiturn,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "enable_thinking": args.enable_thinking,
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
            tasks=tasks,
            args=args,
            output_root=args.output_root,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
        )
        results["models"].append(model_result)
        task_parts = [
            f"{task_name}:{model_result['tasks'][task_name]['primary_metric']}="
            f"{model_result['tasks'][task_name]['primary_value']:.4f}"
            for task_name in tasks
        ]
        print(f"[{name}] {'; '.join(task_parts)}, eval_s={model_result['eval_s']:.3f}")

    results["comparisons"] = build_comparisons(results["models"], tasks)
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"summary_path={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
