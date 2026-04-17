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

"""Offline profiling utility for quantized models (e.g. FlatQuant and QuaRot).

This script profiles one-token generation using vLLM's torch profiler integration
and reports:
1) End-to-end inference latency.
2) Operator-level runtime from `op_statistic.csv`.
3) Component-level runtime (attention/ffn/norm/quantization/etc.) via OP mapping.
4) FFN-focused heuristic breakdown for QuaRot/W4A4 debugging.
"""

# isort: skip_file
import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from vllm_ascend.quantization.quarot_kv_cache import iter_kv_cache_records

DEFAULT_MODEL_SPECS = [
    "flatquant=/data/weights/Qwen3-32B-W4A4",
    "quarot=/data/weights/Qwen3-32B-W4A4-quarot-mock",
]


def parse_optional_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile quantized model inference with per-component summaries.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=("Model spec as name=path. Can be provided multiple times. Default: flatquant and quarot server paths."),
    )
    parser.add_argument("--dtype", default="float16", help="Runtime compute dtype, e.g. float16 or bfloat16.")
    parser.add_argument("--max-model-len", type=int, default=64, help="Maximum model length.")
    parser.add_argument("--prompt", default="Hello", help="Prompt text.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of prompts to generate in one batch.")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="If > 0, build tokenized prompts of exactly this input length instead of using raw text prompts.",
    )
    parser.add_argument("--max-tokens", type=int, default=1, help="Maximum generated tokens per request.")
    parser.add_argument("--warmup-iters", type=int, default=1, help="Warmup iterations before profiling.")
    parser.add_argument("--profile-iters", type=int, default=1, help="Profiled iterations per model.")
    parser.add_argument(
        "--output-root",
        default="/tmp/vllm_quant_profile",
        help="Directory where raw profiler traces and summary JSON are written.",
    )
    parser.add_argument("--topk-ops", type=int, default=12, help="Number of top ops to include in report.")
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile/ACL graph capture to reduce startup overhead while debugging.",
    )
    parser.add_argument(
        "--disable-kv-write",
        action="store_true",
        help="Skip KV-cache writes during profiling to isolate prefill-side compute.",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        type=parse_optional_bool,
        default=None,
        help=(
            "Override vLLM chunked prefill behavior for this run. "
            "Accepts true/false. If omitted, uses the runtime default."
        ),
    )
    return parser.parse_args()


def parse_model_specs(model_specs: list[str]) -> list[tuple[str, str]]:
    specs = model_specs if model_specs else DEFAULT_MODEL_SPECS
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --model spec '{spec}', expected name=path.")
        name, path = spec.split("=", maxsplit=1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --model spec '{spec}', name/path must not be empty.")
        parsed.append((name, path))
    return parsed


def _to_float(value: str | None) -> float:
    if value is None:
        return 0.0
    value = value.strip()
    if value == "":
        return 0.0
    return float(value)


def _to_int(value: str | None) -> int:
    if value is None:
        return 0
    value = value.strip()
    if value == "":
        return 0
    return int(float(value))


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_op_statistic_csv(csv_path: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(csv_path)
    parsed = []
    for row in rows:
        parsed.append(
            {
                "op_type": row.get("OP Type", ""),
                "count": _to_int(row.get("Count")),
                "total_time_us": _to_float(row.get("Total Time(us)")),
                "avg_time_us": _to_float(row.get("Avg Time(us)")),
                "ratio_pct": _to_float(row.get("Ratio(%)")),
            }
        )
    return parsed


def parse_step_trace_csv(csv_path: Path) -> dict[str, float]:
    rows = _read_csv_rows(csv_path)
    if not rows:
        return {}
    first = rows[0]
    result: dict[str, float] = {}
    for key, value in first.items():
        if key in ("Device_id", "Step"):
            continue
        result[key] = _to_float(value)
    return result


def summarize_step_gaps(step_trace: dict[str, float]) -> dict[str, float]:
    stage_us = step_trace.get("Stage", 0.0)
    computing_us = step_trace.get("Computing", 0.0)
    free_us = step_trace.get("Free", 0.0)
    preparing_us = step_trace.get("Preparing", 0.0)
    non_compute_gap_us = max(stage_us - computing_us, 0.0)
    other_gap_us = max(non_compute_gap_us - free_us - preparing_us, 0.0)
    return {
        "stage_us": stage_us,
        "computing_us": computing_us,
        "free_us": free_us,
        "preparing_us": preparing_us,
        "non_compute_gap_us": non_compute_gap_us,
        "other_gap_us": other_gap_us,
    }


def map_op_to_component(op_type: str) -> str:
    op = op_type.lower()
    if "hadamard" in op or "quarot" in op or "fht" in op:
        return "hadamard"
    if "attention" in op or "rope" in op:
        return "attention"
    if "swiglu" in op or "gelu" in op or "silu" in op:
        return "ffn"
    if "rmsnorm" in op or "layernorm" in op:
        return "norm"
    if "cache" in op:
        return "kv_cache"
    if "quantbatchmatmul" in op:
        return "quant_gemm"
    if "matmul" in op:
        return "gemm"
    if "quant" in op or "dequant" in op:
        return "quantize"
    if "argmax" in op or "topk" in op or "multinomial" in op:
        return "sampling"
    return "other"


def map_op_to_ffn_component(op_type: str) -> str:
    op = op_type.lower()
    if "hadamard" in op or "quarot" in op or "fht" in op:
        return "ffn_hadamard"
    if "quantbatchmatmul" in op:
        return "ffn_quant_gemm"
    if "dynamicquant" in op:
        return "ffn_dynamic_quant"
    if "dequant" in op:
        return "ffn_dequant"
    if "swiglu" in op or "gelu" in op or "silu" in op:
        return "ffn_activation"
    if op in {"viewcopy", "stridedslice", "slice", "transpose", "reshape", "concat", "splitvd"}:
        return "ffn_layout"
    if op in {"add", "sub", "mul", "div", "pow", "cast"}:
        return "ffn_elementwise"
    if "rmsnorm" in op or "layernorm" in op:
        return "ffn_norm"
    if "matmul" in op:
        return "ffn_other_gemm"
    return "non_ffn"


def summarize_components(op_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    total_us = sum(row["total_time_us"] for row in op_rows)
    bucket: dict[str, dict[str, Any]] = {}
    for row in op_rows:
        component = map_op_to_component(row["op_type"])
        entry = bucket.setdefault(component, {"total_time_us": 0.0, "ops": []})
        entry["total_time_us"] += row["total_time_us"]
        entry["ops"].append(row["op_type"])

    for component, entry in bucket.items():
        entry["ops"] = sorted(set(entry["ops"]))
        entry["ratio_pct"] = (entry["total_time_us"] / total_us * 100.0) if total_us > 0.0 else 0.0
    return dict(sorted(bucket.items(), key=lambda item: item[1]["total_time_us"], reverse=True))


def summarize_ffn_components(op_rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket: dict[str, dict[str, Any]] = {}
    ffn_total_us = 0.0
    for row in op_rows:
        component = map_op_to_ffn_component(row["op_type"])
        if component == "non_ffn":
            continue
        ffn_total_us += row["total_time_us"]
        entry = bucket.setdefault(component, {"total_time_us": 0.0, "ops": []})
        entry["total_time_us"] += row["total_time_us"]
        entry["ops"].append(row["op_type"])

    for component, entry in bucket.items():
        entry["ops"] = sorted(set(entry["ops"]))
        entry["ratio_pct_within_ffn"] = (entry["total_time_us"] / ffn_total_us * 100.0) if ffn_total_us > 0.0 else 0.0

    ordered = dict(sorted(bucket.items(), key=lambda item: item[1]["total_time_us"], reverse=True))
    return {
        "total_time_us": ffn_total_us,
        "components": ordered,
        "source": "heuristic_op_name",
    }


def map_op_to_gemm_fht_quant_component(op_type: str) -> str:
    op = op_type.lower()
    if "quantbatchmatmul" in op or "matmul" in op:
        return "gemm"
    if "hadamard" in op or "quarot" in op or "fht" in op:
        return "fht"
    if "quant" in op or "dequant" in op:
        return "quant_dequant"
    return "other"


def summarize_gemm_fht_quant_total(op_rows: list[dict[str, Any]]) -> dict[str, Any]:
    components = {
        "gemm": 0.0,
        "fht": 0.0,
        "quant_dequant": 0.0,
    }
    for row in op_rows:
        component = map_op_to_gemm_fht_quant_component(row["op_type"])
        if component in components:
            components[component] += row["total_time_us"]
    return {
        "total_time_us": sum(components.values()),
        "components": components,
    }


def summarize_fragmentation_hints(op_rows: list[dict[str, Any]]) -> dict[str, Any]:
    hint_patterns = {
        "quant_gemm": ("quantbatchmatmul",),
        "dynamic_quant": ("dynamicquant",),
        "hadamard": ("hadamard", "quarot", "fht"),
        "tensor_move": ("tensormove",),
        "cast": ("cast",),
        "slice": ("slice",),
    }
    total_count = sum(row["count"] for row in op_rows)
    hints: dict[str, dict[str, float | int]] = {}
    for hint_name, patterns in hint_patterns.items():
        total_time_us = 0.0
        count = 0
        for row in op_rows:
            op = row["op_type"].lower()
            if any(pattern in op for pattern in patterns):
                total_time_us += row["total_time_us"]
                count += row["count"]
        hints[hint_name] = {
            "count": count,
            "total_time_us": total_time_us,
        }
    return {
        "total_op_invocations": total_count,
        "selected_op_groups": dict(sorted(hints.items(), key=lambda item: item[1]["total_time_us"], reverse=True)),
    }


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.3f} {unit}"


def summarize_kv_cache_usage(kv_caches: Any) -> dict[str, Any]:
    records = iter_kv_cache_records(kv_caches)
    tensor_role_names = {
        2: ["key_cache", "value_cache"],
        3: ["key_cache", "value_cache", "aux_cache"],
        4: ["key_cache", "value_cache", "k_scale", "v_scale"],
    }
    total_bytes = 0
    total_tensor_count = 0
    per_tensor_role_bytes: dict[str, int] = {}
    per_record_bytes: list[int] = []
    first_record_tensors: list[dict[str, Any]] = []

    for record_idx, record in enumerate(records):
        record_total = 0
        role_names = tensor_role_names.get(len(record), [])
        for tensor_idx, tensor in enumerate(record):
            role_name = role_names[tensor_idx] if tensor_idx < len(role_names) else f"tensor_{tensor_idx}"
            tensor_bytes = int(tensor.numel() * tensor.element_size())
            total_bytes += tensor_bytes
            total_tensor_count += 1
            record_total += tensor_bytes
            per_tensor_role_bytes[role_name] = per_tensor_role_bytes.get(role_name, 0) + tensor_bytes
            if record_idx == 0:
                first_record_tensors.append(
                    {
                        "role": role_name,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype).replace("torch.", ""),
                        "bytes": tensor_bytes,
                        "bytes_human": _format_bytes(tensor_bytes),
                    }
                )
        per_record_bytes.append(record_total)

    num_records = len(records)
    bytes_per_record = per_record_bytes[0] if per_record_bytes else 0
    all_records_uniform = bool(per_record_bytes) and all(
        record_bytes == bytes_per_record for record_bytes in per_record_bytes
    )
    token_capacity = 0
    block_size = 0
    num_blocks = 0
    if records and records[0]:
        first_tensor = records[0][0]
        if first_tensor.ndim >= 2:
            num_blocks = int(first_tensor.shape[0])
            block_size = int(first_tensor.shape[1])
            token_capacity = num_blocks * block_size

    bytes_per_token_total = (total_bytes / token_capacity) if token_capacity > 0 else 0.0
    bytes_per_token_per_record = (bytes_per_record / token_capacity) if token_capacity > 0 else 0.0
    return {
        "num_records": num_records,
        "total_tensor_count": total_tensor_count,
        "total_bytes": total_bytes,
        "total_bytes_human": _format_bytes(total_bytes),
        "bytes_per_record": bytes_per_record,
        "bytes_per_record_human": _format_bytes(bytes_per_record),
        "all_records_uniform": all_records_uniform,
        "token_capacity": token_capacity,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "bytes_per_token_total": bytes_per_token_total,
        "bytes_per_token_per_record": bytes_per_token_per_record,
        "per_tensor_role_bytes": {
            key: {
                "bytes": value,
                "bytes_human": _format_bytes(value),
            }
            for key, value in sorted(per_tensor_role_bytes.items())
        },
        "first_record_tensors": first_record_tensors,
    }


def collect_kv_cache_usage_on_worker(worker) -> dict[str, Any]:
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        return {
            "error": "worker has no model_runner",
        }
    kv_caches = getattr(model_runner, "kv_caches", None)
    if kv_caches is None:
        return {
            "error": "model_runner has no kv_caches",
        }
    return summarize_kv_cache_usage(kv_caches)


def find_latest_ascend_output(profile_dir: Path) -> Path:
    candidates = list(profile_dir.glob("**/ASCEND_PROFILER_OUTPUT/op_statistic.csv"))
    if not candidates:
        raise FileNotFoundError(f"No op_statistic.csv found under {profile_dir}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.parent


def build_prompts(model_path: str, args: argparse.Namespace) -> list[Any]:
    if args.seq_len <= 0:
        return [args.prompt] * args.batch_size

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    seed_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not seed_ids:
        seed_ids = tokenizer.encode("Hello", add_special_tokens=False)
    if not seed_ids:
        fallback_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
        seed_ids = [fallback_id]

    repeats = (args.seq_len + len(seed_ids) - 1) // len(seed_ids)
    prompt_token_ids = (seed_ids * repeats)[: args.seq_len]
    return [{"prompt_token_ids": prompt_token_ids} for _ in range(args.batch_size)]


def run_model_profile(name: str, model_path: str, args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    if args.disable_kv_write:
        os.environ["VLLM_ASCEND_QUAROT_PROFILE_DISABLE_KV_WRITE"] = "1"
    else:
        os.environ.pop("VLLM_ASCEND_QUAROT_PROFILE_DISABLE_KV_WRITE", None)
    # This utility uses worker-side callable RPC to inspect live KV-cache tensors.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    model_profile_dir = output_root / name
    if model_profile_dir.exists():
        shutil.rmtree(model_profile_dir)
    model_profile_dir.mkdir(parents=True, exist_ok=True)

    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": str(model_profile_dir),
        "torch_profiler_with_stack": False,
    }
    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "profiler_config": profiler_config,
    }
    if args.enable_chunked_prefill is not None:
        llm_kwargs["enable_chunked_prefill"] = args.enable_chunked_prefill
    llm = LLM(
        **llm_kwargs,
    )
    kv_cache_usage = llm.collective_rpc(collect_kv_cache_usage_on_worker)[0]

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    prompts = build_prompts(model_path=model_path, args=args)

    for _ in range(args.warmup_iters):
        _ = llm.generate(prompts, sampling_params)

    latencies_ms: list[float] = []
    generated_text = ""
    for _ in range(args.profile_iters):
        llm.start_profile()
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end = time.perf_counter()
        llm.stop_profile()

        latencies_ms.append((end - start) * 1000.0)
        generated_text = outputs[0].outputs[0].text

    ascend_output_dir = find_latest_ascend_output(model_profile_dir)
    op_rows = parse_op_statistic_csv(ascend_output_dir / "op_statistic.csv")
    step_trace = parse_step_trace_csv(ascend_output_dir / "step_trace_time.csv")
    top_ops = sorted(op_rows, key=lambda row: row["total_time_us"], reverse=True)[: args.topk_ops]
    component_summary = summarize_components(op_rows)
    heuristic_ffn_breakdown = summarize_ffn_components(op_rows)
    gemm_fht_quant_total = summarize_gemm_fht_quant_total(op_rows)
    step_gap_breakdown = summarize_step_gaps(step_trace)
    fragmentation_hints = summarize_fragmentation_hints(op_rows)

    return {
        "name": name,
        "model_path": model_path,
        "generated_text": generated_text,
        "latency_ms": {
            "iters": latencies_ms,
            "avg": sum(latencies_ms) / len(latencies_ms),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
        },
        "step_trace_us": step_trace,
        "step_gap_breakdown": step_gap_breakdown,
        "kv_cache_usage": kv_cache_usage,
        "top_ops": top_ops,
        "component_breakdown": component_summary,
        "fragmentation_hints": fragmentation_hints,
        "gemm_fht_quant_total": gemm_fht_quant_total,
        "ffn_breakdown": heuristic_ffn_breakdown,
        "profile_output_dir": str(ascend_output_dir),
    }


def print_report(results: dict[str, Any]) -> None:
    config = results["config"]
    print("=== Quantized Model Profiling Summary ===")
    print(f"prompt={config['prompt']!r}, max_tokens={config['max_tokens']}")
    print(
        f"dtype={config['dtype']}, batch_size={config['batch_size']}, "
        f"seq_len={config['seq_len']}, enforce_eager={config['enforce_eager']}, "
        f"enable_chunked_prefill={config.get('enable_chunked_prefill')}, "
        f"disable_kv_write={config['disable_kv_write']}"
    )
    print()
    for model in results["models"]:
        print(f"[{model['name']}] model={model['model_path']}")
        print(
            f"  e2e_latency_ms(avg/min/max)="
            f"{model['latency_ms']['avg']:.3f}/{model['latency_ms']['min']:.3f}/{model['latency_ms']['max']:.3f}"
        )
        kv_cache_usage = model["kv_cache_usage"]
        print(
            "  kv_cache_usage="
            f"{kv_cache_usage['total_bytes_human']} "
            f"(bytes={kv_cache_usage['total_bytes']}, "
            f"records={kv_cache_usage['num_records']}, "
            f"blocks={kv_cache_usage['num_blocks']}, "
            f"block_size={kv_cache_usage['block_size']}, "
            f"token_capacity={kv_cache_usage['token_capacity']}, "
            f"bytes_per_token_total={kv_cache_usage['bytes_per_token_total']:.3f}, "
            f"bytes_per_token_per_record={kv_cache_usage['bytes_per_token_per_record']:.3f})"
        )
        if kv_cache_usage["per_tensor_role_bytes"]:
            print("  kv_cache_tensors:")
            for role, detail in kv_cache_usage["per_tensor_role_bytes"].items():
                print(f"    - {role}: {detail['bytes_human']} ({detail['bytes']} bytes)")
        gemm_fht_quant_total = model["gemm_fht_quant_total"]
        print(
            "  gemm_fht_quant_total_us="
            f"{gemm_fht_quant_total['total_time_us']:.3f} "
            f"(gemm={gemm_fht_quant_total['components']['gemm']:.3f}, "
            f"fht={gemm_fht_quant_total['components']['fht']:.3f}, "
            f"quant_dequant={gemm_fht_quant_total['components']['quant_dequant']:.3f})"
        )
        stage_us = model["step_trace_us"].get("Stage", 0.0)
        computing_us = model["step_trace_us"].get("Computing", 0.0)
        free_us = model["step_trace_us"].get("Free", 0.0)
        preparing_us = model["step_trace_us"].get("Preparing", 0.0)
        print(
            "  step_trace_us(stage/computing/free/preparing)="
            f"{stage_us:.3f}/{computing_us:.3f}/{free_us:.3f}/{preparing_us:.3f}"
        )
        gap = model.get("step_gap_breakdown") or summarize_step_gaps(model["step_trace_us"])
        print(
            "  step_gap_breakdown_us("
            "non_compute/free/preparing/other)="
            f"{gap['non_compute_gap_us']:.3f}/{gap['free_us']:.3f}/"
            f"{gap['preparing_us']:.3f}/{gap['other_gap_us']:.3f}"
        )
        fragmentation_hints = model.get("fragmentation_hints")
        if fragmentation_hints is not None:
            print("  fragmentation_hints:")
            print(f"    - total_op_invocations: {fragmentation_hints['total_op_invocations']}")
            for hint_name, detail in fragmentation_hints["selected_op_groups"].items():
                print(f"    - {hint_name}: count={detail['count']}, total_time_us={detail['total_time_us']:.3f}")
        print("  top_components:")
        for component, detail in list(model["component_breakdown"].items())[:8]:
            print(f"    - {component}: {detail['total_time_us']:.3f} us ({detail['ratio_pct']:.2f}%)")
        ffn_total_us = model["ffn_breakdown"]["total_time_us"]
        print(f"  ffn_total_us={ffn_total_us:.3f} (source={model['ffn_breakdown'].get('source', 'unknown')})")
        print("  ffn_components:")
        for component, detail in list(model["ffn_breakdown"]["components"].items())[:8]:
            print(f"    - {component}: {detail['total_time_us']:.3f} us ({detail['ratio_pct_within_ffn']:.2f}% of ffn)")
        print("  top_ops:")
        for row in model["top_ops"]:
            print(f"    - {row['op_type']}: {row['total_time_us']:.3f} us ({row['ratio_pct']:.2f}%)")
        print(f"  profile_output_dir={model['profile_output_dir']}")
        print()


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    models = []
    for name, model_path in parse_model_specs(args.model):
        models.append(run_model_profile(name=name, model_path=model_path, args=args, output_root=output_root))

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "prompt": args.prompt,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "max_tokens": args.max_tokens,
            "warmup_iters": args.warmup_iters,
            "profile_iters": args.profile_iters,
            "enforce_eager": args.enforce_eager,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "disable_kv_write": args.disable_kv_write,
        },
        "models": models,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Summary JSON written to: {summary_path}")
    return results


def main() -> None:
    args = parse_args()
    results = build_summary(args)
    print_report(results)


if __name__ == "__main__":
    main()
