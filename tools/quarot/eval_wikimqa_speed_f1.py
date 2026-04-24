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
"""Evaluate WikiMQA/MuSiQue F1 and generation speed for dense/QuaRot models.

The default dataset is CacheBlend's small WikiMQA/MuSiQue input file:
https://github.com/YaoJiayi/CacheBlend/blob/main/inputs/wikimqa_s.json

The prompt and F1 normalization intentionally mirror CacheBlend's example utils.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import ssl
import statistics
import string
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_DATASET_URL = "https://raw.githubusercontent.com/YaoJiayi/CacheBlend/main/inputs/wikimqa_s.json"
DEFAULT_SYSTEM_PROMPT = (
    "Answer the question based on the given passages.\n"
    "Answer the question within 5 words. Do NOT repeat the question or output any other words."
)
DEFAULT_QUERY_PROMPT = "\n\nQuestion: "
DEFAULT_MODELS = [
    "dense=/data/weights/qwen3-8B",
    "quarot=/workspace/Qwen3-8B-QuaRot-W4A4-q_random-debug-perchannel",
]


def parse_optional_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


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


def parse_model_int_overrides(specs: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid override spec {spec!r}; expected name=value")
        name, value = spec.split("=", maxsplit=1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Invalid override spec {spec!r}; name/value must not be empty")
        parsed[name] = int(value)
    return parsed


def default_dataset_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "wikimqa_s.json"


def download_dataset(dataset_url: str, dataset_path: Path, *, force: bool = False) -> None:
    if dataset_path.exists() and not force:
        return
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(dataset_url, timeout=60) as response:
            tmp_path.write_bytes(response.read())
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if not isinstance(reason, ssl.SSLCertVerificationError):
            raise
        print(
            "warning: HTTPS certificate verification failed; retrying dataset download without verification",
            file=sys.stderr,
        )
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(dataset_url, timeout=60, context=context) as response:
            tmp_path.write_bytes(response.read())
    # Validate before publishing the file path.
    with tmp_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Downloaded dataset must be a JSON list, got {type(data).__name__}")
    tmp_path.replace(dataset_path)


def load_dataset(dataset_path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON list, got {type(data).__name__}")
    if limit is not None:
        data = data[:limit]
    return data


def normalize_question(question: str) -> str:
    if not question.endswith("?"):
        question = question + "?"
    return question[0].lower() + question[1:] if question else question


def parse_generation(text: str) -> str:
    text = text.lstrip("\n").split("\n")[0].strip()
    words = text.split()
    if text.startswith(("Yes", "yes")):
        return "Yes"
    if words and words[0].startswith(("No", "no")):
        return "No"
    return text


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def _encoded_answer_tokens(tokenizer: Any, text: str, *, drop_first_token: bool) -> list[int]:
    tokens = tokenizer.encode(normalize_answer(text))
    if drop_first_token and tokens:
        return tokens[1:]
    return tokens


def compute_f1(prediction: str, gold: str, tokenizer: Any, *, drop_first_token: bool) -> float:
    prediction = parse_generation(prediction)
    gold_toks = _encoded_answer_tokens(tokenizer, gold, drop_first_token=drop_first_token)
    pred_toks = _encoded_answer_tokens(tokenizer, prediction, drop_first_token=drop_first_token)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def flatten_answers(answers: Any) -> list[str]:
    if isinstance(answers, str):
        return [answers]
    flattened: list[str] = []
    if isinstance(answers, list):
        for item in answers:
            flattened.extend(flatten_answers(item))
    return [answer for answer in flattened if answer]


def build_user_prompt(example: dict[str, Any], query_prompt: str) -> str:
    q = normalize_question(example["question"])
    doc_prompts = []
    for ctx in example["ctxs"]:
        title = ctx.get("title", "")
        text = ctx.get("text", "")
        doc_prompts.append(f"{title}\n\n{text}\n\n")
    return "".join(doc_prompts) + f"{query_prompt}{q}"


def apply_chat_template(
    tokenizer: Any,
    user_prompt: str,
    system_prompt: str,
    *,
    enable_thinking: bool,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def batch_iter(items: list[Any], batch_size: int):
    for begin in range(0, len(items), batch_size):
        yield begin, items[begin : begin + batch_size]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return {
        "avg": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "p50": statistics.median(values),
        "p90": ordered[p90_index],
    }


def cleanup_llm(llm: Any) -> None:
    shutdown = getattr(llm, "shutdown", None)
    if callable(shutdown):
        shutdown()
    del llm
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
    examples: list[dict[str, Any]],
    user_prompts: list[str],
    args: argparse.Namespace,
    output_root: Path,
    tensor_parallel_size: int,
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_output_dir = output_root / name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    profiler_config = None
    if args.torch_profile:
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": str(model_output_dir / "torch_profile"),
            "torch_profiler_with_stack": False,
        }

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    prompts = [
        apply_chat_template(
            tokenizer,
            prompt,
            args.system_prompt,
            enable_thinking=args.enable_thinking,
        )
        for prompt in user_prompts
    ]
    prompt_token_counts = [len(tokenizer.encode(prompt)) for prompt in prompts]
    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": True,
    }
    if args.quantization is not None:
        llm_kwargs["quantization"] = args.quantization
    if args.enable_chunked_prefill is not None:
        llm_kwargs["enable_chunked_prefill"] = args.enable_chunked_prefill
    if profiler_config is not None:
        llm_kwargs["profiler_config"] = profiler_config

    load_begin = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_s = time.perf_counter() - load_begin

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    if args.warmup_examples > 0:
        warmup_prompts = prompts[: args.warmup_examples]
        for _, batch in batch_iter(warmup_prompts, args.batch_size):
            llm.generate(batch, sampling_params)

    records: list[dict[str, Any]] = []
    batch_latencies_ms: list[float] = []
    generated_token_count = 0
    eval_begin = time.perf_counter()
    if args.torch_profile:
        llm.start_profile()
    try:
        for begin, batch_prompts in batch_iter(prompts, args.batch_size):
            batch_begin = time.perf_counter()
            outputs = llm.generate(batch_prompts, sampling_params)
            batch_latency_ms = (time.perf_counter() - batch_begin) * 1000.0
            batch_latencies_ms.append(batch_latency_ms)
            for offset, output in enumerate(outputs):
                idx = begin + offset
                text = output.outputs[0].text
                token_ids = output.outputs[0].token_ids or []
                generated_token_count += len(token_ids)
                answers = flatten_answers(examples[idx].get("answers", []))
                if not answers:
                    answers = [""]
                f1s = [
                    compute_f1(text, answer, tokenizer, drop_first_token=args.cacheblend_drop_first_token)
                    for answer in answers
                ]
                best_f1 = max(f1s) if f1s else 0.0
                parsed = parse_generation(text)
                exact = max(normalize_answer(parsed) == normalize_answer(answer) for answer in answers)
                records.append(
                    {
                        "index": idx,
                        "question": examples[idx].get("question", ""),
                        "gold_answers": answers,
                        "prediction": text,
                        "parsed_prediction": parsed,
                        "f1": best_f1,
                        "exact_match": bool(exact),
                        "prompt_tokens": prompt_token_counts[idx],
                        "generated_tokens": len(token_ids),
                    }
                )
    finally:
        if args.torch_profile:
            llm.stop_profile()
    eval_s = time.perf_counter() - eval_begin

    f1_values = [record["f1"] for record in records]
    exact_values = [1.0 if record["exact_match"] else 0.0 for record in records]
    result = {
        "name": name,
        "model_path": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "num_examples": len(records),
        "f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "exact_match": sum(exact_values) / len(exact_values) if exact_values else 0.0,
        "load_s": load_s,
        "eval_s": eval_s,
        "examples_per_s": len(records) / eval_s if eval_s > 0 else 0.0,
        "generated_tokens_per_s": generated_token_count / eval_s if eval_s > 0 else 0.0,
        "prompt_tokens_per_s": sum(prompt_token_counts[: len(records)]) / eval_s if eval_s > 0 else 0.0,
        "prompt_tokens": summarize([float(x) for x in prompt_token_counts[: len(records)]]),
        "generated_tokens": generated_token_count,
        "batch_latency_ms": summarize(batch_latencies_ms),
        "batch_latencies_ms": batch_latencies_ms,
        "records": records,
    }
    if args.torch_profile:
        result["torch_profile_dir"] = str(model_output_dir / "torch_profile")
    (model_output_dir / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    cleanup_llm(llm)
    return result


def build_comparisons(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(models) < 2:
        return []
    baseline = models[0]
    comparisons = []
    for candidate in models[1:]:
        comparisons.append(
            {
                "baseline": baseline["name"],
                "candidate": candidate["name"],
                "f1_delta": candidate["f1"] - baseline["f1"],
                "exact_match_delta": candidate["exact_match"] - baseline["exact_match"],
                "eval_speedup": baseline["eval_s"] / candidate["eval_s"] if candidate["eval_s"] > 0 else 0.0,
                "examples_per_s_speedup": candidate["examples_per_s"] / baseline["examples_per_s"]
                if baseline["examples_per_s"] > 0
                else 0.0,
                "batch_latency_speedup": baseline["batch_latency_ms"]["avg"] / candidate["batch_latency_ms"]["avg"]
                if candidate["batch_latency_ms"]["avg"] > 0
                else 0.0,
            }
        )
    return comparisons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL)
    parser.add_argument("--dataset-path", type=Path, default=default_dataset_path())
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    parser.add_argument("--model", action="append", default=[], help="Model spec name=path. Can be repeated.")
    parser.add_argument(
        "--model-tp",
        action="append",
        default=[],
        help="Optional tensor parallel override name=int. Can be repeated per model.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/quarot_wikimqa_eval"))
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup-examples", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--quantization", default=None, help="Optional quantization override passed to every model.")
    parser.add_argument("--enable-chunked-prefill", type=parse_optional_bool, default=None)
    parser.add_argument("--use-native-kv-cache", type=parse_optional_bool, default=None)
    parser.add_argument("--torch-profile", action="store_true", help="Enable vLLM torch profiler around the eval loop.")
    parser.add_argument(
        "--cacheblend-drop-first-token",
        type=parse_optional_bool,
        default=True,
        help="Mirror CacheBlend compute_f1 by dropping tokenizer.encode(...)[0].",
    )
    parser.add_argument("--query-prompt", default=DEFAULT_QUERY_PROMPT)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--enable-thinking",
        type=parse_optional_bool,
        default=False,
        help="Pass enable_thinking to tokenizer.apply_chat_template; defaults to false for concise QA.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.use_native_kv_cache is not None:
        os.environ["VLLM_ASCEND_QUAROT_USE_NATIVE_KV_CACHE"] = "1" if args.use_native_kv_cache else "0"
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    download_dataset(args.dataset_url, args.dataset_path, force=args.force_download)
    print(f"dataset_path={args.dataset_path}")
    if args.download_only:
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    examples = load_dataset(args.dataset_path, limit=args.limit)
    user_prompts = [build_user_prompt(example, args.query_prompt) for example in examples]
    model_specs = parse_model_specs(args.model or DEFAULT_MODELS)
    model_tps = parse_model_int_overrides(args.model_tp)

    results = {
        "config": {
            "dataset_url": args.dataset_url,
            "dataset_path": str(args.dataset_path),
            "num_examples": len(examples),
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "warmup_examples": args.warmup_examples,
            "enforce_eager": args.enforce_eager,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "use_native_kv_cache": os.getenv("VLLM_ASCEND_QUAROT_USE_NATIVE_KV_CACHE"),
            "torch_profile": args.torch_profile,
            "cacheblend_drop_first_token": args.cacheblend_drop_first_token,
            "prompt_format": "chat_template",
            "enable_thinking": args.enable_thinking,
            "system_prompt": args.system_prompt,
            "query_prompt": args.query_prompt,
            "model_tps": model_tps,
        },
        "models": [],
    }

    for name, model_path in model_specs:
        print(f"[{name}] model={model_path}")
        tensor_parallel_size = model_tps.get(name, 1)
        print(f"[{name}] tensor_parallel_size={tensor_parallel_size}")
        model_result = run_model(
            name=name,
            model_path=model_path,
            examples=examples,
            user_prompts=user_prompts,
            args=args,
            output_root=args.output_root,
            tensor_parallel_size=tensor_parallel_size,
        )
        results["models"].append(model_result)
        print(
            f"[{name}] f1={model_result['f1']:.4f}, em={model_result['exact_match']:.4f}, "
            f"eval_s={model_result['eval_s']:.3f}, examples/s={model_result['examples_per_s']:.3f}, "
            f"batch_latency_ms_avg={model_result['batch_latency_ms']['avg']:.3f}"
        )

    results["comparisons"] = build_comparisons(results["models"])
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"summary_json={summary_path}")
    for comparison in results["comparisons"]:
        print(
            f"[{comparison['candidate']} vs {comparison['baseline']}] "
            f"f1_delta={comparison['f1_delta']:.4f}, "
            f"eval_speedup={comparison['eval_speedup']:.3f}, "
            f"batch_latency_speedup={comparison['batch_latency_speedup']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
