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

"""Benchmark full Qwen3-8B vs QuaRot W4A4 RTN on WikiText-2 perplexity.

This benchmark uses teacher-forced prompt scoring on exact token IDs to stay
close to the upstream QuaRot WikiText-2 methodology while still running
through the vLLM/vllm-ascend stack.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_FULL_MODEL = "/data/weights/qwen3-8B"
DEFAULT_QUAROT_MODEL = "/data/weights/Qwen3-8B-QuaRot-W4A4-RTN"
DEFAULT_DATASET = "/workspace/wikitext-2-raw/wiki.test.raw"


@dataclass
class EvalResult:
    name: str
    model_path: str
    perplexity: float
    neg_log_likelihood: float
    scored_tokens: int
    chunks: int
    model_load_sec: float
    eval_sec: float
    latency_avg_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    throughput_tok_s: float
    chunk_token_len: int
    stride: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-model", default=DEFAULT_FULL_MODEL, help="Path to full Qwen3-8B model.")
    parser.add_argument("--quarot-model", default=DEFAULT_QUAROT_MODEL, help="Path to QuaRot W4A4 RTN model.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET, help="Path to WikiText-2 raw test file.")
    parser.add_argument("--dtype", default="float16", help="Runtime dtype (default: float16).")
    parser.add_argument(
        "--chunk-token-len",
        type=int,
        default=4096,
        help="Token length per chunk (default: 4096, validated long-context PPL setting).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4096,
        help="Stride for sliding window chunks (default: 4096). Set < chunk-token-len to overlap chunks.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Optional cap on number of chunks per model (default: 100; 0 means all).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4097,
        help="max_model_len for vLLM engine (default: 4097 for 4096-token prompt plus 1 output token).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help=(
            "gpu_memory_utilization passed to vLLM engine startup "
            "(default: 0.80 to leave prompt-logprob workspace headroom)."
        ),
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Use eager mode (disables graph/compile path).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional file path to write machine-readable JSON report.",
    )
    return parser.parse_args()


def read_wikitext(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return p.read_text(encoding="utf-8")


def build_chunks(token_ids: list[int], chunk_len: int, stride: int, max_samples: int) -> list[list[int]]:
    if chunk_len < 2:
        raise ValueError(f"--chunk-token-len must be >= 2, got {chunk_len}")
    if stride < 1:
        raise ValueError(f"--stride must be >= 1, got {stride}")

    chunks: list[list[int]] = []
    n = len(token_ids)
    for start in range(0, n - 1, stride):
        end = min(start + chunk_len, n)
        chunk = token_ids[start:end]
        # Drop tail fragments so every sample has the same teacher-forced
        # scoring contract, matching the common WikiText-2 perplexity setup.
        if len(chunk) < chunk_len:
            continue
        chunks.append(chunk)
        if max_samples > 0 and len(chunks) >= max_samples:
            break
    return chunks


def sum_target_token_logprobs(prompt_logprobs: Any, token_ids: list[int]) -> tuple[float, int]:
    if prompt_logprobs is None:
        return 0.0, 0

    total = 0.0
    counted = 0
    # vLLM returns one prompt_logprobs entry per prompt token, with the first
    # token unscored and stored as an empty entry. Token i is scored at
    # prompt_logprobs[i].
    if len(prompt_logprobs) < len(token_ids):
        return 0.0, 0

    for pos, token_id in enumerate(token_ids[1:], start=1):
        lp_dict = prompt_logprobs[pos]
        if not lp_dict:
            continue
        token_lp = lp_dict.get(token_id)
        if token_lp is None:
            continue
        total += float(token_lp.logprob)
        counted += 1
    return total, counted


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def evaluate_model(
    name: str,
    model_path: str,
    token_chunks: list[list[int]],
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> EvalResult:
    load_start = time.perf_counter()
    llm = LLM(
        model=model_path,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    load_end = time.perf_counter()

    # vLLM's offline generate path requires max_tokens >= 1. We keep a single
    # decode token to access prompt_logprobs while evaluating teacher-forced
    # prompt tokens on exact token IDs.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    eval_start = time.perf_counter()
    nll = 0.0
    scored_tokens = 0
    lat_ms: list[float] = []

    for chunk in token_chunks:
        t0 = time.perf_counter()
        outputs = llm.generate([{"prompt_token_ids": chunk}], sampling_params)
        t1 = time.perf_counter()
        lat_ms.append((t1 - t0) * 1000.0)

        prompt_lp = outputs[0].prompt_logprobs
        sum_lp, count_lp = sum_target_token_logprobs(prompt_lp, chunk)
        nll -= sum_lp
        scored_tokens += count_lp

    eval_end = time.perf_counter()
    eval_sec = eval_end - eval_start
    ppl = math.exp(nll / scored_tokens) if scored_tokens > 0 else float("inf")
    throughput = (scored_tokens / eval_sec) if eval_sec > 0 else 0.0

    return EvalResult(
        name=name,
        model_path=model_path,
        perplexity=ppl,
        neg_log_likelihood=nll,
        scored_tokens=scored_tokens,
        chunks=len(token_chunks),
        model_load_sec=load_end - load_start,
        eval_sec=eval_sec,
        latency_avg_ms=statistics.fmean(lat_ms) if lat_ms else 0.0,
        latency_p50_ms=percentile(lat_ms, 0.50),
        latency_p95_ms=percentile(lat_ms, 0.95),
        throughput_tok_s=throughput,
        chunk_token_len=args.chunk_token_len,
        stride=args.stride,
    )


def result_to_dict(result: EvalResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "model_path": result.model_path,
        "perplexity": result.perplexity,
        "neg_log_likelihood": result.neg_log_likelihood,
        "scored_tokens": result.scored_tokens,
        "chunks": result.chunks,
        "model_load_sec": result.model_load_sec,
        "eval_sec": result.eval_sec,
        "latency_avg_ms": result.latency_avg_ms,
        "latency_p50_ms": result.latency_p50_ms,
        "latency_p95_ms": result.latency_p95_ms,
        "throughput_tok_s": result.throughput_tok_s,
        "chunk_token_len": result.chunk_token_len,
        "stride": result.stride,
    }


def print_summary(results: list[EvalResult], meta: dict[str, Any]) -> None:
    print("=== Qwen3 WikiText-2 Perplexity Benchmark ===")
    print(f"dataset_path={meta['dataset_path']}")
    print(
        f"dataset_tokens={meta['dataset_tokens']}, chunks={meta['chunks']}, "
        f"chunk_token_len={meta['chunk_token_len']}, stride={meta['stride']}"
    )
    print(f"dtype={meta['dtype']}, max_model_len={meta['max_model_len']}, enforce_eager={meta['enforce_eager']}")
    print(
        "env: "
        f"ASCEND_RT_VISIBLE_DEVICES={meta['env'].get('ASCEND_RT_VISIBLE_DEVICES', '')}, "
        f"VLLM_WORKER_MULTIPROC_METHOD={meta['env'].get('VLLM_WORKER_MULTIPROC_METHOD', '')}, "
        f"VLLM_ASCEND_QUAROT_EXEC_MODE={meta['env'].get('VLLM_ASCEND_QUAROT_EXEC_MODE', '')}"
    )
    print()
    for r in results:
        print(f"[{r.name}] {r.model_path}")
        print(f"  ppl={r.perplexity:.6f}  scored_tokens={r.scored_tokens}  nll={r.neg_log_likelihood:.3f}")
        print(f"  load_s={r.model_load_sec:.3f}  eval_s={r.eval_sec:.3f}  tok_s={r.throughput_tok_s:.3f}")
        print(f"  latency_ms(avg/p50/p95)={r.latency_avg_ms:.3f}/{r.latency_p50_ms:.3f}/{r.latency_p95_ms:.3f}")
    if len(results) == 2:
        a, b = results
        print()
        print(
            f"Delta ({b.name} - {a.name}): "
            f"ppl={b.perplexity - a.perplexity:+.6f}, "
            f"tok_s={b.throughput_tok_s - a.throughput_tok_s:+.3f}, "
            f"lat_avg_ms={b.latency_avg_ms - a.latency_avg_ms:+.3f}"
        )


def main() -> None:
    args = parse_args()

    text = read_wikitext(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.full_model, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    chunks = build_chunks(token_ids, args.chunk_token_len, args.stride, args.max_samples)
    if not chunks:
        raise RuntimeError("No chunks generated from dataset. Check --chunk-token-len/--stride and dataset content.")

    base_meta = {
        "dataset_path": args.dataset_path,
        "dataset_tokens": len(token_ids),
        "chunks": len(chunks),
        "chunk_token_len": args.chunk_token_len,
        "stride": args.stride,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "env": {
            "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
            "VLLM_WORKER_MULTIPROC_METHOD": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", ""),
            "VLLM_ASCEND_QUAROT_EXEC_MODE": os.environ.get("VLLM_ASCEND_QUAROT_EXEC_MODE", ""),
        },
    }

    results = [
        evaluate_model("full", args.full_model, chunks, tokenizer, args),
        evaluate_model("quarot_w4a4_rtn", args.quarot_model, chunks, tokenizer, args),
    ]

    print_summary(results, base_meta)

    payload = {
        "meta": base_meta,
        "results": [result_to_dict(r) for r in results],
    }
    print()
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {args.output_json}")


if __name__ == "__main__":
    main()
