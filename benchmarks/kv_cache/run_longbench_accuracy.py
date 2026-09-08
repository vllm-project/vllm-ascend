#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run a deterministic LongBench-E HotpotQA correctness gate.

This is a focused KV-cache correctness check, not a LongBench leaderboard run.
It uses unmodified, untruncated LongBench-E records and the official HotpotQA
prompt, generation length, answer normalization, and QA F1 metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

PROMPT = """Answer the question based on the given passages. Only give me the answer and do not output any other words.

 The following are given passages.
{context}

 Answer the question based on the given passages. Only give me the answer and do not output any other words.

 Question: {input}
 Answer:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--samples-per-bucket", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Keep Qwen thinking mode enabled (disabled by default for the official 32-token answer budget).",
    )
    return parser.parse_args()


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def bucket_name(length: int) -> str:
    if length < 4_000:
        return "0-4K"
    if length < 8_000:
        return "4-8K"
    return "8K+"


def load_selection(path: Path, samples_per_bucket: int, seed: int) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    buckets: dict[str, list[dict[str, Any]]] = {"0-4K": [], "4-8K": [], "8K+": []}
    for record in records:
        buckets[bucket_name(int(record["length"]))].append(record)

    rng = random.Random(seed)
    selected = []
    for name, records_in_bucket in buckets.items():
        if len(records_in_bucket) < samples_per_bucket:
            raise ValueError(f"{name} only has {len(records_in_bucket)} records")
        records_in_bucket.sort(key=lambda record: record["_id"])
        for record in rng.sample(records_in_bucket, samples_per_bucket):
            selected.append(record)
    return selected


def is_degenerate(text: str) -> bool:
    compact = "".join(text.split())
    if not compact:
        return True
    _, count = Counter(compact).most_common(1)[0]
    return len(compact) >= 8 and count / len(compact) >= 0.8


def main() -> None:
    args = parse_args()
    selected = load_selection(args.dataset, args.samples_per_bucket, args.seed)
    model = args.model
    if model is None:
        models = request_json(f"{args.base_url}/models", None, args.request_timeout)
        model = models["data"][0]["id"]

    results = []
    started = time.perf_counter()
    for index, record in enumerate(selected):
        prompt = PROMPT.format(**record)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 32,
            "seed": args.seed,
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        }
        request_started = time.perf_counter()
        error = None
        prediction = ""
        usage = None
        try:
            response = request_json(
                f"{args.base_url}/chat/completions",
                payload,
                args.request_timeout,
            )
            prediction = response["choices"][0]["message"]["content"] or ""
            usage = response.get("usage")
        except Exception as exception:  # Preserve all failed samples in the evidence file.
            error = f"{type(exception).__name__}: {exception}"

        answers = record["answers"]
        scores = [qa_f1(prediction, answer) for answer in answers] if error is None else [0.0]
        exact_matches = [normalize_answer(prediction) == normalize_answer(answer) for answer in answers]
        result = {
            "index": index,
            "id": record["_id"],
            "bucket": bucket_name(int(record["length"])),
            "source_length": int(record["length"]),
            "question": record["input"],
            "answers": answers,
            "prediction": prediction,
            "f1": max(scores),
            "exact_match": any(exact_matches),
            "degenerate": is_degenerate(prediction),
            "latency_seconds": time.perf_counter() - request_started,
            "usage": usage,
            "error": error,
        }
        results.append(result)
        print(
            f"[{index + 1:02d}/{len(selected)}] {result['bucket']} "
            f"F1={result['f1']:.3f} EM={int(result['exact_match'])} "
            f"latency={result['latency_seconds']:.2f}s",
            flush=True,
        )

    bucket_scores = {}
    for name in ("0-4K", "4-8K", "8K+"):
        bucket_results = [result for result in results if result["bucket"] == name]
        bucket_scores[name] = {
            "samples": len(bucket_results),
            "qa_f1_percent": 100 * sum(result["f1"] for result in bucket_results) / len(bucket_results),
            "exact_match_percent": 100 * sum(result["exact_match"] for result in bucket_results) / len(bucket_results),
        }

    report = {
        "schema_version": 1,
        "dataset": "THUDM/LongBench hotpotqa_e",
        "dataset_path": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "selection_seed": args.seed,
        "samples_per_bucket": args.samples_per_bucket,
        "model": model,
        "base_url": args.base_url,
        "generation": {
            "temperature": 0,
            "max_tokens": 32,
            "enable_thinking": args.enable_thinking,
        },
        "summary": {
            "samples": len(results),
            "qa_f1_percent": 100 * sum(result["f1"] for result in results) / len(results),
            "exact_match_percent": 100 * sum(result["exact_match"] for result in results) / len(results),
            "errors": sum(result["error"] is not None for result in results),
            "degenerate_outputs": sum(result["degenerate"] for result in results),
            "wall_time_seconds": time.perf_counter() - started,
            "buckets": bucket_scores,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
