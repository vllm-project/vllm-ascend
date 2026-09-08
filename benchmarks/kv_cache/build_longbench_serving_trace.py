#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Build a deterministic, real-text serving trace from LongBench datasets.

The resulting JSONL is intended for serving-performance experiments, not for
reporting LongBench accuracy. Records that need context truncation are marked
with ``accuracy_valid=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

import requests
from tokenizers import Tokenizer

V2_ROWS_URL = "https://datasets-server.huggingface.co/rows"
V1_SYNTHETIC_TASKS = {"passage_count", "passage_retrieval_en"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--longbench-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--v2-cache",
        type=Path,
        help="Optional raw LongBench-v2 row cache (defaults beside LongBench raw data)",
    )
    parser.add_argument("--per-bucket", type=int, default=50)
    parser.add_argument("--buckets", type=int, nargs="+", default=[512, 2048, 8192, 32768])
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    return parser.parse_args()


def encode_len(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text).ids)


def v1_parts(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    question = str(row.get("input", "")).strip()
    suffix = f"\n\nQuestion:\n{question}\n\nAnswer:"
    return str(row.get("context", "")), suffix, list(row.get("answers", []))


def v2_parts(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    question = str(row.get("question", "")).strip()
    choices = "\n".join(f"{letter}. {row.get(f'choice_{letter}', '')}" for letter in "ABCD")
    suffix = f"\n\nQuestion:\n{question}\n\nChoices:\n{choices}\n\nAnswer:"
    answer = str(row.get("answer", ""))
    return str(row.get("context", "")), suffix, [answer]


PREFIX = "Use the following context to answer the question. Keep the answer concise.\n\nContext:\n"


def load_v1(longbench_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # LongBench-E is deliberately balanced across 0-4K, 4-8K and 8K+.
    files = sorted(longbench_dir.glob("*_e.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *_e.jsonl files found in {longbench_dir}")
    for path in files:
        task = path.stem.removesuffix("_e")
        if task in V1_SYNTHETIC_TASKS:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                context, suffix, answers = v1_parts(row)
                records.append(
                    {
                        "source_dataset": "zai-org/LongBench",
                        "source_subset": f"{task}_e",
                        "source_id": str(row.get("_id", f"{task}:{line_number}")),
                        "task": task,
                        "context": context,
                        "suffix": suffix,
                        "question": str(row.get("input", "")),
                        "reference_answers": answers,
                    }
                )
    return records


def load_v2_cache(cache_path: Path, minimum_rows: int = 200) -> list[dict[str, Any]]:
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    rows: list[dict[str, Any]] = []
    for offset in range(0, 503, 100):
        response = requests.get(
            V2_ROWS_URL,
            params={
                "dataset": "zai-org/LongBench-v2",
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": min(100, 503 - offset),
            },
            timeout=180,
        )
        response.raise_for_status()
        rows.extend(item["row"] for item in response.json()["rows"])
        if len(rows) >= minimum_rows:
            break

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def normalize_v2(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        context, suffix, answers = v2_parts(row)
        records.append(
            {
                "source_dataset": "zai-org/LongBench-v2",
                "source_subset": "default",
                "source_id": str(row.get("_id", index)),
                "task": str(row.get("sub_domain", row.get("domain", "unknown"))),
                "context": context,
                "suffix": suffix,
                "question": str(row.get("question", "")),
                "reference_answers": answers,
            }
        )
    return records


def fit_prompt(tokenizer: Tokenizer, context: str, suffix: str, target: int) -> tuple[str, int, bool]:
    original = PREFIX + context + suffix
    original_len = encode_len(tokenizer, original)
    if original_len <= target:
        return original, original_len, False

    context_ids = tokenizer.encode(context).ids
    low, high = 0, len(context_ids)
    best_prompt = PREFIX + suffix
    best_len = encode_len(tokenizer, best_prompt)
    while low <= high:
        middle = (low + high) // 2
        cut_context = tokenizer.decode(context_ids[:middle], skip_special_tokens=False)
        prompt = PREFIX + cut_context + suffix
        prompt_len = encode_len(tokenizer, prompt)
        if prompt_len <= target:
            best_prompt, best_len = prompt, prompt_len
            low = middle + 1
        else:
            high = middle - 1
    return best_prompt, best_len, True


def source_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        record["source_dataset"],
        record["source_subset"],
        record["source_id"],
    )


def build_trace(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    randomizer = random.Random(args.seed)

    v1_records = load_v1(args.longbench_dir)
    v2_cache = args.v2_cache or args.longbench_dir.parent / "longbench_v2_rows.jsonl"
    v2_records = normalize_v2(load_v2_cache(v2_cache))

    # Computing lengths once makes selection reproducible and avoids repeatedly
    # tokenizing large contexts during bucket construction.
    for record in v1_records + v2_records:
        record["minimum_prompt_tokens"] = encode_len(tokenizer, PREFIX + record["suffix"])
        record["original_prompt_tokens"] = encode_len(tokenizer, PREFIX + record["context"] + record["suffix"])

    selected: list[dict[str, Any]] = []
    used: set[tuple[str, str, str]] = set()
    bucket_stats: dict[str, Any] = {}
    for target in sorted(args.buckets, reverse=True):
        pool = v2_records if target >= 32768 else v1_records
        eligible = [
            record
            for record in pool
            if record["original_prompt_tokens"] >= target
            and record["minimum_prompt_tokens"] <= target
            and source_key(record) not in used
        ]
        randomizer.shuffle(eligible)
        eligible.sort(key=lambda record: record["original_prompt_tokens"] - target)
        chosen = eligible[: args.per_bucket]
        if len(chosen) != args.per_bucket:
            raise RuntimeError(
                f"Bucket {target} has only {len(chosen)} eligible unique samples; need {args.per_bucket}"
            )

        actual_lengths: list[int] = []
        for bucket_index, record in enumerate(chosen):
            prompt, prompt_tokens, truncated = fit_prompt(tokenizer, record["context"], record["suffix"], target)
            used.add(source_key(record))
            actual_lengths.append(prompt_tokens)
            selected.append(
                {
                    "request_id": f"longbench-{target}-{bucket_index:04d}",
                    "source_dataset": record["source_dataset"],
                    "source_subset": record["source_subset"],
                    "source_id": record["source_id"],
                    "task": record["task"],
                    "target_prompt_tokens": target,
                    "prompt_tokens": prompt_tokens,
                    "max_output_tokens": args.max_output_tokens,
                    "prompt": prompt,
                    "question": record["question"],
                    "reference_answers": record["reference_answers"],
                    "original_prompt_tokens": record["original_prompt_tokens"],
                    "truncated": truncated,
                    "accuracy_valid": not truncated,
                    "usage": "serving_performance_only" if truncated else "accuracy_and_performance",
                }
            )

        bucket_stats[str(target)] = {
            "records": len(chosen),
            "min_prompt_tokens": min(actual_lengths),
            "median_prompt_tokens": statistics.median(actual_lengths),
            "max_prompt_tokens": max(actual_lengths),
        }

    selected.sort(key=lambda record: (record["target_prompt_tokens"], record["request_id"]))
    manifest = {
        "description": "Real-text serving trace derived from LongBench/LongBench-v2",
        "not_synthetic": True,
        "seed": args.seed,
        "tokenizer": str(args.tokenizer_json),
        "sources": [
            "https://huggingface.co/datasets/zai-org/LongBench",
            "https://huggingface.co/datasets/zai-org/LongBench-v2",
        ],
        "accuracy_policy": (
            "Records truncated to a target serving length have accuracy_valid=false "
            "and must not be used to report LongBench benchmark accuracy."
        ),
        "excluded_synthetic_tasks": sorted(V1_SYNTHETIC_TASKS),
        "buckets": bucket_stats,
        "records": len(selected),
    }
    return selected, manifest


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records, manifest = build_trace(args)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest["jsonl_sha256"] = digest
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
