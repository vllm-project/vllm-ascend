#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Prepare deterministic real-text traces for the paper-style NPU benchmark."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tokenizers import Tokenizer


def _write(path: Path, records: list[dict], output_tokens: int) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(
                    {
                        "prompt": record["prompt"],
                        "output_tokens": output_tokens,
                        "request_id": record["request_id"],
                        "target_prompt_tokens": record["target_prompt_tokens"],
                        "prompt_tokens": record["prompt_tokens"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _fit_prompt_budget(
    record: dict,
    tokenizer: Tokenizer,
    max_prompt_tokens: int,
) -> dict:
    """Return a copy whose prompt leaves room for the requested output."""
    fitted = dict(record)
    prompt_ids = tokenizer.encode(str(record["prompt"])).ids
    if len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[:max_prompt_tokens]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)

        # Decoding and re-encoding is normally lossless for these LongBench
        # prompts.  Keep trimming if a tokenizer normalization rule expands
        # the round trip so the server-side length limit remains guaranteed.
        encoded = tokenizer.encode(prompt).ids
        while len(encoded) > max_prompt_tokens:
            prompt_ids = prompt_ids[:-1]
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            encoded = tokenizer.encode(prompt).ids
        fitted["prompt"] = prompt
        fitted["truncated_for_serving"] = True
        fitted["prompt_tokens"] = len(encoded)
    else:
        fitted["prompt_tokens"] = len(prompt_ids)
        fitted["truncated_for_serving"] = False
    return fitted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--per-bucket", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=32)
    args = parser.parse_args()

    max_prompt_tokens = args.max_model_len - args.output_tokens
    if max_prompt_tokens <= 0:
        raise ValueError("output tokens must be smaller than max model length")
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))

    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    by_bucket: dict[int, list[dict]] = {}
    for record in records:
        by_bucket.setdefault(int(record["target_prompt_tokens"]), []).append(record)

    rng = random.Random(args.seed)
    selected: dict[int, list[dict]] = {}
    for bucket in (512, 2048, 8192, 32768):
        candidates = list(by_bucket[bucket])
        rng.shuffle(candidates)
        selected[bucket] = [
            _fit_prompt_budget(record, tokenizer, max_prompt_tokens) for record in candidates[: args.per_bucket]
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for bucket, bucket_records in selected.items():
        _write(
            args.output_dir / f"longbench-{bucket}.jsonl",
            bucket_records,
            args.output_tokens,
        )

    mixed: list[dict] = []
    for index in range(args.per_bucket):
        for bucket in (512, 2048, 8192, 32768):
            mixed.append(selected[bucket][index])
    _write(args.output_dir / "longbench-mixed.jsonl", mixed, args.output_tokens)

    manifest = {
        "source": str(args.input),
        "seed": args.seed,
        "per_bucket": args.per_bucket,
        "output_tokens": args.output_tokens,
        "max_model_len": args.max_model_len,
        "max_prompt_tokens": max_prompt_tokens,
        "workloads": {
            "throughput_8k": {"path": "longbench-8192.jsonl", "requests": args.per_bucket},
            "rate_sweep_2k": {"path": "longbench-2048.jsonl", "requests": args.per_bucket},
            "mixed_memory": {
                "path": "longbench-mixed.jsonl",
                "requests": args.per_bucket * 4,
            },
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
