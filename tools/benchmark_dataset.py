# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import filelock

logger = logging.getLogger(__name__)

SUPPORTED_DATASET_TYPES = {"fixed", "prefix"}
DEFAULT_CACHE_ROOT = "/tmp/vllm_ascend_datasets"
SEPARATOR_TOKEN_COUNT = 3

# Performance datasets only need stable prompt text. Keeping the small seed
# corpus in source avoids introducing another network dependency in CI.
SEED_TEXTS = (
    "A shop has several boxes of pencils. Explain how to calculate the total number of pencils step by step.",
    "A train travels between two cities at a constant speed. Determine the travel time and show the calculation.",
    "A library lends books to students during the week. Work out how many books remain and explain each step.",
    "A farmer packs fruit into equal sized baskets. Calculate the number of full baskets and the remainder.",
    "A classroom collects data from an experiment. Summarize the observations and derive the final result carefully.",
    "A runner completes several laps of a track. Compute the total distance and provide a concise explanation.",
    "A warehouse receives and ships packages every day. Find the final inventory using clear intermediate steps.",
    "A recipe is scaled for a larger group. Calculate the required amount of every ingredient in a logical order.",
)


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _fit_token_ids(tokenizer: Any, text: str, target_length: int) -> list[int]:
    """Repeat or truncate encoded text to exactly ``target_length`` token IDs."""
    if target_length == 0:
        return []

    token_ids = _encode(tokenizer, text)
    if not token_ids:
        raise ValueError("Seed text produced no tokens")
    repeats = (target_length + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[:target_length]


def _decode_to_exact_length(tokenizer: Any, token_ids: list[int], target_length: int, filler_text: str) -> str:
    """Decode IDs and correct tokenizer boundary changes until the length is exact."""
    text = _decode(tokenizer, token_ids)
    for _ in range(8):
        encoded = _encode(tokenizer, text)
        if len(encoded) == target_length:
            return text
        if len(encoded) > target_length:
            text = _decode(tokenizer, encoded[:target_length])
            continue
        missing = target_length - len(encoded)
        text = f"{text} {_decode(tokenizer, _fit_token_ids(tokenizer, filler_text, missing))}"
    actual_length = len(_encode(tokenizer, text))
    raise RuntimeError(f"Unable to generate exactly {target_length} tokens; got {actual_length}")


def _build_separator_ids(tokenizer: Any, rng: random.Random, sample_index: int, length: int) -> list[int]:
    separator = f"\nRequest variant {sample_index}-{rng.getrandbits(64):016x}:\n"
    return _fit_token_ids(tokenizer, separator, length)


def _build_fixed_sample(
    tokenizer: Any,
    seed_text: str,
    input_length: int,
    rng: random.Random,
    sample_index: int,
) -> str:
    separator_length = min(SEPARATOR_TOKEN_COUNT, input_length)
    token_ids = _build_separator_ids(tokenizer, rng, sample_index, separator_length)
    token_ids += _fit_token_ids(tokenizer, seed_text, input_length - separator_length)
    return _decode_to_exact_length(tokenizer, token_ids, input_length, seed_text)


def _build_prefix_samples(
    tokenizer: Any,
    *,
    input_length: int,
    num_samples: int,
    prefix_ratio: float,
    prefix_num: int,
    seed: int,
) -> list[str]:
    prefix_length = int(input_length * prefix_ratio)
    separator_length = min(SEPARATOR_TOKEN_COUNT, input_length - prefix_length)
    suffix_length = input_length - prefix_length - separator_length
    rng = random.Random(seed)

    prefix_ids = [
        _fit_token_ids(
            tokenizer,
            f"Prefix group {index}: {SEED_TEXTS[index % len(SEED_TEXTS)]}\n",
            prefix_length,
        )
        for index in range(prefix_num)
    ]
    samples = []
    for index in range(num_samples):
        selected_prefix = prefix_ids[index % prefix_num]
        separator_ids = _build_separator_ids(tokenizer, rng, index, separator_length)
        suffix_seed = SEED_TEXTS[(index + prefix_num) % len(SEED_TEXTS)]
        suffix_ids = _fit_token_ids(tokenizer, suffix_seed, suffix_length)
        sample_ids = selected_prefix + separator_ids + suffix_ids
        filler_text = SEED_TEXTS[(index % prefix_num) % len(SEED_TEXTS)]
        samples.append(_decode_to_exact_length(tokenizer, sample_ids, input_length, filler_text))
    return samples


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset_type = str(config.get("type", "")).lower()
    if dataset_type not in SUPPORTED_DATASET_TYPES:
        raise ValueError(f"dataset_generator.type must be one of {sorted(SUPPORTED_DATASET_TYPES)}")

    normalized = {
        "type": dataset_type,
        "input_len": int(config["input_len"]),
        "num_samples": int(config["num_samples"]),
        "seed": int(config.get("seed", 1)),
        "prefix_ratio": float(config.get("prefix_ratio", 0.0)),
        "prefix_num": int(config.get("prefix_num", 1)),
        "trust_remote_code": bool(config.get("trust_remote_code", True)),
    }
    if normalized["input_len"] < 1:
        raise ValueError("dataset_generator.input_len must be >= 1")
    if normalized["num_samples"] < 1:
        raise ValueError("dataset_generator.num_samples must be >= 1")
    if normalized["prefix_num"] < 1:
        raise ValueError("dataset_generator.prefix_num must be >= 1")
    if not 0.0 <= normalized["prefix_ratio"] <= 1.0:
        raise ValueError("dataset_generator.prefix_ratio must be in [0, 1]")
    if dataset_type == "fixed" and normalized["prefix_ratio"] != 0.0:
        raise ValueError("fixed datasets do not accept a non-zero prefix_ratio")
    if dataset_type == "prefix" and normalized["prefix_ratio"] <= 0.0:
        raise ValueError("prefix datasets require prefix_ratio > 0")
    if dataset_type == "prefix" and int(normalized["input_len"] * normalized["prefix_ratio"]) == 0:
        raise ValueError("prefix_ratio is too small to produce a prefix token")
    return normalized


def _dataset_cache_dir(model_path: str, config: dict[str, Any], cache_root: str | Path) -> Path:
    cache_description = {"version": 1, "model_path": os.path.abspath(model_path), **config}
    digest = hashlib.sha256(json.dumps(cache_description, sort_keys=True).encode()).hexdigest()[:20]
    return Path(cache_root) / digest


def _is_complete_dataset(dataset_file: Path, expected_rows: int) -> bool:
    if not dataset_file.is_file():
        return False
    try:
        with dataset_file.open(encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip()) == expected_rows
    except OSError:
        return False


def _write_dataset(dataset_file: Path, samples: list[str]) -> None:
    temporary_file = dataset_file.with_name(f"{dataset_file.name}.{os.getpid()}.tmp")
    try:
        with temporary_file.open("w", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps({"question": sample, "answer": "none"}, ensure_ascii=False) + "\n")
        os.replace(temporary_file, dataset_file)
    finally:
        if temporary_file.exists():
            temporary_file.unlink()


def generate_benchmark_dataset(
    *,
    model_path: str,
    config: dict[str, Any],
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    tokenizer: Any | None = None,
) -> str:
    """Generate a deterministic fixed-length performance dataset.

    The returned path is a directory containing ``test.jsonl``, matching the
    directory layout expected by the AISBench GSM8K dataset configuration.
    """
    normalized = _validate_config(config)
    dataset_dir = _dataset_cache_dir(model_path, normalized, cache_root)
    dataset_file = dataset_dir / "test.jsonl"
    lock_path = dataset_dir.with_suffix(".lock")

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(str(lock_path)):
        if _is_complete_dataset(dataset_file, normalized["num_samples"]):
            logger.info("Reusing generated benchmark dataset: %s", dataset_dir)
            return str(dataset_dir)

        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=normalized["trust_remote_code"],
            )

        dataset_dir.mkdir(parents=True, exist_ok=True)
        if normalized["type"] == "fixed":
            rng = random.Random(normalized["seed"])
            samples = [
                _build_fixed_sample(
                    tokenizer,
                    rng.choice(SEED_TEXTS),
                    normalized["input_len"],
                    rng,
                    index,
                )
                for index in range(normalized["num_samples"])
            ]
        else:
            samples = _build_prefix_samples(
                tokenizer,
                input_length=normalized["input_len"],
                num_samples=normalized["num_samples"],
                prefix_ratio=normalized["prefix_ratio"],
                prefix_num=normalized["prefix_num"],
                seed=normalized["seed"],
            )

        _write_dataset(dataset_file, samples)
        metadata_file = dataset_dir / "metadata.json"
        metadata_file.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("Generated benchmark dataset with %d samples: %s", len(samples), dataset_dir)
        return str(dataset_dir)
