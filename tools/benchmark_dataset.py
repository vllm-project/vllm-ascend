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
PREFIX_DATASET_SUBDIR = "prefix"
SEPARATOR_TOKEN_COUNT = 3

# Update this single path when running in another local or CI environment.
GSM8K_SOURCE_PATH = Path("/mnt/share/c00893695/datasets/GSM8K.jsonl")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_texts(source_dataset_path: str) -> list[str]:
    source_texts = []
    with Path(source_dataset_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {source_dataset_path}:{line_number}") from error
            question = item.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Missing non-empty question at {source_dataset_path}:{line_number}")
            source_texts.append(question)

    if not source_texts:
        raise ValueError(f"No questions found in source dataset: {source_dataset_path}")
    return source_texts


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


def _build_random_token_ids(tokenizer: Any, rng: random.Random, length: int) -> list[int]:
    if length == 0:
        return []
    vocab_size = len(tokenizer)
    if length > vocab_size:
        raise ValueError(f"Cannot sample {length} unique tokens from a vocabulary of size {vocab_size}")
    return rng.sample(range(vocab_size), length)


def _build_source_sample(tokenizer: Any, source_text: str, input_length: int) -> str:
    token_ids = _fit_token_ids(tokenizer, source_text, input_length)
    return _decode_to_exact_length(tokenizer, token_ids, input_length, source_text)


def _build_prefix_token_ids(
    tokenizer: Any,
    *,
    prefix_length: int,
    prefix_texts: list[str],
) -> list[list[int]]:
    return [
        _fit_token_ids(
            tokenizer,
            f"{prefix_text}\n",
            prefix_length,
        )
        for prefix_text in prefix_texts
    ]


def _build_prefix_samples(
    tokenizer: Any,
    *,
    input_length: int,
    num_samples: int,
    prefix_ratio: float,
    prefix_token_ids: list[list[int]],
    source_texts: list[str],
    seed: int,
) -> list[str]:
    prefix_length = int(input_length * prefix_ratio)
    separator_length = min(SEPARATOR_TOKEN_COUNT, input_length - prefix_length)
    suffix_length = input_length - prefix_length - separator_length
    rng = random.Random(seed)
    prefix_num = len(prefix_token_ids)
    samples = []
    for index in range(num_samples):
        selected_prefix = prefix_token_ids[index % prefix_num]
        separator_ids = _build_random_token_ids(tokenizer, rng, separator_length)
        suffix_seed = rng.choice(source_texts)
        suffix_ids = _fit_token_ids(tokenizer, suffix_seed, suffix_length)
        sample_ids = selected_prefix + separator_ids + suffix_ids
        samples.append(_decode_to_exact_length(tokenizer, sample_ids, input_length, suffix_seed))
    return samples


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset_type = str(config.get("type", "")).lower()
    if dataset_type not in SUPPORTED_DATASET_TYPES:
        raise ValueError(f"dataset_generator.type must be one of {sorted(SUPPORTED_DATASET_TYPES)}")

    warmup_prefix = config.get("warmup_prefix", False)
    if not isinstance(warmup_prefix, bool):
        raise ValueError("dataset_generator.warmup_prefix must be a boolean")

    expanded_path = os.path.expandvars(os.path.expanduser(str(GSM8K_SOURCE_PATH)))
    resolved_path = Path(expanded_path).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"GSM8K source dataset not found: {resolved_path}")
    source_dataset_path = str(resolved_path)
    source_dataset_sha256 = _file_sha256(resolved_path)

    normalized = {
        "type": dataset_type,
        "input_len": int(config["input_len"]),
        "num_samples": int(config["num_samples"]),
        "seed": int(config.get("seed", 1)),
        "prefix_ratio": float(config.get("prefix_ratio", 0.0)),
        "prefix_num": int(config.get("prefix_num", 1)),
        "warmup_prefix": warmup_prefix,
        "dp": int(config.get("dp", 1)),
        "source_dataset_path": source_dataset_path,
        "source_dataset_sha256": source_dataset_sha256,
        "trust_remote_code": bool(config.get("trust_remote_code", True)),
    }
    if normalized["input_len"] < 1:
        raise ValueError("dataset_generator.input_len must be >= 1")
    if normalized["num_samples"] < 1:
        raise ValueError("dataset_generator.num_samples must be >= 1")
    if normalized["prefix_num"] < 1:
        raise ValueError("dataset_generator.prefix_num must be >= 1")
    if normalized["dp"] < 1:
        raise ValueError("dataset_generator.dp must be >= 1")
    if not 0.0 <= normalized["prefix_ratio"] <= 1.0:
        raise ValueError("dataset_generator.prefix_ratio must be in [0, 1]")
    if dataset_type == "fixed" and normalized["prefix_ratio"] != 0.0:
        raise ValueError("fixed datasets do not accept a non-zero prefix_ratio")
    if dataset_type == "fixed" and normalized["warmup_prefix"]:
        raise ValueError("warmup_prefix is only supported for prefix datasets")
    if dataset_type == "prefix" and normalized["prefix_ratio"] <= 0.0:
        raise ValueError("prefix datasets require prefix_ratio > 0")
    if dataset_type == "prefix" and int(normalized["input_len"] * normalized["prefix_ratio"]) == 0:
        raise ValueError("prefix_ratio is too small to produce a prefix token")
    return normalized


def _dataset_cache_dir(model_path: str, config: dict[str, Any], cache_root: str | Path) -> Path:
    cache_description = {"version": 1, "model_path": os.path.abspath(model_path), **config}
    digest = hashlib.sha256(json.dumps(cache_description, sort_keys=True).encode()).hexdigest()[:20]
    model_name = Path(str(model_path).rstrip("/\\")).name or "model"
    model_name = "".join(character if character.isalnum() or character in "-._" else "-" for character in model_name)
    if config["type"] == "fixed":
        dataset_name = f"GSM8K-in{config['input_len']}-num{config['num_samples']}"
    else:
        prefix_percentage = f"{config['prefix_ratio'] * 100:g}"
        dataset_name = f"prefix{prefix_percentage}-in{config['input_len']}-num{config['num_samples']}"
    return Path(cache_root) / f"{dataset_name}-{model_name}-{digest}"


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


def _ensure_empty_train_file(dataset_dir: Path) -> None:
    train_file = dataset_dir / "train.jsonl"
    if train_file.is_file() and train_file.stat().st_size == 0:
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    temporary_file = train_file.with_name(f"{train_file.name}.{os.getpid()}.tmp")
    try:
        temporary_file.write_text("", encoding="utf-8")
        os.replace(temporary_file, train_file)
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

    The returned path contains ``test.jsonl`` and an empty ``train.jsonl``,
    matching the directory layout expected by the AISBench GSM8K configuration.
    """
    normalized = _validate_config(config)
    dataset_dir = _dataset_cache_dir(model_path, normalized, cache_root)
    dataset_file = dataset_dir / "test.jsonl"
    prefix_dataset_dir = dataset_dir / PREFIX_DATASET_SUBDIR
    prefix_dataset_file = prefix_dataset_dir / "test.jsonl"
    lock_path = dataset_dir.with_suffix(".lock")

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(str(lock_path)):
        full_dataset_ready = _is_complete_dataset(dataset_file, normalized["num_samples"])
        prefix_dataset_ready = not normalized["warmup_prefix"] or _is_complete_dataset(
            prefix_dataset_file,
            normalized["prefix_num"] * normalized["dp"],
        )
        if full_dataset_ready and prefix_dataset_ready:
            _ensure_empty_train_file(dataset_dir)
            if normalized["warmup_prefix"]:
                _ensure_empty_train_file(prefix_dataset_dir)
            logger.info("Reusing generated benchmark dataset: %s", dataset_dir)
            return str(dataset_dir)

        source_texts = _load_source_texts(normalized["source_dataset_path"])
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
                _build_source_sample(tokenizer, rng.choice(source_texts), normalized["input_len"])
                for _ in range(normalized["num_samples"])
            ]
        else:
            if normalized["prefix_num"] > len(source_texts):
                raise ValueError(
                    f"prefix_num ({normalized['prefix_num']}) exceeds source question count ({len(source_texts)})"
                )
            prefix_length = int(normalized["input_len"] * normalized["prefix_ratio"])
            prefix_rng = random.Random(normalized["seed"])
            prefix_texts = prefix_rng.sample(source_texts, normalized["prefix_num"])
            prefix_token_ids = _build_prefix_token_ids(
                tokenizer,
                prefix_length=prefix_length,
                prefix_texts=prefix_texts,
            )
            samples = _build_prefix_samples(
                tokenizer,
                input_length=normalized["input_len"],
                num_samples=normalized["num_samples"],
                prefix_ratio=normalized["prefix_ratio"],
                prefix_token_ids=prefix_token_ids,
                source_texts=source_texts,
                seed=normalized["seed"],
            )
            if normalized["warmup_prefix"]:
                prefix_samples = []
                for index, token_ids in enumerate(prefix_token_ids):
                    filler_text = prefix_texts[index]
                    prefix_text = _decode_to_exact_length(tokenizer, token_ids, prefix_length, filler_text)
                    prefix_samples.extend([prefix_text] * normalized["dp"])
                prefix_dataset_dir.mkdir(parents=True, exist_ok=True)
                _write_dataset(prefix_dataset_file, prefix_samples)
                _ensure_empty_train_file(prefix_dataset_dir)

        _write_dataset(dataset_file, samples)
        _ensure_empty_train_file(dataset_dir)
        metadata_file = dataset_dir / "metadata.json"
        metadata_file.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("Generated benchmark dataset with %d samples: %s", len(samples), dataset_dir)
        return str(dataset_dir)


def get_prefix_dataset_path(dataset_path: str | Path) -> str:
    """Return the prefix-only dataset directory for a generated dataset."""
    prefix_dataset_path = Path(dataset_path) / PREFIX_DATASET_SUBDIR
    if not (prefix_dataset_path / "test.jsonl").is_file():
        raise FileNotFoundError(f"Generated prefix dataset not found: {prefix_dataset_path}")
    return str(prefix_dataset_path)
