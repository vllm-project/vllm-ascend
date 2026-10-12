# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend port of test_pard2_anchor_acceptance_length in vLLM PR #49406.

Reference: https://github.com/ganeshr10/vllm/blob/
9a54e5a94f294c2b991a0f606dd25cf16f77c6ea/tests/v1/e2e/spec_decode/acceptance_rates/pard2/test_pard2.py

Use GSM8K's first 50 test questions with its first five training examples,
plain completion prompts, greedy sampling, K=15 and max_tokens=256. This is
an acceptance-length regression, not an answer-accuracy or performance test.
It requires an Ascend NPU and vLLM containing the upstream PARD-2 implementation.
Models use repository IDs like the neighboring DFlash test. GSM8K is downloaded
from the upstream CI dataset and reused from pytest's cache on subsequent runs.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import pytest
from vllm import SamplingParams
from vllm.assets.base import VLLM_S3_BUCKET_URL
from vllm.platforms import current_platform
from vllm.v1.metrics.reader import Counter, Metric

from tests.e2e.conftest import VllmRunner

PARD2_MAIN_MODEL = ["Qwen/Qwen3-8B"]
PARD2_MODELS = ["amd/PARD2-Qwen3-8B"]

NUM_QUESTIONS = 50
NUM_SHOTS = 5
NUM_SPECULATIVE_TOKENS = 15
MIN_ACCEPTANCE_LENGTH = 4.0


def _read_gsm8k_split(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip() and not line.startswith("#")]
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) for key in ("question", "answer")):
            raise ValueError(f"{path}: record {index + 1} must contain string question and answer fields")
    return rows


def _build_gsm8k_prompts(data_dir: Path) -> list[str]:
    # Do not import upstream tests: this repository has its own `tests` package.
    # Match _build_gsm8k_prompts(num_questions=50, num_shots=5, gen_prefix="").
    train = _read_gsm8k_split(data_dir / "train.jsonl")
    test = _read_gsm8k_split(data_dir / "test.jsonl")
    assert len(train) >= NUM_SHOTS, f"GSM8K train.jsonl needs at least {NUM_SHOTS} records"
    assert len(test) >= NUM_QUESTIONS, f"GSM8K test.jsonl needs at least {NUM_QUESTIONS} records"
    examples = "".join(f"Question: {row['question']}\nAnswer: {row['answer']}\n\n" for row in train[:NUM_SHOTS])
    return [examples + f"Question: {row['question']}\nAnswer:" for row in test[:NUM_QUESTIONS]]


@pytest.fixture(scope="module")
def pard2_gsm8k_prompts(pytestconfig: pytest.Config) -> list[str]:
    data_dir = pytestconfig.cache.mkdir("pard2_gsm8k")
    for split in ("train", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.is_file():
            url = f"{VLLM_S3_BUCKET_URL}/ci-datasets/gsm8k/{split}.jsonl"
            with urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
    return _build_gsm8k_prompts(data_dir)


def _compute_acceptance_length(metrics: list[Metric]) -> float:
    required = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_accepted_tokens")
    counts: dict[str, float] = {}
    for metric in metrics:
        name = metric.name.removesuffix("_total")
        if name in required:
            assert isinstance(metric, Counter), f"Expected a counter for {name}"
            counts[name] = counts.get(name, 0.0) + metric.value
    assert all(name in counts for name in required), "Missing speculative decoding counters"
    num_drafts, num_accepted = (counts[name] for name in required)
    assert math.isfinite(num_drafts) and num_drafts > 0, "No valid PARD-2 draft rounds were recorded"
    assert math.isfinite(num_accepted) and 0 <= num_accepted <= NUM_SPECULATIVE_TOKENS * num_drafts, (
        f"Invalid accepted-token count: {num_accepted} across {num_drafts} draft rounds"
    )
    print(f"PARD-2 draft rounds: {num_drafts:.0f}; accepted draft tokens: {num_accepted:.0f}")
    return 1.0 + num_accepted / num_drafts


@pytest.mark.parametrize("model", PARD2_MAIN_MODEL)
@pytest.mark.parametrize("pard2_model", PARD2_MODELS)
@pytest.mark.parametrize("max_tokens", [256])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("compilation_config", [pytest.param({}, id="eager")])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
def test_pard2_anchor_acceptance_length(
    model: str,
    pard2_model: str,
    max_tokens: int,
    enforce_eager: bool,
    compilation_config: dict,
    pard2_gsm8k_prompts: list[str],
) -> None:
    """Catch loss of token 0 / its zero target feature through acceptance collapse."""
    assert current_platform.device_type == "npu", "This regression must execute on an Ascend NPU"
    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
    # Follow the shared Ascend runner defaults, including block_size=16,
    # enable_chunked_prefill=True and trust_remote_code=True. Prompt construction,
    # sampling and the acceptance threshold match the upstream PARD-2 test.
    with VllmRunner(
        model,
        dtype="bfloat16",
        speculative_config={
            "method": "pard2",
            "model": pard2_model,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            "draft_tensor_parallel_size": 1,
        },
        max_model_len=4096,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        compilation_config=compilation_config,
        # Ascend adapter settings: both draft passes are eager and PP/PCP/DCP=1.
        # Explicit capacities also keep K=15 dummy batches within the draft buffer.
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.9,
        async_scheduling=False,
        enable_prefix_caching=False,
        generation_config="vllm",
        seed=0,
    ) as runner:
        # Submit all 50 completion prompts at once, without a chat template,
        # benchmark warmup, or a concurrency-1 admission loop.
        outputs = runner.model.generate(pard2_gsm8k_prompts, sampling_params)
        assert len(outputs) == NUM_QUESTIONS and all(output.finished for output in outputs), (
            "The GSM8K regression did not complete all 50 requests"
        )
        metrics = runner.model.get_metrics()

    acceptance_length = _compute_acceptance_length(metrics)
    print(f"PARD-2 anchor acceptance length: {acceptance_length:.4f} (min {MIN_ACCEPTANCE_LENGTH})")
    # Upstream reports ~5.19 for these prompts; its ~7.7 is a different dataset.
    assert acceptance_length >= MIN_ACCEPTANCE_LENGTH, (
        f"PARD-2 acceptance length {acceptance_length:.4f} below min {MIN_ACCEPTANCE_LENGTH}; "
        "check the anchor row, target-feature alignment, attention metadata and KV cache"
    )
