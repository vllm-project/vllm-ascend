# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-8B PARD-2 acceptance regression on Ascend, following vLLM PR #49406.

GSM8K downloading, JSONL reading and prompt construction are copied from
upstream gsm8k_eval.py. Downloads are reused from the upstream temporary cache.
Calls the prompt builder directly; there is no prompt fixture.
"""

from __future__ import annotations

import ast
import json
import math
import os
import tempfile
from collections.abc import Generator
from unittest.mock import patch

import pytest
import regex as re
import requests
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
INVALID = -9999999


# GSM8K helpers copied from:
# https://github.com/vllm-project/vllm/blob/main/tests/evals/gsm8k/gsm8k_eval.py
# Source Git blob: 4eec87945b6fc8ce6388bf52f3e3a069400f7b2f (2026-10-12).
def download_and_cache_file(url: str, filename: str | None = None) -> str:
    """Download and cache a file from a URL."""
    if filename is None:
        filename = os.path.join(tempfile.gettempdir(), url.split("/")[-1])

    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

    return filename


def load_gsm8k_data() -> tuple[list[dict], list[dict]]:
    """Load GSM8K train and test data."""
    train_url = f"{VLLM_S3_BUCKET_URL}/ci-datasets/gsm8k/train.jsonl"
    test_url = f"{VLLM_S3_BUCKET_URL}/ci-datasets/gsm8k/test.jsonl"

    train_file = download_and_cache_file(train_url)
    test_file = download_and_cache_file(test_url)

    train_data = list(read_jsonl(train_file))
    test_data = list(read_jsonl(test_file))

    return train_data, test_data


def read_jsonl(filename: str) -> Generator[dict, None, None]:
    """Read a JSONL file."""
    with open(filename) as fin:
        for line in fin:
            if not line.startswith("#"):
                yield json.loads(line)


def get_answer_value(answer_str: str) -> int:
    """Extract the numerical answer from the response."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def _build_gsm8k_prompts(
    num_questions: int = 1319,
    num_shots: int = 5,
    gen_prefix: str = "",
) -> tuple[list[str], list[int]]:
    """Build few-shot GSM8K completion prompts and ground-truth labels."""
    if num_questions == 0:
        return [], []
    train_data, test_data = load_gsm8k_data()
    num_questions = min(num_questions, len(test_data))

    few_shot_examples = ""
    for i in range(num_shots):
        few_shot_examples += f"Question: {train_data[i]['question']}\nAnswer:{gen_prefix} {train_data[i]['answer']}\n\n"

    prompts = []
    labels = []
    for i in range(num_questions):
        prompts.append(few_shot_examples + f"Question: {test_data[i]['question']}\nAnswer:{gen_prefix}")
        labels.append(get_answer_value(test_data[i]["answer"]))

    assert all(label != INVALID for label in labels), "Some labels are invalid"
    return prompts, labels


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
) -> None:
    """Catch loss of token 0 / its zero target feature through acceptance collapse."""
    assert current_platform.device_type == "npu", "This regression must execute on an Ascend NPU"
    prompts = _build_gsm8k_prompts(num_questions=NUM_QUESTIONS, num_shots=NUM_SHOTS)[0]
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
        outputs = runner.model.generate(prompts, sampling_params)
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
