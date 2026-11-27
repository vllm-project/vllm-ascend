# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import os
from typing import TypedDict
from unittest.mock import patch

import pytest
from huggingface_hub import snapshot_download as hf_snapshot_download
from huggingface_hub.constants import HF_HUB_OFFLINE
from modelscope import snapshot_download as ms_snapshot_download  # type: ignore[import-untyped]
from vllm import SamplingParams, envs
from vllm.v1.metrics.reader import Counter

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = "Qwen/Qwen3.5-0.8B"
PROMPTS = ["The capital of France is", "1, 2, 3, 4, 5,"]
MAX_TOKENS = 16
NUM_SPECULATIVE_TOKENS = 2


class _RunnerKwargs(TypedDict):
    dtype: str
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    language_model_only: bool
    enable_prefix_caching: bool
    generation_config: str
    logprobs_mode: str


def _spec_decode_count(runner, metric_name):
    counters = [metric for metric in runner.model.get_metrics() if metric.name == metric_name]
    assert all(isinstance(metric, Counter) for metric in counters)
    return sum(metric.value for metric in counters)


@pytest.mark.e2e_model(MODEL)
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "full_decode_only"])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mtp_logprobs_match_greedy_reference(enforce_eager: bool) -> None:
    if envs.VLLM_USE_MODELSCOPE:
        model_path = ms_snapshot_download(MODEL, local_files_only=HF_HUB_OFFLINE)
    else:
        model_path = hf_snapshot_download(MODEL, local_files_only=HF_HUB_OFFLINE)
    runner_kwargs: _RunnerKwargs = {
        "dtype": "bfloat16",
        "max_model_len": 256,
        "max_num_seqs": len(PROMPTS),
        "gpu_memory_utilization": 0.3,
        "language_model_only": True,
        "enable_prefix_caching": False,
        "generation_config": "vllm",
        "logprobs_mode": "raw_logprobs",
    }
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=5, ignore_eos=True)
    with VllmRunner(model_path, enforce_eager=True, **runner_kwargs) as runner:
        reference = runner.generate_w_logprobs(PROMPTS, params, use_tqdm=False)

    compilation_config = (
        {}
        if enforce_eager
        else {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [3, 6],
        }
    )
    with VllmRunner(
        model_path,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": NUM_SPECULATIVE_TOKENS},
        compilation_config=compilation_config,
        **runner_kwargs,
    ) as runner:
        drafts_before = _spec_decode_count(runner, "vllm:spec_decode_num_drafts")
        accepted_before = _spec_decode_count(runner, "vllm:spec_decode_num_accepted_tokens")
        actual = runner.generate_w_logprobs(PROMPTS, params, use_tqdm=False)
        assert _spec_decode_count(runner, "vllm:spec_decode_num_drafts") > drafts_before
        assert _spec_decode_count(runner, "vllm:spec_decode_num_accepted_tokens") > accepted_before

    assert len(actual) == len(reference) == len(PROMPTS)
    for (token_ids, _, logprobs), (ref_ids, _, ref_logprobs) in zip(actual, reference, strict=True):
        assert len(token_ids) == MAX_TOKENS
        assert token_ids == ref_ids
        assert logprobs is not None and ref_logprobs is not None
        for token_id, scores, ref_scores in zip(token_ids, logprobs, ref_logprobs, strict=True):
            assert scores[token_id].logprob == pytest.approx(ref_scores[token_id].logprob, abs=0.1)
