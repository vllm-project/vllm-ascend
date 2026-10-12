# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os
import time
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free


@pytest.mark.parametrize("num_speculative_tokens,enforce_eager", [(1, True), (3, True), (3, False)])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_draft_model_matches_target(num_speculative_tokens, enforce_eager):
    """Check correction/rejection, chunked prefill, prefix hits and graph target execution."""
    prompts = [
        "Hello, my name is",
        "Explain why the sky is blue.",
        "Write a Python function to sort integers.",
        "Explain the following story: " + "A traveler crossed the mountains and reached a village. " * 16,
    ]
    params = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)
    config = dict(
        max_model_len=512,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        async_scheduling=True,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
    )
    with VllmRunner("Qwen/Qwen3-1.7B", **config) as runner:
        expected = runner.model.generate(prompts, params)
        start = time.perf_counter()
        runner.model.generate(prompts, params)
        target_seconds = time.perf_counter() - start
    with VllmRunner(
        "Qwen/Qwen3-1.7B",
        **config,
        speculative_config=dict(
            method="draft_model", model="Qwen/Qwen3-0.6B", num_speculative_tokens=num_speculative_tokens
        ),
    ) as runner:
        actual = runner.model.generate(prompts, params)
        start = time.perf_counter()
        prefix_hit_outputs = runner.model.generate(prompts, params)
        draft_seconds = time.perf_counter() - start
        metrics = runner.model.get_metrics()
    expected_ids = [out.outputs[0].token_ids for out in expected]
    assert [out.outputs[0].token_ids for out in actual] == expected_ids
    assert [out.outputs[0].token_ids for out in prefix_hit_outputs] == expected_ids
    counts = {metric.name: metric.value for metric in metrics if hasattr(metric, "value")}
    assert counts["vllm:spec_decode_num_drafts"] > 0
    assert counts["vllm:spec_decode_num_accepted_tokens"] > 0
    assert counts["vllm:spec_decode_num_draft_tokens"] > counts["vllm:spec_decode_num_accepted_tokens"]
    print(
        f"target={target_seconds:.3f}s, draft_model={draft_seconds:.3f}s, speedup={target_seconds / draft_seconds:.3f}"
    )


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_draft_model_probabilistic_sampling():
    prompts = ["Tell a story about a traveler.", "Write a poem about the ocean."]
    with VllmRunner(
        "Qwen/Qwen3-1.7B",
        max_model_len=512,
        enforce_eager=True,
        async_scheduling=True,
        speculative_config=dict(
            method="draft_model",
            model="Qwen/Qwen3-0.6B",
            num_speculative_tokens=3,
            draft_sample_method="probabilistic",
        ),
    ) as runner:
        ids = []
        for seed in (42, 1234):
            outputs = runner.model.generate(
                prompts, SamplingParams(temperature=0.7, seed=seed, max_tokens=64, ignore_eos=True)
            )
            ids.append([out.outputs[0].token_ids for out in outputs])
    assert ids[0] != ids[1]
    assert all(len(tokens) == 64 and len(set(tokens)) > 1 for run in ids for tokens in run)
