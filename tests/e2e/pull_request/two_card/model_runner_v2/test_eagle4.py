# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""EAGLE-4 NPU acceptance and target agreement, including a >4096-token prompt.

Requires the EAGLE-4 vLLM hooks and eagle-harness==0.1.1 installed on the worker.
"""

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

pytest.importorskip("eagle_harness")

TARGET = "Qwen/Qwen3.8-27B"
DRAFT = "EAGLE-Corp/Qwen3.8-27B-EAGLE4"
PROMPTS = [
    "Write a Python function that returns the Fibonacci numbers below 100.",
    "Explain why a hash table has amortized constant-time lookup.",
    "List the first ten prime numbers and explain what makes them prime.",
]


def assert_logprobs_agree(reference, actual):
    for expected, output in zip(reference, actual, strict=True):
        expected, output = expected.outputs[0], output.outputs[0]
        assert expected.token_ids == output.token_ids
        for token, expected_row, row in zip(output.token_ids, expected.logprobs, output.logprobs, strict=True):
            assert row[token].logprob == pytest.approx(expected_row[token].logprob, abs=0.05)


@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "acl_graph"])
def test_eagle4_chain_target_agreement_and_slot_reuse(monkeypatch, enforce_eager):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    settings = dict(
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=4,
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=enforce_eager,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        gpu_memory_utilization=0.85,
        disable_log_stats=False,
        generation_config="vllm",
    )
    params = SamplingParams(temperature=0, seed=17, max_tokens=64, logprobs=5)
    with VllmRunner(TARGET, **settings) as runner:
        tokenizer = runner.model.get_tokenizer()
        long_prompt = tokenizer.encode("A short note about computer science. ", add_special_tokens=False)
        long_prompt = (long_prompt * (4200 // len(long_prompt) + 1))[:4200]
        prompts = [*PROMPTS, {"prompt_token_ids": long_prompt}]
        reference = runner.model.generate(prompts, params)

    with VllmRunner(
        TARGET,
        **settings,
        speculative_config={"method": "eagle4", "model": DRAFT, "num_speculative_tokens": 3},
    ) as runner:
        actual = runner.model.generate(prompts, params)
        assert_logprobs_agree(reference, actual)
        # Exercise recycled request slots and a cached prompt after rejection.
        repeated = runner.model.generate(list(reversed(prompts)), params)
        assert_logprobs_agree(list(reversed(reference)), repeated)
        sampled = runner.model.generate(PROMPTS, SamplingParams(temperature=1, seed=31, max_tokens=32, logprobs=5))
        for output in sampled:
            tokens = output.outputs[0]
            assert tokens.token_ids
            assert len(tokens.token_ids) == len(tokens.logprobs)
            for token, row in zip(tokens.token_ids, tokens.logprobs, strict=True):
                assert token in row
                assert row[token].logprob <= 0
        metrics = {metric.name: metric for metric in runner.model.get_metrics()}
        drafts = metrics["vllm:spec_decode_num_drafts"].value
        accepted = metrics["vllm:spec_decode_num_accepted_tokens"].value
        assert drafts > 0
        assert accepted > 0, "The draft must accept tokens; starting successfully alone is insufficient."

    # Check sampled tokens on their own prefixes, rather than assuming that a
    # speculative sampler consumes random numbers in the target-only order.
    score_prompts = [
        {"prompt_token_ids": output.prompt_token_ids + list(output.outputs[0].token_ids)} for output in sampled
    ]
    with VllmRunner(TARGET, **{**settings, "enable_prefix_caching": False}) as runner:
        scored = runner.model.generate(score_prompts, SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1))
    for output, score in zip(sampled, scored, strict=True):
        offset = len(output.prompt_token_ids)
        tokens = output.outputs[0]
        for index, (token, row) in enumerate(zip(tokens.token_ids, tokens.logprobs, strict=True)):
            assert row[token].logprob == pytest.approx(score.prompt_logprobs[offset + index][token].logprob, abs=0.05)
