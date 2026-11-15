# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact nonbinding-budget comparisons with batch-invariant model computation."""

import json

import pytest
from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.thinking_budget_generation_utils import (
    KV_CACHE_BYTES,
    _speculative_config,
)


@pytest.fixture(autouse=True)
def use_v2_spawn_workers(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-1.7B"])
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("speculative", [None, "ngram_gpu", "draft_model", "draft_model_probabilistic"])
def test_nonbinding_thinking_budget_preserves_output(monkeypatch, model_name, enforce_eager, speculative):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    spec = _speculative_config(speculative)
    with VllmRunner(
        model_name,
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        # Keep prefill computation consistent for exact cross-request output
        # comparisons. Serving tests separately cover the default prefix cache.
        enable_prefix_caching=False,
        kv_cache_memory_bytes=KV_CACHE_BYTES,
        enforce_eager=enforce_eager,
        speculative_config=spec,
        reasoning_config={
            "reasoning_parser": "qwen3",
            "reasoning_start_str": "<think>",
            "reasoning_end_str": "</think>",
        },
    ) as runner:
        tokenizer = runner.model.get_tokenizer()
        end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        assert len(end_ids) == 1
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 2 + 3? Give the final answer as a number."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        # An explicit marker at the end makes the output-token budget observable
        # without counting template whitespace already present in the prompt.
        prompt = prompt.rstrip()
        if not prompt.endswith("<think>"):
            prompt += "\n<think>"
        for invalid in [2**64, 2**80]:
            with pytest.raises(VLLMValidationError, match="thinking_token_budget must be at most") as error:
                runner.model.generate(prompt, SamplingParams(thinking_token_budget=invalid), use_tqdm=False)
            assert error.value.parameter == "thinking_token_budget"
            print(json.dumps({"invalid_budget": invalid, "error": str(error.value)}))
        # Model forward passes can vary with batch shape in the default BF16
        # path, including when every budget is disabled. Use batch invariance
        # for this exact comparison; default-mode natural generation and
        # serving are covered separately. Include the cold batch without retry.
        boundary_tokens = []
        for budgets in ([None, None], [-1, -1], [None, 2**64 - 1], [2**64 - 1, 2**64 - 1]):
            boundary_outputs = runner.model.generate(
                [prompt] * 2,
                [SamplingParams(temperature=0, max_tokens=16, thinking_token_budget=b) for b in budgets],
                use_tqdm=False,
            )
            boundary_tokens.append([output.outputs[0].token_ids for output in boundary_outputs])
        # Repeat the disabled baseline to distinguish output drift from a
        # nonbinding-budget regression if the comparison below fails.
        baseline_repeat = runner.model.generate(
            [prompt] * 2,
            [SamplingParams(temperature=0, max_tokens=16) for _ in range(2)],
            use_tqdm=False,
        )
        baseline_repeat_tokens = [output.outputs[0].token_ids for output in baseline_repeat]
        nonbinding_output_equal = all(tokens == boundary_tokens[0] for tokens in boundary_tokens[1:])
        print(
            json.dumps(
                {
                    "transport_upper_bound": 2**64 - 1,
                    "nonbinding_output_equal": nonbinding_output_equal,
                    "boundary_token_ids": boundary_tokens,
                    "batch_invariant": True,
                    "disabled_repeat_equal": baseline_repeat_tokens == boundary_tokens[0],
                    "disabled_repeat_token_ids": baseline_repeat_tokens,
                }
            )
        )
        assert nonbinding_output_equal, boundary_tokens
        assert baseline_repeat_tokens == boundary_tokens[0], (baseline_repeat_tokens, boundary_tokens[0])
