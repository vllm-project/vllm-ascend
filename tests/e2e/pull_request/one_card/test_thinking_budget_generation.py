# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real-model budget checks, including n-gram verification across the budget."""

import json

import pytest
from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.thinking_budget_generation_utils import (
    BUDGETS,
    KV_CACHE_BYTES,
    MAX_OUTPUT_TOKENS,
    PROBABILISTIC_MAX_OUTPUT_TOKENS,
    _speculative_config,
)


@pytest.fixture(autouse=True)
def use_v2_spawn_workers(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-1.7B"])
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("speculative", [None, "ngram_gpu", "draft_model", "draft_model_probabilistic"])
def test_thinking_budget_mixed_generation(monkeypatch, model_name, enforce_eager, speculative):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    spec = _speculative_config(speculative)
    max_output_tokens = (
        PROBABILISTIC_MAX_OUTPUT_TOKENS if speculative == "draft_model_probabilistic" else MAX_OUTPUT_TOKENS
    )
    with VllmRunner(
        model_name,
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=512,
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
        for temperature in [0.0, 0.6]:
            params = [
                SamplingParams(
                    temperature=temperature,
                    seed=42,
                    max_tokens=max_output_tokens,
                    thinking_token_budget=budget,
                )
                for budget in BUDGETS
            ]
            outputs = runner.model.generate([prompt] * len(params), params, use_tqdm=False)
            for budget, output in zip(BUDGETS, outputs):
                result = output.outputs[0]
                ids = list(result.token_ids)
                end = ids.index(end_ids[0]) if end_ids[0] in ids else len(ids)
                print(
                    json.dumps(
                        {
                            "runner": "v2",
                            "request_id": output.request_id,
                            "budget": budget,
                            "thinking_tokens": end,
                            "token_ids": ids,
                            "text": result.text,
                            "finish_reason": result.finish_reason,
                            "eager": enforce_eager,
                            "temperature": temperature,
                            "speculative": speculative,
                        },
                        ensure_ascii=False,
                    )
                )
                if budget is not None and budget >= 0:
                    assert end_ids[0] in ids, (budget, result.text)
                    assert end <= budget, (budget, end, result.text)
                elif result.finish_reason == "length":
                    # Unlimited thinking still obeys the total output limit.
                    assert len(ids) == max_output_tokens
                    continue
                assert end_ids[0] in ids, (budget, result.text)
                assert "5" in tokenizer.decode(ids[end + 1 :], skip_special_tokens=True), (budget, result.text)
                assert tokenizer.decode(ids[end + 1 :], skip_special_tokens=True).strip(), (budget, result.text)
                assert result.finish_reason == "stop", (budget, result.text)
            # Repeat with new requests to catch stale state after slots are reused.
            outputs = runner.model.generate([prompt] * 2, [params[-1], params[BUDGETS.index(0)]], use_tqdm=False)
            for budget, output in zip([BUDGETS[-1], 0], outputs):
                assert budget is not None
                assert list(output.outputs[0].token_ids).index(end_ids[0]) <= budget


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_plain_model_unlimited_budget_preserves_output(enforce_eager):
    with VllmRunner(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=1024,
        max_num_seqs=2,
        kv_cache_memory_bytes=KV_CACHE_BYTES,
        enforce_eager=enforce_eager,
    ) as runner:
        prompt = "The capital of France is"
        for temperature in [0.0, 0.6]:
            results = []
            for budget in [None, -1, None]:
                params = SamplingParams(temperature=temperature, seed=42, max_tokens=32, thinking_token_budget=budget)
                outputs = runner.model.generate([prompt], params, use_tqdm=False)
                results.append(outputs[0].outputs[0].token_ids)
            assert all(result == results[0] for result in results)
            print(json.dumps({"plain_model_equal": True, "eager": enforce_eager, "temperature": temperature}))
        for budget in [0, 16]:
            with pytest.raises(VLLMValidationError, match="thinking_token_budget.*reasoning_config"):
                runner.model.generate(
                    [prompt], SamplingParams(thinking_token_budget=budget, max_tokens=32), use_tqdm=False
                )
