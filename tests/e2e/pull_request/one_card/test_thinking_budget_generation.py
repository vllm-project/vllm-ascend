# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real-model budget checks, including n-gram verification across the budget."""

import json

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

BUDGETS = [None, -1, 0, 1, 2, 3, 4, 16, 64, 128]
MAX_OUTPUT_TOKENS = 512
KV_CACHE_BYTES = 2 * 1024**3


@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-1.7B"])
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("speculative", [False, True])
def test_thinking_budget_mixed_generation(monkeypatch, model_name, enforce_eager, speculative, runner_version="1"):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", runner_version)
    spec = None
    if speculative:
        spec = {
            "method": "ngram_gpu",
            "num_speculative_tokens": 3,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 5,
        }
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
        for temperature in [0.0, 0.6]:
            params = [
                SamplingParams(
                    temperature=temperature,
                    seed=42,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    thinking_token_budget=budget,
                )
                for budget in BUDGETS
            ]
            outputs = runner.model.generate([prompt] * len(params), params)
            for budget, output in zip(BUDGETS, outputs):
                result = output.outputs[0]
                ids = list(result.token_ids)
                end = ids.index(end_ids[0]) if end_ids[0] in ids else len(ids)
                print(
                    json.dumps(
                        {
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
                    assert len(ids) == MAX_OUTPUT_TOKENS
                    continue
                assert end_ids[0] in ids, (budget, result.text)
                assert "5" in tokenizer.decode(ids[end + 1 :], skip_special_tokens=True), (budget, result.text)
                assert tokenizer.decode(ids[end + 1 :], skip_special_tokens=True).strip(), (budget, result.text)
                assert result.finish_reason == "stop", (budget, result.text)
            # Repeat with new requests to catch stale state after slots are reused.
            outputs = runner.model.generate([prompt] * 2, [params[-1], params[BUDGETS.index(0)]])
            for budget, output in zip([BUDGETS[-1], 0], outputs):
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
                outputs = runner.model.generate([prompt], params)
                results.append(outputs[0].outputs[0].token_ids)
            assert results[0] == results[1] == results[2]


def test_thinking_budget_preemption_matches_unconstrained_cache(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    common = dict(
        max_model_len=256,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        disable_log_stats=False,
        enable_prefix_caching=False,
        enforce_eager=True,
        reasoning_config={
            "reasoning_parser": "qwen3",
            "reasoning_start_str": "<think>",
            "reasoning_end_str": "</think>",
        },
    )
    outputs = []
    for blocks in [128, 6]:
        with VllmRunner("Qwen/Qwen3-1.7B", num_gpu_blocks_override=blocks, **common) as runner:
            tokenizer = runner.model.get_tokenizer()
            start = tokenizer.encode("<think>", add_special_tokens=False)
            filler = tokenizer.encode("Think carefully about basic arithmetic. ", add_special_tokens=False)
            prompt = (filler * 120)[: 120 - len(start)] + start
            budgets = [0, 8, 16, 32]
            params = [
                SamplingParams(
                    temperature=0,
                    max_tokens=128,
                    ignore_eos=True,
                    thinking_token_budget=b,
                    # A dominant ordinary token isolates scheduler state from
                    # BF16 near-tie changes between batch/recompute shapes.
                    logit_bias={10: 100.0},
                )
                for b in budgets
            ]
            results = runner.model.generate([{"prompt_token_ids": prompt}] * 4, params)
            end = tokenizer.encode("</think>", add_special_tokens=False)[0]
            for budget, result in zip(budgets, results):
                expected = [10] * budget + [end] + [10] * (127 - budget)
                assert list(result.outputs[0].token_ids) == expected
            metrics = runner.model.get_metrics()
            preemptions = sum(metric.value for metric in metrics if metric.name == "vllm:num_preemptions")
            print(
                json.dumps(
                    {
                        "kv_blocks": blocks,
                        "preemptions": preemptions,
                        "outputs": [list(r.outputs[0].token_ids) for r in results],
                    }
                )
            )
            if blocks == 6:
                assert preemptions > 0, metrics
            else:
                assert preemptions == 0, metrics
            outputs.append([list(result.outputs[0].token_ids) for result in results])
    assert outputs[0] == outputs[1]
