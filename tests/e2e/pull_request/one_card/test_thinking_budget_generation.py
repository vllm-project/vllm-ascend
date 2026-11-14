# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real-model budget checks, including n-gram verification across the budget."""

import json

import pytest
from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError

from tests.e2e.conftest import VllmRunner

BUDGETS = [None, -1, 0, 1, 2, 3, 4, 16, 64, 128]
MAX_OUTPUT_TOKENS = 512
PROBABILISTIC_MAX_OUTPUT_TOKENS = 2048
KV_CACHE_BYTES = 2 * 1024**3
PREEMPTION_PROMPT_TOKENS = 120
PREEMPTION_OUTPUT_TOKENS = 128
NATURAL_PREEMPTION_OUTPUT_TOKENS = 384
PROBABILISTIC_PREEMPTION_REQUESTS = 16
PROBABILISTIC_PREEMPTION_BLOCKS = 18
PROBABILISTIC_PREEMPTION_CONTEXT = 2176
PROBABILISTIC_PREEMPTION_BATCHED_TOKENS = 4096


def _speculative_config(speculative):
    if speculative is None:
        return None
    if speculative == "ngram_gpu":
        spec = {
            "method": "ngram_gpu",
            "num_speculative_tokens": 3,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 5,
        }
    else:
        spec = {
            "method": "draft_model",
            "model": "Qwen/Qwen3-0.6B",
            "num_speculative_tokens": 3,
        }
    if speculative == "draft_model_probabilistic":
        spec["draft_sample_method"] = "probabilistic"
    return spec


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
        boundary_outputs = runner.model.generate(
            [prompt] * 2,
            [SamplingParams(temperature=0, max_tokens=16, thinking_token_budget=b) for b in [None, 2**64 - 1]],
            use_tqdm=False,
        )
        assert boundary_outputs[0].outputs[0].token_ids == boundary_outputs[1].outputs[0].token_ids
        print(json.dumps({"transport_upper_bound": 2**64 - 1, "nonbinding_output_equal": True}))
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


@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("speculative", [None, "ngram_gpu", "draft_model", "draft_model_probabilistic"])
@pytest.mark.parametrize("temperature", [0.0, 0.6])
@pytest.mark.parametrize("constrain_tokens", [True, False], ids=["constrained", "natural"])
def test_thinking_budget_preemption_preserves_budget_and_answer(
    monkeypatch, enforce_eager, speculative, temperature, constrain_tokens
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    # Leave room for a natural answer while keeping enough simultaneous
    # near-block-boundary prompts to force preemption in the small cache.
    probabilistic_natural = speculative == "draft_model_probabilistic" and not constrain_tokens
    num_requests = PROBABILISTIC_PREEMPTION_REQUESTS if probabilistic_natural else 4
    low_cache_blocks = PROBABILISTIC_PREEMPTION_BLOCKS if probabilistic_natural else 6
    max_output_tokens = PROBABILISTIC_MAX_OUTPUT_TOKENS if probabilistic_natural else NATURAL_PREEMPTION_OUTPUT_TOKENS
    outputs = []
    for blocks in [128, low_cache_blocks]:
        with VllmRunner(
            "Qwen/Qwen3-1.7B",
            num_gpu_blocks_override=blocks,
            max_model_len=(
                PROBABILISTIC_PREEMPTION_CONTEXT if probabilistic_natural else (256 if constrain_tokens else 512)
            ),
            max_num_seqs=num_requests,
            max_num_batched_tokens=PROBABILISTIC_PREEMPTION_BATCHED_TOKENS if probabilistic_natural else 256,
            disable_log_stats=False,
            enable_prefix_caching=False,
            enforce_eager=enforce_eager,
            speculative_config=_speculative_config(speculative),
            reasoning_config={
                "reasoning_parser": "qwen3",
                "reasoning_start_str": "<think>",
                "reasoning_end_str": "</think>",
            },
        ) as runner:
            tokenizer = runner.model.get_tokenizer()
            start = tokenizer.encode("<think>", add_special_tokens=False)
            filler = tokenizer.encode("Think carefully about basic arithmetic. ", add_special_tokens=False)
            prompt = (filler * PREEMPTION_PROMPT_TOKENS)[: PREEMPTION_PROMPT_TOKENS - len(start)] + start
            if not constrain_tokens:
                chat = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "What is 2 + 3? Give the final answer as a number."}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                ).rstrip()
                if not chat.endswith("<think>"):
                    chat += "\n<think>"
                chat_ids = tokenizer.encode(chat, add_special_tokens=False)
                assert len(chat_ids) < PREEMPTION_PROMPT_TOKENS
                prompt = (filler * PREEMPTION_PROMPT_TOKENS)[: PREEMPTION_PROMPT_TOKENS - len(chat_ids)] + chat_ids
            budgets = [0, 8, 16, 32] * (num_requests // 4)
            params = [
                SamplingParams(
                    temperature=temperature,
                    seed=42,
                    max_tokens=PREEMPTION_OUTPUT_TOKENS if constrain_tokens else max_output_tokens,
                    ignore_eos=constrain_tokens,
                    # Qwen3 model-card settings curb repetition in natural answers.
                    # Greedy is also exercised as an acceptance requirement.
                    top_p=0.95 if probabilistic_natural else 1.0,
                    top_k=20 if probabilistic_natural else -1,
                    presence_penalty=1.0 if probabilistic_natural else 0.0,
                    thinking_token_budget=b,
                    # Restrict ordinary tokens to isolate scheduler state from
                    # BF16 near ties. The budget must restore its masked end
                    # marker, including during speculative verification.
                    allowed_token_ids=[10] if constrain_tokens else None,
                )
                for b in budgets
            ]
            results = runner.model.generate([{"prompt_token_ids": prompt}] * num_requests, params, use_tqdm=False)
            end = tokenizer.encode("</think>", add_special_tokens=False)[0]
            for budget, result in zip(budgets, results):
                ids = list(result.outputs[0].token_ids)
                print(
                    json.dumps(
                        {
                            "preemption_request": True,
                            "kv_blocks": blocks,
                            "budget": budget,
                            "eager": enforce_eager,
                            "speculative": speculative,
                            "temperature": temperature,
                            "constrain_tokens": constrain_tokens,
                            "top_p": 0.95 if probabilistic_natural else 1.0,
                            "top_k": 20 if probabilistic_natural else -1,
                            "presence_penalty": 1.0 if probabilistic_natural else 0.0,
                            "token_ids": ids,
                            "thinking_tokens": ids.index(end) if end in ids else len(ids),
                            "finish_reason": result.outputs[0].finish_reason,
                            "text": result.outputs[0].text,
                        },
                        ensure_ascii=False,
                    )
                )
                if constrain_tokens:
                    expected = [10] * budget + [end] + [10] * (PREEMPTION_OUTPUT_TOKENS - 1 - budget)
                    assert ids == expected
                else:
                    assert end in ids
                    boundary = ids.index(end)
                    assert boundary <= budget
                    assert "5" in tokenizer.decode(ids[boundary + 1 :], skip_special_tokens=True)
                    assert result.outputs[0].finish_reason == "stop", result.outputs[0].text
            metrics = runner.model.get_metrics()
            preemptions = sum(metric.value for metric in metrics if metric.name == "vllm:num_preemptions")
            print(
                json.dumps(
                    {
                        "runner": "v2",
                        "eager": enforce_eager,
                        "speculative": speculative,
                        "kv_blocks": blocks,
                        "constrain_tokens": constrain_tokens,
                        "budgets": budgets,
                        "temperature": temperature,
                        "preemptions": preemptions,
                        "outputs": [list(r.outputs[0].token_ids) for r in results],
                    }
                )
            )
            if blocks == low_cache_blocks:
                assert preemptions > 0, metrics
            else:
                assert preemptions == 0, metrics
            outputs.append([list(result.outputs[0].token_ids) for result in results])
    if constrain_tokens:
        assert outputs[0] == outputs[1]
    print(
        json.dumps(
            {
                "preemption_outputs_equal": outputs[0] == outputs[1],
                "eager": enforce_eager,
                "speculative": speculative,
                "temperature": temperature,
                "constrain_tokens": constrain_tokens,
            }
        )
    )
