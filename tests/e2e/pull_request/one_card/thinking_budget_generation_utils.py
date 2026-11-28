# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared real-model thinking-budget matrix helpers."""

import json

from vllm import SamplingParams

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


def check_preemption_budget_and_answer(monkeypatch, enforce_eager, speculative, temperature, constrain_tokens):
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
