# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NPU checks for V1 draftless layout and committed end-sequence progress."""

from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card import test_thinking_budget_generation as generation_checks
from tests.e2e.pull_request.one_card import test_thinking_budget_serving as serving_checks
from vllm_ascend.sample.thinking_budget import NPUThinkingBudgetStateHolder

START = 100
END = [40, 41, 42, 43, 44, 200]
VOCAB = 256


@pytest.mark.parametrize("num_spec", [0, 3])
@pytest.mark.parametrize("temperature", [0.0, 0.6])
def test_v1_end_sequence_on_npu(num_spec, temperature):
    holder = NPUThinkingBudgetStateHolder(
        SimpleNamespace(reasoning_start_token_ids=[START], reasoning_end_token_ids=END),
        3,
        num_spec,
        torch.device("npu"),
        False,
    )
    outputs = [[START, 7, 7], [START, 7, 7], [START, 7, 7]]
    holder.sync_batch(
        BatchUpdate(
            batch_size=3,
            removed=(),
            moved=(),
            added=[
                (i, SamplingParams(thinking_token_budget=budget), None, outputs[i])
                for i, budget in enumerate([2, None, 2])
            ],
        )
    )
    for step in range(len(END)):
        # First decode step has no drafts; later steps propose matching tokens.
        progress = len(outputs[0]) - 3
        proposed = (END[progress:] + [9] * num_spec)[:num_spec] if step else []
        drafts = [proposed, [9] * len(proposed), proposed]
        holder.update_state(outputs, drafts)
        bonus = torch.zeros(3, VOCAB, device="npu")
        bonus[:, 77] = 1
        holder.apply_to_logits(bonus, True, drafts)
        width = len(proposed)
        target = torch.zeros(3 * max(1, width), VOCAB, device="npu")
        target[:, 77] = 1
        original = target.clone()
        holder.apply_to_logits(target, False, drafts)
        plain_start = max(1, width)
        assert torch.equal(target[plain_start : 2 * plain_start], original[plain_start : 2 * plain_start])
        if temperature:
            sampled = torch.multinomial(torch.softmax(target / temperature, dim=-1), 1).flatten().tolist()
        else:
            sampled = target.argmax(-1).tolist()
        for request in [0, 2]:
            if width:
                for j, draft in enumerate(proposed):
                    token = sampled[request * width + j]
                    outputs[request].append(token)
                    if token != draft:
                        break
                else:
                    outputs[request].append(int(bonus[request].argmax()))
            else:
                outputs[request].append(sampled[request])
        if all(END[-1] in outputs[i] for i in [0, 2]):
            break
    for i in [0, 2]:
        assert outputs[i][3 : 3 + len(END)] == END
    torch.npu.synchronize()


@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("speculative", [False, True])
def test_v1_model_mixed_budget(monkeypatch, enforce_eager, speculative):
    # This test deliberately selects V1; the existing model matrix selects V2.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    end_text = "Time to answer. </think>"
    spec = (
        {
            "method": "ngram_gpu",
            "num_speculative_tokens": 3,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 5,
        }
        if speculative
        else None
    )
    with VllmRunner(
        "Qwen/Qwen3-1.7B",
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        kv_cache_memory_bytes=2 * 1024**3,
        enable_prefix_caching=False,
        enforce_eager=enforce_eager,
        speculative_config=spec,
        reasoning_config={"reasoning_parser": "qwen3", "reasoning_start_str": "<think>", "reasoning_end_str": end_text},
    ) as runner:
        tok = runner.model.get_tokenizer()
        start = tok.encode("<think>", add_special_tokens=False)
        end = tok.encode(end_text, add_special_tokens=False)
        assert len(end) > 1
        prompt = {"prompt_token_ids": [10] * 32 + start}
        budgets = [0, None, 1, 8]
        for temperature in [0.0, 0.6]:
            params = [
                SamplingParams(
                    temperature=temperature,
                    seed=42,
                    max_tokens=32,
                    ignore_eos=True,
                    thinking_token_budget=budget,
                    allowed_token_ids=[10],
                )
                for budget in budgets
            ]
            outputs = runner.model.generate([prompt] * len(params), params)
            for budget, output in zip(budgets, outputs):
                expected = [10] * 32 if budget is None else [10] * budget + end + [10] * (32 - budget - len(end))
                assert list(output.outputs[0].token_ids) == expected
            # Reverse request order to exercise reuse and different budget rows.
            outputs = runner.model.generate([prompt] * len(params), params[::-1])
            for budget, output in zip(budgets[::-1], outputs):
                expected = [10] * 32 if budget is None else [10] * budget + end + [10] * (32 - budget - len(end))
                assert list(output.outputs[0].token_ids) == expected


@pytest.mark.parametrize("progress", [1, 3, 5])
def test_v1_resumed_end_prefix_on_npu(progress):
    holder = NPUThinkingBudgetStateHolder(
        SimpleNamespace(reasoning_start_token_ids=[START], reasoning_end_token_ids=END),
        2,
        3,
        torch.device("npu"),
        False,
    )
    output = [7, 7] + END[:progress]
    holder.sync_batch(
        BatchUpdate(
            batch_size=1, removed=(), moved=(), added=[(0, SamplingParams(thinking_token_budget=2), [START], output)]
        )
    )
    holder.update_state([output], [[]])
    logits = holder.apply_to_logits(torch.zeros(1, VOCAB, device="npu"), False, [[]])
    assert int(logits[0].argmax()) == END[progress]


@pytest.mark.parametrize("speculative", [False, True])
def test_v1_natural_generation(monkeypatch, speculative):
    monkeypatch.setattr(generation_checks, "BUDGETS", [None, -1, 0, 1, 16, 64])
    generation_checks.test_thinking_budget_mixed_generation(
        monkeypatch, "Qwen/Qwen3-1.7B", True, speculative, runner_version="0"
    )


@pytest.mark.parametrize("speculative", [False, True])
def test_v1_natural_serving(monkeypatch, speculative, tmp_path):
    monkeypatch.setattr(serving_checks, "BUDGETS", [None, -1, 0, 1, 16, 64])
    serving_checks.test_thinking_budget_serving(False, speculative, tmp_path, runner_version="0")
