# Adapt from https://github.com/vllm-project/vllm/blob/main/tests/v1/determinism/test_batch_invariant.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import os
import random

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import ModelName

os.environ["VLLM_BATCH_INVARIANT"] = "1"

DEFAULT_MODEL = ModelName.QWEN3_30B_A3B

# Shrink Qwen3-30B-A3B so it fits in CI: only the first 2 of the 48 hidden
# layers are instantiated, keeping memory and runtime small. Weights are
# random-initialized via `--load-format dummy` (see the test marker below).
SMALL_QWEN3_OVERRIDES = {
    "num_hidden_layers": 2,
}


@pytest.fixture(autouse=True)
def enable_batch_invariant_mode(monkeypatch: pytest.MonkeyPatch):
    """Automatically enable batch invariant kernel overrides for all tests."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")


def _random_prompt(min_words: int = 1024, max_words: int = 1024 * 2) -> str:
    # Generate more realistic prompts that will actually produce varied tokens
    # Use a mix of common English text patterns

    prompt_templates = [
        # Question-answer style
        "Question: What is the capital of France?\nAnswer: The capital of France is",
        "Q: How does photosynthesis work?\nA: Photosynthesis is the process by which",
        "User: Can you explain quantum mechanics?\nAssistant: Quantum mechanics is",
        # Story/narrative style
        "Once upon a time in a distant galaxy, there lived",
        "The old man walked slowly down the street, remembering",
        "In the year 2157, humanity finally discovered",
        # Technical/code style
        "To implement a binary search tree in Python, first we need to",
        "The algorithm works by iterating through the array and",
        "Here's how to optimize database queries using indexing:",
        # Factual/informative style
        "The Renaissance was a period in European history that",
        "Climate change is caused by several factors including",
        "The human brain contains approximately 86 billion neurons which",
        # Conversational style
        "I've been thinking about getting a new laptop because",
        "Yesterday I went to the store and bought",
        "My favorite thing about summer is definitely",
    ]

    # Pick a random template
    base_prompt = random.choice(prompt_templates)

    if max_words < min_words:
        max_words = min_words
    target_words = random.randint(min_words, max_words)

    if target_words > 50:
        # For longer prompts, repeat context
        padding_text = " This is an interesting topic that deserves more explanation. " * (target_words // 50)
        base_prompt = base_prompt + padding_text

    return base_prompt


def _extract_step_logprobs(request_output):
    if getattr(request_output, "outputs", None):
        inner = request_output.outputs[0]
        if hasattr(inner, "logprobs") and inner.logprobs is not None:
            t = torch.tensor(
                [inner.logprobs[i][tid].logprob for i, tid in enumerate(inner.token_ids)],
                dtype=torch.float32,
            )
            return t, inner.token_ids

    return None, None


@pytest.mark.e2e_model("Qwen/Qwen3-30B-A3B")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="batch_invariant,logprobs",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="piecewise",
)
@pytest.mark.model(
    model_name=DEFAULT_MODEL,
    max_num_seqs=144,
    gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.95")),
    max_model_len=8192,
    dtype="bfloat16",
    tensor_parallel_size=int(os.getenv("VLLM_TEST_TP_SIZE", "4")),
    enable_prefix_caching=False,
    distributed_executor_backend="mp",
    compilation_config={
        "cudagraph_mode": "PIECEWISE",
        "cudagraph_capture_sizes": [1, 32, 64],
    },
    extra_kwargs={
        "load_format": "dummy",
        "hf_overrides": SMALL_QWEN3_OVERRIDES,
    },
)
def test_logprobs_bitwise_batch_invariance_bs1_vs_bsN(vllm_runner, monkeypatch: pytest.MonkeyPatch):
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)
    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "4"))

    # For batch invariance, disable custom all-reduce to ensure deterministic
    # all-reduce operations (custom all-reduce may not be deterministic)
    import vllm.envs as envs

    disable_custom_ar = envs.VLLM_BATCH_INVARIANT

    if disable_custom_ar:
        print(f"\n{'=' * 80}")
        print(f"BATCH INVARIANCE MODE: Disabling custom all-reduce (TP={tp_size})")
        print(f"{'=' * 80}\n")

    # Use more realistic prompts for better token generation
    prompts = [_random_prompt(10, 50) for i in range(32)]

    sp = SamplingParams(
        temperature=0.6,
        top_p=1.0,
        max_tokens=8,
        seed=1234,
        logprobs=5,
    )

    # BS=1: run prompts individually and collect logprobs per step.
    print("\n" + "=" * 80)
    print("STARTING BS=1 RUNS (each prompt individually)")
    print("=" * 80 + "\n")

    bs1_logprobs_per_prompt = []
    bs1_tokens_per_prompt = []
    for idx, p in enumerate(prompts):
        print(f"\n[BS=1] Running prompt {idx}/{len(prompts)} - Preview: {p[:80]}...")
        outs = vllm_runner.model.generate([p], sp, use_tqdm=False)
        assert len(outs) == 1
        step_logprobs, token_ids = _extract_step_logprobs(outs[0])
        if step_logprobs is None:
            pytest.skip("Logits are not available on RequestOutput; enable logprobs return to run this test.")
        bs1_logprobs_per_prompt.append(step_logprobs)
        bs1_tokens_per_prompt.append(token_ids)
        print(f"[BS=1] Prompt {idx} generated tokens: {token_ids}")

    # BS=N: run prompts in a batch and collect logprobs per step for each
    # prompt.
    print("\n" + "=" * 80)
    print(f"STARTING BS={len(prompts)} RUN (all prompts batched)")
    print("=" * 80 + "\n")

    outs_batched = vllm_runner.model.generate(prompts, sp, use_tqdm=False)
    assert len(outs_batched) == len(prompts)
    bsN_logprobs_per_prompt = []
    bsN_tokens_per_prompt = []

    print(f"\n[BS={len(prompts)}] Processing batched outputs...")
    for idx, o in enumerate(outs_batched):
        tokens = o.outputs[0].token_ids if o.outputs else "N/A"
        print(f"[BS={len(prompts)}] Prompt {idx} generated tokens: {tokens}")
        step_logprobs, token_ids = _extract_step_logprobs(o)
        if step_logprobs is None:
            pytest.skip("Logits are not available on RequestOutput; enable logprobs return to run this test.")
        bsN_logprobs_per_prompt.append(step_logprobs)
        bsN_tokens_per_prompt.append(token_ids)

    # Compare step-by-step logprobs for each prompt between BS=1 and BS=N runs.
    failed_prompts = []
    for i, (logprobs_bs1, logprobs_bsN, tokens_bs1, tokens_bsN) in enumerate(
        zip(
            bs1_logprobs_per_prompt,
            bsN_logprobs_per_prompt,
            bs1_tokens_per_prompt,
            bsN_tokens_per_prompt,
        )
    ):
        if len(logprobs_bs1) != len(logprobs_bsN):
            reason = f"Different number of steps: {len(logprobs_bs1)} (BS=1) vs {len(logprobs_bsN)} (BS=N)"
            failed_prompts.append(
                {
                    "prompt_idx": i,
                    "step": "all",
                    "reason": reason,
                    "prompt_preview": prompts[i][:100],
                    "bs1_tokens": tokens_bs1,
                    "bsN_tokens": tokens_bsN,
                }
            )
            continue

        # Check if tokens match first
        if tokens_bs1 != tokens_bsN:
            failed_prompts.append(
                {
                    "prompt_idx": i,
                    "step": "sampling",
                    "reason": "Different tokens sampled",
                    "prompt_preview": prompts[i][:100],
                    "bs1_tokens": tokens_bs1,
                    "bsN_tokens": tokens_bsN,
                    "bs1_all_logprobs": [logprobs_bs1[s].tolist() for s in range(len(logprobs_bs1))],
                    "bsN_all_logprobs": [logprobs_bsN[s].tolist() for s in range(len(logprobs_bsN))],
                }
            )
            continue

        for t, (a, b) in enumerate(zip(logprobs_bs1, logprobs_bsN)):
            if a.shape != b.shape:
                failed_prompts.append(
                    {
                        "prompt_idx": i,
                        "step": t,
                        "reason": f"Shape mismatch: {a.shape} vs {b.shape}",
                        "prompt_preview": prompts[i][:100],
                        "bs1_tokens": tokens_bs1,
                        "bsN_tokens": tokens_bsN,
                    }
                )
                break

            if not torch.equal(a, b):
                max_diff = torch.abs(a - b).max().item()
                # Print which token failed
                print(f"\n[DIVERGENCE] Prompt {i}, Token {t}: max_diff={max_diff:.6e}")
                bs1_tok = tokens_bs1[t] if t < len(tokens_bs1) else "N/A"
                bsN_tok = tokens_bsN[t] if t < len(tokens_bsN) else "N/A"
                print(f"  Token IDs: bs1={bs1_tok}, bsN={bsN_tok}")
                print(f"  BS=1 logprob: {a.tolist()}")
                print(f"  BS=N logprob: {b.tolist()}")
                failed_prompts.append(
                    {
                        "prompt_idx": i,
                        "step": t,
                        "reason": f"Bitwise mismatch (max_diff={max_diff:.6e})",
                        "prompt_preview": prompts[i][:100],
                        "bs1_tokens": tokens_bs1,
                        "bsN_tokens": tokens_bsN,
                        "bs1_all_logprobs": [logprobs_bs1[s].tolist() for s in range(len(logprobs_bs1))],
                        "bsN_all_logprobs": [logprobs_bsN[s].tolist() for s in range(len(logprobs_bsN))],
                    }
                )
                break

    # Print summary of all failures
    if failed_prompts:
        print(f"\n{'=' * 80}")
        fail_msg = f"BATCH INVARIANCE FAILURES: {len(failed_prompts)}/{len(prompts)} prompts failed"
        print(fail_msg)
        print(f"{'=' * 80}")
        for fail in failed_prompts:
            print(f"\nPrompt {fail['prompt_idx']} (step {fail['step']}):")
            print(f"  Reason: {fail['reason']}")
            print(f"  Preview: {fail['prompt_preview']}...")

            # Always show the tokens
            if "bs1_tokens" in fail:
                print(f"  BS=1 tokens: {fail['bs1_tokens']}")
            if "bsN_tokens" in fail:
                print(f"  BS=N tokens: {fail['bsN_tokens']}")

            if "bs1_all_logprobs" in fail:
                print(f"  BS=1 logprobs for all {len(fail['bs1_all_logprobs'])} steps:")
                for step_idx, logprobs in enumerate(fail["bs1_all_logprobs"]):
                    print(f"    Step {step_idx}: {logprobs}")
                print(f"  BS=N logprobs for all {len(fail['bsN_all_logprobs'])} steps:")
                for step_idx, logprobs in enumerate(fail["bsN_all_logprobs"]):
                    print(f"    Step {step_idx}: {logprobs}")
        print(f"{'=' * 80}\n")

        # Fail the test with summary
        msg = (
            f"Batch invariance violated in {len(failed_prompts)}/{len(prompts)} prompts. See output above for details."
        )
        pytest.fail(msg)
