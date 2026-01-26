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
from tests.e2e.conftest import VllmRunner

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


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
        padding_text = (
            " This is an interesting topic that deserves more explanation. " *
            (target_words // 50))
        base_prompt = base_prompt + padding_text

    return base_prompt


def _extract_step_logprobs(generate_output):
    """
    extract logprobs and token IDs from VllmRunner.generate_w_logprobs()
    """
    if not isinstance(generate_output, tuple) or len(generate_output) < 3:
        return None, None

    output_ids, output_str, output_logprobs = generate_output[:3]

    if output_logprobs is None:
        return None, None

    logprobs_list = []
    for i, logprob_dict in enumerate(output_logprobs):
        if logprob_dict is not None:
            # logprob_dict is a dictionary where the keys are token_ids and the values are Logprob objects.
            token_id = output_ids[i]
            logprob = logprob_dict.get(token_id)
            if logprob is not None:
                logprobs_list.append(logprob.logprob)
            else:
                logprobs_list.append(0.0)
        else:
            logprobs_list.append(0.0)

    logprobs_tensor = torch.tensor(logprobs_list, dtype=torch.float32)
    return logprobs_tensor, output_ids


@pytest.mark.timeout(1000)
def test_aclgraph_v1_generation_is_deterministic_across_batch_sizes_with_needle(
        monkeypatch: pytest.MonkeyPatch):
    """
    Ensures that the same request (the 'needle' prompt) yields identical output
    whether run alone (bs=1) or mixed into a larger batch (e.g., bs=64),
    using the high-level v1 LLM() API only (no manual batching).

    Strategy:
    - Create two LLM engines with identical config except max_num_seqs: 1 vs N.
    - Compute a baseline output for the needle prompt with the bs=1 engine.
    - For many trials, generate a batch (size N) where the needle appears at a
      random position among random filler prompts using the bs=N engine.
    - Track how many trials match vs mismatch, and report totals at the end.
      The test fails if any mismatches occur, but we still dump pass/fail
      counts.

    Notes:
    - Use seeded stochastic sampling with a fixed seed to test determinism.
    - Outputs are intentionally longer and sampled at higher temperature/top_p
      to produce a more random-sounding phrase, yet remain deterministic by
      seed.
    - Keep max_tokens and max_model_len bounded for speed and memory use.
    """
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)

    # Allow overrides from environment (useful for CI tuning)
    model = DEFAULT_MODEL

    num_trials = int(os.getenv("VLLM_NEEDLE_TRIALS", "5"))
    max_batch_size = int(os.getenv("VLLM_NEEDLE_BATCH_SIZE", "64"))
    min_random_prompt = int(os.getenv("VLLM_MIN_PROMPT", "1024"))
    max_random_prompt = int(os.getenv("VLLM_MAX_PROMPT", "2048"))
    assert max_batch_size >= 2, "Batch size should be >= 2 to mix needle."

    # Keep GPU memory usage low to avoid startup allocation failures.
    gpu_mem_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.95"))
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "5120"))

    # Sampling parameters: longer outputs with a more random-sounding
    # continuation,but still deterministic due to fixed seed.
    temperature = float(os.getenv("VLLM_NEEDLE_TEMPERATURE", "0.0"))
    top_p = float(os.getenv("VLLM_NEEDLE_TOP_P", "0.95"))
    max_tokens = int(os.getenv("VLLM_NEEDLE_MAX_TOKENS", "35"))

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=20240919,
    )

    needle_prompt = "There once was a "

    with VllmRunner(
            model_name=model,
            max_num_seqs=max_batch_size,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            dtype="bfloat16",
            tensor_parallel_size=int(os.getenv("VLLM_TP_SIZE", "1")),
            enable_prefix_caching=False,
            distributed_executor_backend="mp",
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 32, 64]
            }
    ) as vllm_model:

        # Baseline generation for the needle prompt alone.
        baseline_out = vllm_model.generate([needle_prompt], sampling)
        assert len(baseline_out) == 1
        assert len(baseline_out[0][1]) >= 1
        baseline_text = baseline_out[0][1][0]

        mismatches = 0

        for trial in range(num_trials):
            # Create a batch of size `max_batch_size` and insert the needle at
            # a random index
            prompts: list[str] = []
            batch_size = random.randint(max_batch_size // 2, max_batch_size)
            needle_pos = random.randint(0, batch_size - 1)
            for i in range(batch_size):
                if i == needle_pos:
                    prompts.append(needle_prompt)
                else:
                    prompts.append(
                        _random_prompt(min_random_prompt, max_random_prompt))

            # Generate with the larger-batch engine
            outputs = vllm_model.generate(prompts, sampling)
            # Find the needle output by position
            needle_output = outputs[needle_pos][1]
            text = needle_output[0]

            if text != baseline_text:
                print(
                    f"{text}\n\n== Not the same as ==\n\n{baseline_text}\n\n")
                mismatches += 1

        passes = num_trials - mismatches
        # Dump how many passed vs failed
        print(f"[determinism] total={num_trials}, passed={passes}, "
              f"failed={mismatches}, max_batch_size={max_batch_size}")

        if mismatches > 0:
            pytest.fail(
                f"Nondeterministic outputs detected: {mismatches} failed out "
                f"of {num_trials} trials (max_batch_size={max_batch_size}).")



def test_aclgraph_logprobs_bitwise_batch_invariance_bs1_vs_bsN(
        monkeypatch: pytest.MonkeyPatch):
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)
    model_name = DEFAULT_MODEL
    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))

    # For batch invariance, disable custom all-reduce to ensure deterministic
    # all-reduce operations (custom all-reduce may not be deterministic)
    from vllm_ascend.batch_invariant import vllm_is_batch_invariant

    disable_custom_ar = vllm_is_batch_invariant()

    if disable_custom_ar:
        print(f"\n{'=' * 80}")
        print(
            f"BATCH INVARIANCE MODE: Disabling custom all-reduce (TP={tp_size})"
        )
        print(f"{'=' * 80}\n")

    with VllmRunner(
            model_name=model_name,
            tensor_parallel_size=tp_size,
            enable_prefix_caching=False,
            max_num_seqs=32,
            max_model_len=8192,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            distributed_executor_backend="mp",
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 32, 64]
            }
    ) as vllm_model:
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
            print(
                f"\n[BS=1] Running prompt {idx}/{len(prompts)} - Preview: {p[:80]}..."
            )
            outs = vllm_model.generate_w_logprobs([p], sp, use_tqdm=False)
            assert len(outs) == 1
            # print(outs)
            step_logprobs, token_ids = _extract_step_logprobs(outs[0])
            if step_logprobs is None:
                pytest.skip("Logits are not available on RequestOutput; "
                            "enable logprobs return to run this test.")
            bs1_logprobs_per_prompt.append(step_logprobs)
            bs1_tokens_per_prompt.append(token_ids)
            print(f"[BS=1] Prompt {idx} generated tokens: {token_ids}")

        # BS=N: run prompts in a batch and collect logprobs per step for each
        # prompt.
        print("\n" + "=" * 80)
        print(f"STARTING BS={len(prompts)} RUN (all prompts batched)")
        print("=" * 80 + "\n")

        outs_batched = vllm_model.generate_w_logprobs(prompts, sp, use_tqdm=False)
        assert len(outs_batched) == len(prompts)
        bsN_logprobs_per_prompt = []
        bsN_tokens_per_prompt = []

        print(f"\n[BS={len(prompts)}] Processing batched outputs...")
        for idx, o in enumerate(outs_batched):
            tokens = o[0]
            print(f"[BS={len(prompts)}] Prompt {idx} generated tokens: {tokens}")
            step_logprobs, token_ids = _extract_step_logprobs(o)
            if step_logprobs is None:
                pytest.skip("Logits are not available on RequestOutput; "
                            "enable logprobs return to run this test.")
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
                )):
            if len(logprobs_bs1) != len(logprobs_bsN):
                reason = (f"Different number of steps: {len(logprobs_bs1)} (BS=1) "
                          f"vs {len(logprobs_bsN)} (BS=N)")
                failed_prompts.append({
                    "prompt_idx": i,
                    "step": "all",
                    "reason": reason,
                    "prompt_preview": prompts[i][:100],
                    "bs1_tokens": tokens_bs1,
                    "bsN_tokens": tokens_bsN,
                })
                continue

            # Check if tokens match first
            if tokens_bs1 != tokens_bsN:
                failed_prompts.append({
                    "prompt_idx":
                    i,
                    "step":
                    "sampling",
                    "reason":
                    "Different tokens sampled",
                    "prompt_preview":
                    prompts[i][:100],
                    "bs1_tokens":
                    tokens_bs1,
                    "bsN_tokens":
                    tokens_bsN,
                    "bs1_all_logprobs":
                    [logprobs_bs1[s].tolist() for s in range(len(logprobs_bs1))],
                    "bsN_all_logprobs":
                    [logprobs_bsN[s].tolist() for s in range(len(logprobs_bsN))],
                })
                continue

            for t, (a, b) in enumerate(zip(logprobs_bs1, logprobs_bsN)):
                if a.shape != b.shape:
                    failed_prompts.append({
                        "prompt_idx": i,
                        "step": t,
                        "reason": f"Shape mismatch: {a.shape} vs {b.shape}",
                        "prompt_preview": prompts[i][:100],
                        "bs1_tokens": tokens_bs1,
                        "bsN_tokens": tokens_bsN,
                    })
                    break

                if not torch.equal(a, b):
                    max_diff = torch.abs(a - b).max().item()
                    # Print which token failed
                    print(
                        f"\n[DIVERGENCE] Prompt {i}, Token {t}: max_diff={max_diff:.6e}"
                    )
                    bs1_tok = tokens_bs1[t] if t < len(tokens_bs1) else "N/A"
                    bsN_tok = tokens_bsN[t] if t < len(tokens_bsN) else "N/A"
                    print(f"  Token IDs: bs1={bs1_tok}, bsN={bsN_tok}")
                    print(f"  BS=1 logprob: {a.tolist()}")
                    print(f"  BS=N logprob: {b.tolist()}")
                    failed_prompts.append({
                        "prompt_idx":
                        i,
                        "step":
                        t,
                        "reason":
                        f"Bitwise mismatch (max_diff={max_diff:.6e})",
                        "prompt_preview":
                        prompts[i][:100],
                        "bs1_tokens":
                        tokens_bs1,
                        "bsN_tokens":
                        tokens_bsN,
                        "bs1_all_logprobs": [
                            logprobs_bs1[s].tolist()
                            for s in range(len(logprobs_bs1))
                        ],
                        "bsN_all_logprobs": [
                            logprobs_bsN[s].tolist()
                            for s in range(len(logprobs_bsN))
                        ],
                    })
                    break


        # Print summary of all failures
    if failed_prompts:
        print(f"\n{'=' * 80}")
        fail_msg = (f"BATCH INVARIANCE FAILURES: {len(failed_prompts)}/"
                    f"{len(prompts)} prompts failed")
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
                print(
                    f"  BS=1 logprobs for all {len(fail['bs1_all_logprobs'])} steps:"
                )
                for step_idx, logprobs in enumerate(fail["bs1_all_logprobs"]):
                    print(f"    Step {step_idx}: {logprobs}")
                print(
                    f"  BS=N logprobs for all {len(fail['bsN_all_logprobs'])} steps:"
                )
                for step_idx, logprobs in enumerate(fail["bsN_all_logprobs"]):
                    print(f"    Step {step_idx}: {logprobs}")
        print(f"{'=' * 80}\n")

        # Fail the test with summary
        msg = (f"Batch invariance violated in {len(failed_prompts)}/"
               f"{len(prompts)} prompts. See output above for details.")
        pytest.fail(msg)


def test_aclgraph_simple_generation(monkeypatch: pytest.MonkeyPatch):
    """
    Simple test that runs the model with a basic prompt and prints the output.
    Useful for quick smoke testing and debugging.
    """
    model = DEFAULT_MODEL

    with VllmRunner(
            model_name=model,
            max_num_seqs=1,
            tensor_parallel_size=int(os.getenv("VLLM_TP_SIZE", "1")),
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            dtype="float16",
            enable_prefix_caching=False,
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 32, 64]
            },
            distributed_executor_backend="mp",
    ) as vllm_model:
        prompt = "The capital of France is"
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=20,
        )

        print(f"\n{'=' * 80}")
        print("Running simple generation test")
        print(f"Prompt: '{prompt}'")
        print(f"{'=' * 80}\n")

        outputs = vllm_model.generate([prompt], sampling_params)

        assert len(outputs) == 1
        output_text = outputs[0][1][0]

        print(f"Full completion: '{output_text}'")
        print(f"{'=' * 80}\n")





def test_aclgraph_logprobs_without_batch_invariance_should_fail(
        monkeypatch: pytest.MonkeyPatch):
    """
    This test is the inverse of test_logprobs_bitwise_batch_invariance_bs1_vs_bsN.
    It DISABLES batch invariance mode and expects to see non-deterministic behavior
    between BS=1 and BS=N runs. This demonstrates that batch invariance is actually
    doing something useful.

    The test will PASS if we detect differences (proving batch invariance matters).
    The test will FAIL if everything matches (suggesting batch invariance isn't needed).
    """
    # CRITICAL: Disable batch invariance for this test
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)
    model_name = DEFAULT_MODEL
    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))

    print(f"\n{'=' * 80}")
    print("BATCH INVARIANCE DISABLED: Expecting non-deterministic behavior")
    print(f"{'=' * 80}\n")

    with VllmRunner(
            model_name=model_name,
            tensor_parallel_size=tp_size,
            enable_prefix_caching=False,
            max_num_seqs=32,
            max_model_len=8192,
            dtype="bfloat16",
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 32, 64]
            },
            distributed_executor_backend="mp",
    ) as vllm_model:

        # build ragged prompts to change shapes significantly across BS=1 vs BS=N
        long_min = int(os.getenv("VLLM_MIN_PROMPT", "768"))
        long_max = int(os.getenv("VLLM_MAX_PROMPT", "2048"))
        prompts: list[str] = []
        options = [
            (max(long_min, 1536), max(long_max, 3072)),  # very long
            (max(1024, long_min), max(2048, long_max)),  # long
            (256, 512),  # mid
            (10, 20),  # short
        ]

        for _ in range(32):
            lo, hi = random.choice(options)
            prompts.append(_random_prompt(lo, hi))

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
            print(
                f"\n[BS=1] Running prompt {idx}/{len(prompts)} - Preview: {p[:80]}..."
            )
            outs = vllm_model.generate_w_logprobs([p], sp, use_tqdm=False)

            assert len(outs) == 1
            step_logprobs, token_ids = _extract_step_logprobs(outs[0])
            if step_logprobs is None:
                pytest.skip("Logits are not available on RequestOutput; "
                            "enable logprobs return to run this test.")
            bs1_logprobs_per_prompt.append(step_logprobs)
            bs1_tokens_per_prompt.append(token_ids)
            print(f"[BS=1] Prompt {idx} generated tokens: {token_ids}")

        # BS=N: run prompts in a batch and collect logprobs per step for each prompt.
        print("\n" + "=" * 80)
        print(f"STARTING BS={len(prompts)} RUN (all prompts batched)")
        print("=" * 80 + "\n")

        outs_batched = vllm_model.generate_w_logprobs(prompts, sp, use_tqdm=False)
        assert len(outs_batched) == len(prompts)
        bsN_logprobs_per_prompt = []
        bsN_tokens_per_prompt = []

        print(f"\n[BS={len(prompts)}] Processing batched outputs...")
        for idx, o in enumerate(outs_batched):
            tokens = o[0]
            print(f"[BS={len(prompts)}] Prompt {idx} generated tokens: {tokens}")
            step_logprobs, token_ids = _extract_step_logprobs(o)
            if step_logprobs is None:
                pytest.skip("Logits are not available on RequestOutput; "
                            "enable logprobs return to run this test.")
            bsN_logprobs_per_prompt.append(step_logprobs)
            bsN_tokens_per_prompt.append(token_ids)

        # Compare step-by-step logprobs for each prompt between BS=1 and BS=N runs.
        differences_found = []
        for i, (logprobs_bs1, logprobs_bsN, tokens_bs1, tokens_bsN) in enumerate(
                zip(
                    bs1_logprobs_per_prompt,
                    bsN_logprobs_per_prompt,
                    bs1_tokens_per_prompt,
                    bsN_tokens_per_prompt,
                )):
            if len(logprobs_bs1) != len(logprobs_bsN):
                reason = (f"Different number of steps: {len(logprobs_bs1)} (BS=1) "
                          f"vs {len(logprobs_bsN)} (BS=N)")
                differences_found.append({
                    "prompt_idx": i,
                    "step": "all",
                    "reason": reason,
                    "prompt_preview": prompts[i][:100],
                    "bs1_tokens": tokens_bs1,
                    "bsN_tokens": tokens_bsN,
                })
                continue

            # Check if tokens match first
            if tokens_bs1 != tokens_bsN:
                differences_found.append({
                    "prompt_idx": i,
                    "step": "sampling",
                    "reason": "Different tokens sampled",
                    "prompt_preview": prompts[i][:100],
                    "bs1_tokens": tokens_bs1,
                    "bsN_tokens": tokens_bsN,
                })
                continue

            for t, (a, b) in enumerate(zip(logprobs_bs1, logprobs_bsN)):
                if a.shape != b.shape:
                    differences_found.append({
                        "prompt_idx": i,
                        "step": t,
                        "reason": f"Shape mismatch: {a.shape} vs {b.shape}",
                        "prompt_preview": prompts[i][:100],
                        "bs1_tokens": tokens_bs1,
                        "bsN_tokens": tokens_bsN,
                    })
                    break

                if not torch.equal(a, b):
                    max_diff = torch.abs(a - b).max().item()
                    print(f"\n[EXPECTED DIVERGENCE FOUND] Prompt {i}, "
                          f"Token {t}: max_diff={max_diff:.6e}")
                    bs1_tok = tokens_bs1[t] if t < len(tokens_bs1) else "N/A"
                    bsN_tok = tokens_bsN[t] if t < len(tokens_bsN) else "N/A"
                    print(f"  Token IDs: bs1={bs1_tok}, bsN={bsN_tok}")
                    print(f"  BS=1 logprob: {a.tolist()}")
                    print(f"  BS=N logprob: {b.tolist()}")
                    differences_found.append({
                        "prompt_idx": i,
                        "step": t,
                        "reason": f"Bitwise mismatch (max_diff={max_diff:.6e})",
                        "prompt_preview": prompts[i][:100],
                        "bs1_tokens": tokens_bs1,
                        "bsN_tokens": tokens_bsN,
                    })
                    break


    # Print summary
    print(f"\n{'=' * 80}")
    if differences_found:
        success_msg = (
            f"✓ SUCCESS: Batch invariance is doing something! "
            f"Found {len(differences_found)}/{len(prompts)} prompts "
            f"with differences when batch invariance was DISABLED.")
        print(success_msg)
        print(f"{'=' * 80}")
        for diff in differences_found:
            print(f"\nPrompt {diff['prompt_idx']} (step {diff['step']}):")
            print(f"  Reason: {diff['reason']}")
            print(f"  Preview: {diff['prompt_preview']}...")
            if "bs1_tokens" in diff:
                print(f"  BS=1 tokens: {diff['bs1_tokens']}")
            if "bsN_tokens" in diff:
                print(f"  BS=N tokens: {diff['bsN_tokens']}")
        print(f"{'=' * 80}\n")
        # Test PASSES because we found differences (batch invariance matters!)
        return
    else:
        # Test FAILS because everything matched even without batch invariance
        fail_msg = (
            f"✗ UNEXPECTED: All {len(prompts)} prompts matched "
            f"between BS=1 and BS=N even with batch invariance DISABLED. "
            f"This suggests batch invariance might not be necessary, "
            f"or the test needs more sensitive prompts.")
        print(fail_msg)
        print(f"{'=' * 80}\n")
        pytest.fail(fail_msg)
