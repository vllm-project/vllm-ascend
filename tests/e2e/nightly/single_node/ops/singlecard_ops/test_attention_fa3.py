"""
End-to-end test for Flash Attention 3 (FA3) on Ascend.

This test verifies that FA3 produces correct results by comparing its output
with the default Fused Infer Attention (FIA) backend. Both backends are run
with the same model and prompts in eager mode, and the generated token ids
are compared for consistency.
"""

import os
import pytest
from importlib import import_module, util

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_MODEL_LEN = 512
MAX_TOKENS = 32

SHORT_PROMPTS = [
    "The capital of France is",
    "In a hole in the ground there lived a hobbit.",
    "The meaning of life is",
    "To be or not to be, that is the",
]

LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 50

def _fa3_available() -> bool:
    try:
        if util.find_spec("flash_attn_v3") is None:
            return False
        mod = import_module("flash_attn_v3")
        return hasattr(mod, "flash_attn_with_kvcache")
    except ImportError:
        return False


def _generate_with_backend(prompts, env_value, max_tokens=MAX_TOKENS, **runner_kwargs):
    original_value = os.environ.get("VLLM_ASCEND_ENABLE_FLASH_ATTN", "0")
    os.environ["VLLM_ASCEND_ENABLE_FLASH_ATTN"] = str(env_value)
    try:
        with VllmRunner(
            MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
            gpu_memory_utilization=0.7,
            **runner_kwargs,
        ) as runner:
            return runner.generate_greedy(prompts, max_tokens)
    finally:
        os.environ["VLLM_ASCEND_ENABLE_FLASH_ATTN"] = original_value


def _generate_logprobs_with_backend(prompts, env_value, max_tokens=5, num_logprobs=5):
    original_value = os.environ.get("VLLM_ASCEND_ENABLE_FLASH_ATTN", "0")
    os.environ["VLLM_ASCEND_ENABLE_FLASH_ATTN"] = str(env_value)
    try:
        with VllmRunner(
            MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
            gpu_memory_utilization=0.7,
        ) as runner:
            return runner.generate_greedy_logprobs(
                prompts, max_tokens=max_tokens, num_logprobs=num_logprobs
            )
    finally:
        os.environ["VLLM_ASCEND_ENABLE_FLASH_ATTN"] = original_value


def _assert_outputs_match(fia_outputs, fa3_outputs, label=""):
    for i, (fia_out, fa3_out) in enumerate(zip(fia_outputs, fa3_outputs)):
        fia_ids, fia_text = fia_out
        fa3_ids, fa3_text = fa3_out
        assert fia_ids == fa3_ids, (
            f"{label}Prompt {i}: FA3 and FIA token ids differ.\n"
            f"  FIA ids: {fia_ids}\n"
            f"  FA3 ids: {fa3_ids}\n"
            f"  FIA text: {fia_text}\n"
            f"  FA3 text: {fa3_text}"
        )

@pytest.mark.skipif(not _fa3_available(), reason="flash_attn_v3 is not installed")
def test_fa3_vs_fia_single_prompt():
    """Compare FA3 and FIA with a single prompt (minimal batch size).

    This is an edge case that verifies FA3 works correctly when there
    is only one sequence in the batch, which may exercise different
    code paths in the attention kernel.
    """
    single_prompt = ["Explain quantum computing in simple terms."]
    fia_outputs = _generate_with_backend(single_prompt, env_value=0)
    fa3_outputs = _generate_with_backend(single_prompt, env_value=1)
    _assert_outputs_match(fia_outputs, fa3_outputs, label="[SinglePrompt] ")


@pytest.mark.skipif(not _fa3_available(), reason="flash_attn_v3 is not installed")
def test_fa3_vs_fia_mixed_lengths():
    """Compare FA3 and FIA with mixed prompt lengths in the same batch.

    This exercises both prefill and decode paths within a single batch,
    verifying that FA3 handles variable-length sequences correctly.
    """
    mixed_prompts = [
        "Hi",
        "The capital of France is",
        LONG_PROMPT[:256],
        "What is 2+2?",
        LONG_PROMPT[:MAX_MODEL_LEN],
    ]
    fia_outputs = _generate_with_backend(mixed_prompts, env_value=0)
    fa3_outputs = _generate_with_backend(mixed_prompts, env_value=1)
    _assert_outputs_match(fia_outputs, fa3_outputs, label="[MixedLen] ")


@pytest.mark.skipif(not _fa3_available(), reason="flash_attn_v3 is not installed")
def test_fa3_vs_fia_with_chunkprefill():
    """Compare FA3 and FIA with single token generation where chunkprefill is used.
    """
    fia_outputs = _generate_with_backend(SHORT_PROMPTS, env_value=0, max_tokens=2, max_num_seqs=2, max_num_batched_tokens=5)
    fa3_outputs = _generate_with_backend(SHORT_PROMPTS, env_value=1, max_tokens=2, max_num_seqs=2, max_num_batched_tokens=5)
    _assert_outputs_match(fia_outputs, fa3_outputs, label="[Chunkprefill] ")


@pytest.mark.skipif(not _fa3_available(), reason="flash_attn_v3 is not installed")
def test_fa3_vs_fia_logprobs():
    """Compare FA3 and FIA logprobs for fine-grained numerical verification."""
    fia_logprobs = _generate_logprobs_with_backend(SHORT_PROMPTS[:1], env_value=0)
    fa3_logprobs = _generate_logprobs_with_backend(SHORT_PROMPTS[:1], env_value=1)

    for i, (fia_out, fa3_out) in enumerate(zip(fia_logprobs, fa3_logprobs)):
        fia_ids, _, fia_lp = fia_out
        fa3_ids, _, fa3_lp = fa3_out

        assert fia_ids == fa3_ids, (
            f"Prompt {i}: FA3 and FIA token ids differ.\n"
            f"  FIA ids: {fia_ids}\n"
            f"  FA3 ids: {fa3_ids}"
        )

        assert len(fia_lp) == len(fa3_lp), (
            f"Prompt {i}: Different number of logprob steps: "
            f"FIA {len(fia_lp)} vs FA3 {len(fa3_lp)}"
        )

        for t, (fia_token_lp, fa3_token_lp) in enumerate(zip(fia_lp, fa3_lp)):
            assert set(fia_token_lp.keys()) == set(fa3_token_lp.keys()), (
                f"Prompt {i}, token {t}: Logprob token sets differ.\n"
                f"  FIA keys: {sorted(fia_token_lp.keys())}\n"
                f"  FA3 keys: {sorted(fa3_token_lp.keys())}"
            )
            for token_id in fia_token_lp:
                fia_logprob = fia_token_lp[token_id].logprob
                fa3_logprob = fa3_token_lp[token_id].logprob
                assert fia_logprob==fa3_logprob, (
                    f"Prompt {i}, token {t}, token_id {token_id} ("
                    f"'{fia_token_lp[token_id].decoded_token}'): "
                    f"logprobs differ: FIA {fia_logprob:.6f} vs FA3 {fa3_logprob:.6f}"
                )
