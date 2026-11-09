# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare MLA DCP against replicated KV, including prefix hits and graph padding."""

import math

import pytest
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.utils import _check_decode_token, _check_prefill_token

MODEL = "deepseek-ai/DeepSeek-V2-Lite"
MAX_TOKENS = 24
PREFILL_ATOL = 0.0689
DECODE_ATOL = 2 * PREFILL_ATOL


def _check_outputs(baseline, comparison):
    assert len(baseline) == len(comparison) == 3
    for prompt_idx, (base_out, comp_out) in enumerate(zip(baseline, comparison)):
        assert base_out.prompt_token_ids == comp_out.prompt_token_ids
        assert base_out.num_cached_tokens == comp_out.num_cached_tokens
        base_seq, comp_seq = base_out.outputs[0], comp_out.outputs[0]
        for seq in (base_seq, comp_seq):
            assert len(seq.token_ids) == MAX_TOKENS
            assert seq.logprobs is not None and len(seq.logprobs) == MAX_TOKENS
            assert all(math.isfinite(probs[token].logprob) for token, probs in zip(seq.token_ids, seq.logprobs))
        _check_prefill_token(base_seq, comp_seq, prompt_idx, atol=PREFILL_ATOL)
        for token_idx in range(1, MAX_TOKENS):
            _check_decode_token(base_seq, comp_seq, token_idx, prompt_idx, decode_atol=DECODE_ATOL)
            _check_decode_token(comp_seq, base_seq, token_idx, prompt_idx, decode_atol=DECODE_ATOL)
            # Compare distributions through the first differing argmax. Later
            # tokens have different contexts and cannot be compared directly.
            if base_seq.token_ids[token_idx] != comp_seq.token_ids[token_idx]:
                break


@pytest.mark.e2e_model(MODEL)
@pytest.mark.parametrize("enforce_eager", [True, False])
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mla_dcp_matches_replicated_kv(monkeypatch, enforce_eager):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    prompt_texts = [
        "The capital of France is",
        "A distributed system stores attention keys and values. " * 96,
        "one two three four five six seven eight " * 128,
    ]
    sampling = SamplingParams(temperature=0, seed=42, max_tokens=MAX_TOKENS, ignore_eos=True, logprobs=20)
    outputs = []
    for dcp_size in (1, 2):
        with VllmRunner(
            MODEL,
            tensor_parallel_size=2,
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=128,
            block_size=128,
            max_model_len=2048,
            max_num_batched_tokens=128,
            max_num_seqs=4,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            enforce_eager=enforce_eager,
            seed=42,
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4]},
        ) as runner:
            tokenizer = runner.model.get_tokenizer()
            # Both DCP sizes can cache these prefixes in full: their scheduler
            # block spans are 128 and 256. Leave one token to process on a hit,
            # so the warm runs also exercise three-request graph padding.
            prompt_ids = [tokenizer.encode(text) for text in prompt_texts]
            assert len(prompt_ids[1]) >= 513 and len(prompt_ids[2]) >= 769
            prompt_ids[1] = prompt_ids[1][:513]
            prompt_ids[2] = prompt_ids[2][:769]
            prompts: list[TokensPrompt] = [{"prompt_token_ids": token_ids} for token_ids in prompt_ids]
            # Three requests exercise a four-request capture. Repeating the
            # batch also exercises chunked prefill with existing cached KV.
            cold = runner.model.generate(prompts, sampling)
            warm = runner.model.generate(prompts, sampling)
            assert [output.num_cached_tokens for output in cold] == [0, 0, 0]
            assert [output.num_cached_tokens for output in warm] == [0, 512, 768]
            outputs.append((cold, warm))
    # Compare DCP layouts with the same prefix-cache state and query lengths.
    # Cold versus warm also changes prefill execution and is not a DCP oracle.
    for baseline, comparison in zip(outputs[0], outputs[1]):
        _check_outputs(baseline, comparison)
