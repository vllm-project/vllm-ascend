# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Storage-light Kimi K3 execution parity guard.

The committed fixture keeps Kimi K3's production dimensions and its mixed
KDA/MLA layout, but limits the model to five layers and sixteen experts. Dummy
weights deliberately make this an execution-parity test, not a semantic
accuracy test. Full-checkpoint GPQA remains a separate release gate.
"""

from pathlib import Path

import pytest
import torch
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL_CONFIG = Path(__file__).parent / "fixtures" / "kimi_k3_5layers_16experts"
SCHEDULER_BLOCK_SIZE = 16
PROMPT_TOKEN_IDS = [163584, *range(100, 100 + SCHEDULER_BLOCK_SIZE)]
MAX_TOKENS = 4


def _assert_complete_output(request_output):
    assert request_output is not None
    assert request_output.finished
    assert request_output.outputs is not None
    assert len(request_output.outputs) == 1

    completion = request_output.outputs[0]
    assert completion is not None
    assert completion.token_ids is not None
    assert len(completion.token_ids) == MAX_TOKENS
    assert completion.logprobs is not None
    assert len(completion.logprobs) == MAX_TOKENS

    chosen_logprobs = []
    for token_id, step_logprobs in zip(completion.token_ids, completion.logprobs):
        assert step_logprobs is not None
        assert token_id in step_logprobs
        logprob = step_logprobs[token_id].logprob
        assert logprob is not None
        assert torch.isfinite(torch.tensor(logprob))
        chosen_logprobs.append(logprob)

    return list(completion.token_ids), torch.tensor(chosen_logprobs, dtype=torch.float32)


@pytest.mark.e2e_model("sgl-npu/Kimi-K3-W4A8")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="aclgraph,prefix_caching,logprobs",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_kimi_k3_dummy_prefix_cache_one_token_prefill_parity():
    """Compare cold prefill with the cached block-size-plus-one path."""
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=MAX_TOKENS,
        logprobs=1,
        ignore_eos=True,
        seed=0,
    )
    prompt = TokensPrompt(prompt_token_ids=PROMPT_TOKEN_IDS)

    with VllmRunner(
        str(MODEL_CONFIG),
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        seed=0,
        block_size=SCHEDULER_BLOCK_SIZE,
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        tensor_parallel_size=16,
        enable_expert_parallel=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.75,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
        },
    ) as vllm_model:
        cold = vllm_model.model.generate([prompt], sampling_params, use_tqdm=False)[0]
        hit = vllm_model.model.generate([prompt], sampling_params, use_tqdm=False)[0]

        assert cold.num_cached_tokens in (None, 0)
        assert hit.num_cached_tokens == SCHEDULER_BLOCK_SIZE
        assert len(PROMPT_TOKEN_IDS) - hit.num_cached_tokens == 1

        cold_tokens, cold_logprobs = _assert_complete_output(cold)
        hit_tokens, hit_logprobs = _assert_complete_output(hit)
        assert hit_tokens == cold_tokens
        torch.testing.assert_close(hit_logprobs, cold_logprobs, rtol=5e-3, atol=5e-3)

        assert vllm_model.model.reset_prefix_cache()
        reset = vllm_model.model.generate([prompt], sampling_params, use_tqdm=False)[0]
        assert reset.num_cached_tokens in (None, 0)
        reset_tokens, reset_logprobs = _assert_complete_output(reset)
        assert reset_tokens == cold_tokens
        torch.testing.assert_close(reset_logprobs, cold_logprobs, rtol=5e-3, atol=5e-3)
