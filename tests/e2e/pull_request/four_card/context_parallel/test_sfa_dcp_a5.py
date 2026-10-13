# SPDX-License-Identifier: Apache-2.0
"""Ascend950 SFA DCP regression with real weights and BF16 KV.

Keep one GLM-5.2 layer to fit the four-card PR runner. This checks the
prefill/decode kernel path, not full-model task accuracy or SFA C8.
"""

import math
import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from vllm_ascend.device.device_config import is_950

MODEL = "Eco-Tech/GLM-5.2-w4a8"
TP_SIZE = 4
BLOCK_SIZE = 128
MAX_TOKENS = 8
PROMPT_LENGTHS = (1, 127, 128, 129, 511, 512, 513, 2049)

pytestmark = [pytest.mark.e2e_model(MODEL), pytest.mark.skipif(not is_950(), reason="Ascend950 only")]


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dcp,sfa",
    parallel="TP,EP,DCP",
    deploy="pd_mix",
    hardware="A5",
    quantization="W4A8",
    graph_mode="eager",
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "0", "HCCL_BUFFSIZE": "768"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_glm52_sfa_dcp_bf16_kv():
    """Exercise empty local shards, padded top-k indices and DCP LSE merge."""
    first_token_logprobs = []
    for dcp_size in (1, TP_SIZE):
        with VllmRunner(
            maybe_model_redirect(MODEL),
            dtype="bfloat16",
            quantization="ascend",
            hf_overrides={"num_hidden_layers": 1, "num_nextn_predict_layers": 0},
            tensor_parallel_size=TP_SIZE,
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=1,
            enable_expert_parallel=True,
            kv_cache_dtype="auto",
            block_size=BLOCK_SIZE,
            cp_kv_cache_interleave_size=BLOCK_SIZE,
            enforce_eager=True,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            distributed_executor_backend="mp",
            max_model_len=4096,
            max_num_batched_tokens=4096,
            max_num_seqs=1,
            gpu_memory_utilization=0.8,
            seed=1024,
            additional_config={"enable_sparse_sfa_c8": False},
        ) as runner:
            tokenizer = runner.model.get_tokenizer()
            tokens = tokenizer.encode("Explain how computers store information. ", add_special_tokens=False)
            assert tokens
            mode_logprobs = []
            for length in PROMPT_LENGTHS:
                prompt = (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]
                (output,) = runner.model.generate(
                    [{"prompt_token_ids": prompt}],
                    SamplingParams(temperature=0, ignore_eos=True, max_tokens=MAX_TOKENS, logprobs=20),
                    use_tqdm=False,
                )
                assert output.finished
                assert len(output.outputs) == 1
                completion = output.outputs[0]
                assert len(completion.token_ids) == MAX_TOKENS
                assert completion.logprobs is not None
                assert len(completion.logprobs) == MAX_TOKENS
                for step in completion.logprobs:
                    assert step and all(math.isfinite(value.logprob) for value in step.values())
                mode_logprobs.append((completion.token_ids[0], completion.logprobs[0]))
            first_token_logprobs.append(mode_logprobs)

    # Compare the same first-token candidate before greedy trajectories diverge.
    # BF16 reduction order can change close rankings, so do not require exact text.
    for length, (token_id, baseline), (_, dcp) in zip(PROMPT_LENGTHS, first_token_logprobs[0], first_token_logprobs[1]):
        assert token_id in dcp, f"DCP lost the baseline top token at prompt length {length}"
        assert math.isclose(baseline[token_id].logprob, dcp[token_id].logprob, rel_tol=0.01, abs_tol=0.1), (
            f"DCP logprob mismatch at prompt length {length}: {baseline[token_id]} vs {dcp[token_id]}"
        )
