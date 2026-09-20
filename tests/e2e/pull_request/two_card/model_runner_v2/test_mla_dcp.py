# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare MLA DCP against replicated KV, including prefix hits and graph padding."""

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = "deepseek-ai/DeepSeek-V2-Lite"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.parametrize("enforce_eager", [True, False])
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mla_dcp_matches_replicated_kv(monkeypatch, enforce_eager):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    prompts = [
        "The capital of France is",
        "A distributed system stores attention keys and values. " * 48 + "Explain:",
        "one two three four five six seven eight " * 32 + "Continue:",
    ]
    sampling = SamplingParams(temperature=0, seed=42, max_tokens=24, ignore_eos=True)
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
            # Three requests exercise a four-request capture. Repeating the
            # batch also exercises chunked prefill with existing cached KV.
            cold = runner.generate(prompts, sampling)
            warm = runner.generate(prompts, sampling)
            assert cold == warm
            outputs.append(cold)
    assert outputs[0] == outputs[1]
