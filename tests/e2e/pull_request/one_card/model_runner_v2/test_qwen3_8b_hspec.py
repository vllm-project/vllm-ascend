# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
# ruff: noqa: E402

import os
from unittest.mock import patch

import pytest

pytest.importorskip(
    "vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.speculator",
    reason="Requires vLLM H-Spec support (PR #62780)",
)

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free


@pytest.mark.parametrize("enforce_eager", [True, False])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_hspec_greedy_tokens_match_target(enforce_eager):
    prompts = ["The capital of France is", "Explain why the sky is blue.", "Summarize this: " + "The cat sat. " * 100]
    params = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True)
    config = dict(
        max_model_len=1024,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        enforce_eager=enforce_eager,
        async_scheduling=True,
    )

    with VllmRunner("Qwen/Qwen3-8B", **config) as runner:
        outputs = runner.model.generate(prompts, params)
        expected = [output.outputs[0].token_ids for output in outputs]

    with VllmRunner(
        "Qwen/Qwen3-8B",
        **config,
        speculative_config={
            "model": "weifanjiang/qwen3-8b.speculators.hspec",
            "method": "mamba_attn_hybrid",
            "num_speculative_tokens": 7,
        },
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [8, 16, 32]},
    ) as runner:
        # Reuse the engine with changing request counts to expose stale seed,
        # graph padding and rejected-suffix state across successive batches.
        for batch in (prompts, prompts[:1], prompts[:2]):
            outputs = runner.model.generate(batch, params)
            assert [output.outputs[0].token_ids for output in outputs] == expected[: len(batch)]
