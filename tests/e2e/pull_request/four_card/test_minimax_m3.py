#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

from __future__ import annotations

import math
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MINIMAX_M3_MODEL_PATH = os.environ.get("MINIMAX_M3_MODEL_PATH", "Eco-Tech/MiniMax-M3-w8a8-0626")
GSM8K_QUESTION = "Ali had $21. Leila gave him half of her $100. How much does Ali have now?"
GSM8K_PROMPT_TEMPLATE = (
    'Answer the following question. The last line of the response should follow this format: "answer:$ANSWER" '
    "(without quotes), where ANSWER is a number. Let's think step by step.\n\nQuestion: {question}"
)

NUM_HIDDEN_LAYERS = 5
MAX_TOKENS = 8
NUM_LOGPROBS = 5
LOGPROB_ATOL = 1e-5

os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _configure_jemalloc() -> None:
    jemalloc_path = "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
    if Path(jemalloc_path).exists():
        ld_preload = os.environ.get("LD_PRELOAD", "")
        os.environ["LD_PRELOAD"] = f"{jemalloc_path}:{ld_preload}" if ld_preload else jemalloc_path


def _apply_minimax_chat_template(tokenizer) -> str:
    prompt = GSM8K_PROMPT_TEMPLATE.format(question=GSM8K_QUESTION)
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        thinking_mode="disabled",
    )

    assert isinstance(rendered_prompt, str)
    assert tokenizer.eos_token is not None
    assert tokenizer.eos_token in rendered_prompt, (
        "MiniMax-M3 chat template did not delimit the user turn with its EOS token"
    )
    return rendered_prompt


def _assert_logprobs_match(baseline, replay) -> None:
    baseline_ids, _, baseline_logprobs = baseline
    replay_ids, _, replay_logprobs = replay

    assert baseline_ids, "MiniMax-M3 produced no output tokens"
    assert baseline_ids == replay_ids
    assert baseline_logprobs is not None
    assert replay_logprobs is not None
    assert len(baseline_logprobs) == len(replay_logprobs) == len(baseline_ids)

    for step, (token_id, baseline_topk, replay_topk) in enumerate(
        zip(baseline_ids, baseline_logprobs, replay_logprobs)
    ):
        assert token_id in baseline_topk
        assert token_id in replay_topk
        assert baseline_topk.keys() == replay_topk.keys(), f"top-k token mismatch at decode step {step}"

        for candidate_id in baseline_topk:
            baseline_value = baseline_topk[candidate_id].logprob
            replay_value = replay_topk[candidate_id].logprob
            assert math.isfinite(baseline_value), f"non-finite baseline logprob at decode step {step}"
            assert math.isfinite(replay_value), f"non-finite replay logprob at decode step {step}"
            assert replay_value == pytest.approx(baseline_value, abs=LOGPROB_ATOL), (
                f"logprob mismatch at decode step {step} for token {candidate_id}: "
                f"baseline={baseline_value}, replay={replay_value}"
            )


@pytest.mark.e2e_model(str(MINIMAX_M3_MODEL_PATH))
@pytest.mark.e2e_coverage(
    arch="multimodal",
    feature="aclgraph,logprobs",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_decode_only",
)
@patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3"})
@wait_until_npu_memory_free()
def test_minimax_m3_tp4_dummy_logprobs() -> None:
    _configure_jemalloc()

    with VllmRunner(
        MINIMAX_M3_MODEL_PATH,
        seed=0,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        dtype="auto",
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.95,
        quantization="ascend",
        load_format="dummy",
        hf_overrides={"text_config": {"num_hidden_layers": NUM_HIDDEN_LAYERS}},
        language_model_only=True,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4],
        },
        additional_config={
            "enable_cpu_binding": True,
            "ascend_compilation_config": {
                "enable_static_kernel": True,
                "fuse_norm_quant": False,
            },
            "multistream_overlap_shared_expert": False,
            "weight_nz_mode": 2,
            "enable_shared_expert_dp": True,
        },
    ) as vllm_model:
        tokenizer = vllm_model.model.get_tokenizer()
        prompt = _apply_minimax_chat_template(tokenizer)

        baseline = vllm_model.generate_greedy_logprobs([prompt], max_tokens=MAX_TOKENS, num_logprobs=NUM_LOGPROBS)[0]
        replay = vllm_model.generate_greedy_logprobs([prompt], max_tokens=MAX_TOKENS, num_logprobs=NUM_LOGPROBS)[0]

    _assert_logprobs_match(baseline, replay)
