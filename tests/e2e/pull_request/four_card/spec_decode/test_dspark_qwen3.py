#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""DSpark draft acceptance on TP4.

The DSpark markov head (markov_w1 VocabParallelEmbedding + markov_w2
ParallelLMHead) is replicated on every TP rank via the ReplicatedGroup
instead of being TP-sharded, so the per-draft-step TP all-reduce and
all-gather disappear. The one-card TP1 case
(tests/e2e/pull_request/one_card/spec_decode/test_dspark.py) already
exercises the routing at tp_size==1; this TP4 case pins that holding the
full markov tables on all four ranks keeps draft acceptance at the TP1
level.

Run `pytest tests/e2e/pull_request/four_card/spec_decode/test_dspark_qwen3.py`.
"""

import os
from typing import Any

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = ["Qwen/Qwen3-8B"]
# Same models as the one-card TP1 dspark case
# (tests/e2e/pull_request/one_card/spec_decode/utils.py).
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_8b_block7"
NUM_SPECULATIVE_TOKENS = 7
# Same acceptance baseline as the one-card TP1 dspark case.
GOLDEN = [1.0, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6]


def acceptance_per_pos(metrics: list[Any], num_speculative_tokens: int) -> list[float]:
    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                num_accepted_tokens_per_pos[pos] += metric.values[pos]
    return [num_accepted / num_drafts for num_accepted in num_accepted_tokens_per_pos]


@pytest.mark.parametrize("model_name", MODELS)
def test_dspark_acceptance_tp4(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    prompts = [{"role": "user", "content": "Hello, your name is"}]
    prompts = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]

    speculative_config = {
        "method": "dspark",
        "model": DRAFT_MODEL,
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
    }

    compilation_config = CompilationConfig(cudagraph_mode="PIECEWISE", cudagraph_capture_sizes=[7, 8])

    with VllmRunner(
        model_name,
        max_model_len=4096,
        disable_log_stats=False,
        tensor_parallel_size=4,
        max_num_seqs=256,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        enable_prefix_caching=False,
    ) as llm:
        outputs = llm.model.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        output_tokens = output.outputs[0].token_ids
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print(f"Output tokens: {output_tokens}")

    acceptance = acceptance_per_pos(metrics, NUM_SPECULATIVE_TOKENS)

    match = all(abs(a - b) < 0.1 for a, b in zip(acceptance, GOLDEN))
    assert match, f"acceptance_per_pos {acceptance} does not match golden {GOLDEN}"
    cleanup_dist_env_and_memory()
