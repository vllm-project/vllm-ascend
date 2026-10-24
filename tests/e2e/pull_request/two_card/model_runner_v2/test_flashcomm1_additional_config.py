#
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

import os
from math import isfinite

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = os.environ.get("DEEPSEEK_V2_MODEL", "vllm-ascend/DeepSeek-V2-Lite-W8A8")
PROMPTS = [
    "Hello, what's your name?",
    "The capital of France is",
    "用三句话解释张量并行。",
]


@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_flashcomm1_additional_config_tp2_eager(monkeypatch) -> None:
    # The explicit additional-config value must win over the legacy environment
    # switch and reach worker subprocesses without a current-vLLM-config lookup.
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    with VllmRunner(
        MODEL,
        max_model_len=1024,
        max_num_seqs=3,
        max_num_batched_tokens=256,
        enforce_eager=True,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.75,
        additional_config={"enable_flashcomm1": True},
    ) as runner:
        outputs = runner.generate_greedy(PROMPTS, max_tokens=8)
        single_request_outputs = runner.model.generate(
            prompts=["用三句话解释张量并行。"],
            sampling_params=SamplingParams(
                max_tokens=8,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                logprobs=20,
                seed=1234,
            ),
        )

    assert outputs
    assert all(text for _, text in outputs)
    # A one-token decode step on TP2 is rank-local size 1 as well as full batch
    # size 1. The decoder must track the SP layout explicitly instead of
    # inferring it from that ambiguous row count.
    sequence = single_request_outputs[0].outputs[0]
    assert sequence.token_ids
    assert sequence.logprobs is not None
    logprobs = [value.logprob for topk in sequence.logprobs for value in topk.values()]
    assert logprobs
    assert all(isfinite(value) for value in logprobs)
