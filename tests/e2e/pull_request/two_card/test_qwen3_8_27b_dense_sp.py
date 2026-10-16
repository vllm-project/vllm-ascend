#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Dense sequence parallelism (FlashComm SP) for the Qwen3.5 dense family."""

import os

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.utils import compare_logprobs

# Qwen3.8-27B is the dense Qwen3.5-family checkpoint (model_type
# "qwen3_5_text", Qwen3_5ForConditionalGeneration architecture).
MODEL = os.environ.get("DENSE_SP_TEST_MODEL", "Qwen/Qwen3.8-27B")

PROMPTS = [
    "Hello, what's your name?",
    "The capital of the United States is",
    "The capital of France is",
    "The future of AI is",
]

RUNNER_KWARGS = {
    "model_name": MODEL,
    "max_model_len": 1024,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 256,
    "tensor_parallel_size": 2,
    "additional_config": {
        "enable_flashcomm1": True,
    },
}


@wait_until_npu_memory_free(0.7)
def test_qwen3_8_27b_dense_sequence_parallel_tp2_eager(monkeypatch) -> None:
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)

    # Dense FlashComm SP must stay numerically consistent with the plain
    # eager baseline. `additional_config` is excluded from the baseline by
    # compare_logprobs, so the baseline runs without the feature.
    compare_logprobs(
        runner_kwargs={
            **RUNNER_KWARGS,
            "enforce_eager": True,
        },
        prompts=PROMPTS,
    )


@wait_until_npu_memory_free(0.7)
def test_qwen3_8_27b_dense_sequence_parallel_tp2_graph(monkeypatch) -> None:
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)

    # Dense SP with decode ACL graphs (capture sizes aligned to the TP size
    # by the SP config) must stay numerically consistent with the plain
    # eager baseline. `additional_config` and `compilation_config` are
    # excluded from the baseline by compare_logprobs.
    compare_logprobs(
        runner_kwargs={
            **RUNNER_KWARGS,
            "compilation_config": {
                "cudagraph_capture_sizes": [4],
                "cudagraph_mode": "FULL_DECODE_ONLY",
            },
        },
        prompts=PROMPTS,
    )
