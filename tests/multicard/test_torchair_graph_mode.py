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
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/multicard/test_torchair_graph_mode.py`.
"""
import os

import pytest

from tests.conftest import VllmRunner

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"


@pytest.mark.skipif(os.getenv("VLLM_USE_V1") == "0",
                    reason="torchair graph is not supported on v0")
def test_e2e_deepseekv3_with_torchair(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_MODELSCOPE", "True")
        m.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        example_prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        dtype = "half"
        max_tokens = 5
        # torchair is only work without chunked-prefill now
        with VllmRunner(
                "vllm-ascend/DeepSeek-V3-Pruning",
                dtype=dtype,
                tensor_parallel_size=4,
                distributed_executor_backend="mp",
                additional_config={
                    "torchair_graph_config": {
                        "enabled": True,
                    },
                    "ascend_scheduler_config": {
                        "enabled": True,
                    },
                    "refresh": True,
                },
                enforce_eager=False,
        ) as vllm_model:
            # use greedy sampler to make sure the generated results are fix
            vllm_output = vllm_model.generate_greedy(example_prompts,
                                                     max_tokens)
        # NOTE: vllm-ascend/DeepSeek-V3-Pruning is a random weight of
        # DeepSeek-V3 with 2 hidden layers, thus the golden results seems
        # inaccurate. This will only change if accuracy improves with the
        # official weights of DeepSeek-V3.
        golden_results = [
            'Hello, my name is feasibility伸 spazio debtor添',
            'The president of the United States is begg"""\n杭州风和 bestimm',
            'The capital of France is frequentlyশามalinkAllowed',
            'The future of AI is deleting俯احت怎么样了حراف',
        ]

        assert len(golden_results) == len(vllm_output)
        for i in range(len(vllm_output)):
            assert golden_results[i] == vllm_output[i][1]
            print(f"Generated text: {vllm_output[i][1]!r}")
