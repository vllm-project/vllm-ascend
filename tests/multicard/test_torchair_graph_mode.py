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


def test_e2e_deepseekv3_with_torchair(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_MODELSCOPE", "True")
        example_prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        dtype = "half"
        max_tokens = 5
        with VllmRunner(
                "deepseek-ai/DeepSeek-V3",
                dtype=dtype,
                tensor_parallel_size=4,
                distributed_executor_backend="mp",
                additional_config={"torchair_graph_config": {
                    "enable": True
                }},
                load_format="dummy",
        ) as vllm_model:
            vllm_model.generate_greedy(example_prompts, max_tokens)
