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
from typing import Dict

from tests.e2e.conftest import VllmRunner

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"


def _deepseek_torchair_test_fixture(
    additional_config: Dict,
    *,
    tensor_parallel_size=2,
    use_v1_schduler=False,
):
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    kwargs = {}
    if not use_v1_schduler:
        kwargs = {
            "ascend_scheduler_config": {
                "enabled": True,
            },
            "refresh": True,
        }
    additional_config.update(**kwargs)

    with VllmRunner(
            "vllm-ascend/DeepSeek-V3-Pruning",
            dtype="half",
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="mp",
            additional_config=additional_config,
    ) as vllm_model:
        # use greedy sampler to make sure the generated results are fix
        vllm_output = vllm_model.generate_greedy(example_prompts, 5)

    # NOTE: vllm-ascend/DeepSeek-V3-Pruning is a random weight of
    # DeepSeek-V3 with 2 hidden layers, thus the golden results seems
    # inaccurate. This will only change if accuracy improves with the
    # official weights of DeepSeek-V3.
    golden_results = [
        'Hello, my name is下载早点向前很有่อง',
        'The president of the United States isSender)## physiological Albany',
        'The capital of France is Rocky转角 hospitalizedinterval sparked',
        'The future of AI is её asegο BIOS一扫',
    ]

    assert len(golden_results) == len(vllm_output)
    for i in range(len(vllm_output)):
        assert golden_results[i] == vllm_output[i][1]
        print(f"Generated text: {vllm_output[i][1]!r}")


def test_e2e_deepseekv3_with_torchair():
    additional_config = {
        "torchair_graph_config": {
            "enabled": True,
        },
    }
    _deepseek_torchair_test_fixture(additional_config)


def test_e2e_deepseekv3_with_torchair_ms_mla():
    additional_config = {
        "torchair_graph_config": {
            "enabled": True,
            "enable_multistream_mla": True,
        },
    }
    _deepseek_torchair_test_fixture(additional_config)


def test_e2e_deepseekv3_with_torchair_v1scheduler():
    additional_config = {
        "torchair_graph_config": {
            "enabled": True,
        },
    }
    _deepseek_torchair_test_fixture(additional_config, use_v1_schduler=True)


def _pangu_torchair_test_fixture(
    additional_config: Dict,
    *,
    tensor_parallel_size=2,
):
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # torchair is only work without chunked-prefill now
    kwargs = {
        "ascend_scheduler_config": {
            "enabled": True,
        },
        "refresh": True,
    }
    additional_config.update(**kwargs)

    with VllmRunner(
            "vllm-ascend/pangu-pro-moe-pruing",
            dtype="half",
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="mp",
            additional_config=additional_config,
            enable_expert_parallel=True,
    ) as vllm_model:
        # use greedy sampler to make sure the generated results are fix
        vllm_output = vllm_model.generate_greedy(example_prompts, 5)

    # NOTE: vllm-ascend/pangu-pro-moe-pruing is only part of PanguProMoE
    # with 2 hidden layers, thus the golden results seems inaccurate.
    # This will only change if accuracy changes with the official weights
    # of PanguProMoE.
    golden_results = [
        'Hello, my name is Remempondeprecatedmiot忱',
        'The president of the United States is Remem下的一个 rever ceremoni Segnali',
        'The capital of France is Rememvoud administrativ Remem投',
        'The future of AI isotope Segnali Zoeken精细化 supus',
    ]

    assert len(golden_results) == len(vllm_output)
    for i in range(len(vllm_output)):
        assert golden_results[i] == vllm_output[i][1]
        print(f"Generated text: {vllm_output[i][1]!r}")


def test_e2e_pangu_with_torchair():
    additional_config = {
        "torchair_graph_config": {
            "enabled": True,
        },
    }
    _pangu_torchair_test_fixture(additional_config)


def _qwen_torchair_test_fixture(
    model,
    tp,
    enable_expert_parallel,
):
    # The current access control does not support 16 cards,
    # so the MC2 operator in Qwen's graph mode cannot run.
    # Once 16-card support is available,
    # this e2e can be switched to graph mode.
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    additional_config = {
        "torchair_graph_config": {
            "enabled": False,
        },
        "ascend_scheduler_config": {
            "enabled": True,
        },
        "refresh": True,
    }

    with VllmRunner(
            model,
            dtype="half",
            tensor_parallel_size=tp,
            distributed_executor_backend="mp",
            enforce_eager=True,
            additional_config=additional_config,
            enable_expert_parallel=enable_expert_parallel,
    ) as vllm_model:
        # use greedy sampler to make sure the generated results are fix
        vllm_output = vllm_model.generate_greedy(example_prompts, 5)

    # NOTE: vllm-ascend/pangu-pro-moe-pruing is only part of PanguProMoE
    # with 2 hidden layers, thus the golden results seems inaccurate.
    # This will only change if accuracy changes with the official weights
    # of PanguProMoE.
    golden_results = [
        'Hello, my name is Remempondeprecatedmiot忱',
        'The president of the United States is Remem下的一个 rever ceremoni Segnali',
        'The capital of France is Rememvoud administrativ Remem投',
        'The future of AI isotope Segnali Zoeken精细化 supus',
    ]

    assert len(golden_results) == len(vllm_output)
    for i in range(len(vllm_output)):
        print(f"Generated text: {vllm_output[i][1]!r}")


def test_e2e_qwen2_with_torchair():
    _qwen_torchair_test_fixture("Qwen/Qwen2.5-0.5B-Instruct", 2, False)


def test_e2e_qwen3_moe_with_torchair():
    _qwen_torchair_test_fixture("Qwen/Qwen3-30B-A3B", 2, True)
