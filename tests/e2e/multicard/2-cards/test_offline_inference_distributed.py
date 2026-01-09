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
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/test_offline_inference.py`.
"""
import os
from unittest.mock import patch

import pytest
from modelscope import snapshot_download  # type: ignore
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

QWEN_DENSE_MODELS = [
    "vllm-ascend/Qwen3-0.6B-W8A8",
]

QWEN_W4A8_MODELS = [
    "vllm-ascend/Qwen3-1.7B-W4A8-V1",
]

DEEPSEEK_W4A8_MODELS = [
    "vllm-ascend/DeepSeek-V3.1-W4A8-puring",
]


def test_deepseek_multistream_moe_tp2():
    example_prompts = [
        "Hello, my name is",
    ]
    dtype = "half"
    max_tokens = 5
    with VllmRunner(
            "vllm-ascend/DeepSeek-V3-Pruning",
            dtype=dtype,
            tensor_parallel_size=2,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            distributed_executor_backend="mp",
            additional_config={
                "enable_multistream_moe": True,
                "refresh": True,
            },
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@pytest.mark.parametrize("model", QWEN_W4A8_MODELS)
def test_qwen3_w4a8_dynamic_tp2(model):
    prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
            snapshot_download(model),
            max_model_len=8192,
            dtype="auto",
            tensor_parallel_size=2,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(prompts, max_tokens)


def test_qwen3_moe_sp_tp2() -> None:
    example_prompts = [
        "Hello, my name is",
    ]
    sampling_params = SamplingParams(max_tokens=5,
                                     temperature=0.0,
                                     top_k=50,
                                     top_p=0.9)

    with VllmRunner(snapshot_download("Qwen/Qwen3-30B-A3B"),
                    dtype="auto",
                    tensor_parallel_size=2,
                    distributed_executor_backend="mp",
                    compilation_config={"pass_config": {
                        "enable_sp": True
                    }},
                    enable_expert_parallel=True,
                    enforce_eager=True) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)


@pytest.mark.parametrize("model", DEEPSEEK_W4A8_MODELS)
@patch.dict(os.environ, {"HCCL_BUFFSIZE": "2048"})
def test_deepseek_w4a8_accuracy_tp2(model):
    prompts = [
        "Hello, my name is", "The president of the United States is",
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs"
    ]
    vllm_ds_w4a8_answers = [
        '逍遙而至地去 accrued', '平行于我udo madreHelen', 'ysteepaolis backwards Kj'
    ]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)
    with VllmRunner(snapshot_download(model),
                    dtype="auto",
                    tensor_parallel_size=2,
                    cudagraph_capture_sizes=[1, 2, 4, 8],
                    quantization="ascend",
                    enable_expert_parallel=True) as vllm_model:
        vllm_quant_outputs = vllm_model.model.generate(prompts,
                                                       sampling_params)

    vllm_quant_outputs_list = []
    for output in vllm_quant_outputs:
        vllm_quant_outputs_list.append(
            ([output.outputs[0].index], output.outputs[0].text))
    vllm_answer_list = []
    vllm_answer_list = ([([0], answer) for answer in vllm_ds_w4a8_answers])

    check_outputs_equal(outputs_0_lst=vllm_answer_list,
                        outputs_1_lst=vllm_quant_outputs_list,
                        name_0="vllm_quant_outputs",
                        name_1="vllm_answer_outputs")


@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"})
@patch.dict(os.environ, {"VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": "1"})
def test_qwen3_moe_fc2_tp2() -> None:
    example_prompts = [
        "Hello, my name is",
    ]
    sampling_params = SamplingParams(max_tokens=5,
                                     temperature=0.0,
                                     top_k=50,
                                     top_p=0.9)

    with VllmRunner(snapshot_download("Qwen/Qwen3-30B-A3B"),
                    dtype="auto",
                    tensor_parallel_size=2,
                    distributed_executor_backend="mp",
                    enable_expert_parallel=True,
                    enforce_eager=True) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)


@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"})
@patch.dict(os.environ, {"VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": "1"})
def test_qwen3_moe_fc2_oshard_tp2() -> None:
    example_prompts = [
        "Hello, my name is",
    ]
    sampling_params = SamplingParams(max_tokens=5,
                                     temperature=0.0,
                                     top_k=50,
                                     top_p=0.9)

    with VllmRunner(
            snapshot_download("Qwen/Qwen3-30B-A3B"),
            dtype="auto",
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            enable_expert_parallel=True,
            enforce_eager=
            True,  # TODO(Levi-JQ): support graph mode for fc2 in Qwen 
            additional_config={"layer_sharding": ["o_proj"]}) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)


@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"})
def test_deepseek_v2_lite_fc1_tp2() -> None:
    example_prompts = [
        "test" * 1001,
    ]
    sampling_params = SamplingParams(max_tokens=5,
                                     temperature=0.0,
                                     top_k=50,
                                     top_p=0.9)
    with VllmRunner(snapshot_download("vllm-ascend/DeepSeek-V2-Lite-W8A8"),
                    dtype="auto",
                    tensor_parallel_size=2,
                    distributed_executor_backend="mp",
                    enable_expert_parallel=True,
                    enforce_eager=True,
                    quantization="ascend") as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)


@pytest.mark.parametrize("model", QWEN_DENSE_MODELS)
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"})
def test_qwen3_dense_fc1_tp2(model):
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5

    with VllmRunner(
            snapshot_download(model),
            max_model_len=8192,
            dtype="auto",
            tensor_parallel_size=2,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@pytest.mark.parametrize("model", QWEN_DENSE_MODELS)
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_PREFETCH_MLP": "1"})
def test_qwen3_dense_prefetch_mlp_weight_tp2(model):
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5

    with VllmRunner(
            snapshot_download(model),
            max_model_len=8192,
            dtype="auto",
            tensor_parallel_size=2,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
