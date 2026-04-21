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
import pytest

from tests.e2e.conftest import DPVllmRunner, VllmRunner, wait_until_npu_memory_free
from tests.e2e.model_utils import check_outputs_equal

DS3 = "deepseek-ai/DeepSeek-V2-Lite-Chat"
MODELS = [
    DS3,
]
MOE_MODELS = [
    DS3,
]

DATA_PARALLELS = [2]
TENSOR_PARALLELS = [1, 2]
PIPELINE_PARALLELS = [2]
DIST_EXECUTOR_BACKEND = ["mp", "ray"]

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
GOLDEN = [
    (
        [
            100000,
            17464,
            11,
            601,
            1210,
            317,
            46462,
            608,
            245,
            4541,
            7712,
            13,
            2682,
            6207,
            317,
            276,
            2774,
            340,
            366,
            254,
            1608,
            2784,
        ],
        "Hello, my name is***** am a computer expert. My goal is to provide you with the best experience",
    ),
    (
        [
            100000,
            549,
            3680,
            280,
            20838,
            317,
            6464,
            11,
            548,
            359,
            487,
            82,
            441,
            1673,
            895,
            10694,
            13,
            1733,
            20838,
            5495,
            11106,
            276,
        ],
        "The future of AI is bright, but it’s not without its challenges. As AI technology continues to",
    ),
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("pp_size", PIPELINE_PARALLELS)
@pytest.mark.parametrize("distributed_executor_backend", DIST_EXECUTOR_BACKEND)
@wait_until_npu_memory_free(target_free_percentage=0.6)
def test_models_pp2_tp2(model: str, tp_size: int, pp_size: int, distributed_executor_backend: str) -> None:
    with VllmRunner(
        model,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        cudagraph_capture_sizes=[1, 2, 4],
        distributed_executor_backend=distributed_executor_backend,
        gpu_memory_utilization=0.7,
        enable_expert_parallel=model in MOE_MODELS,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, 16)
        check_outputs_equal(
            outputs_0_lst=outputs,
            outputs_1_lst=GOLDEN,
            name_0=f"{model}-tp{tp_size}pp{pp_size}",
            name_1="GOLDEN",
        )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dp_size", DATA_PARALLELS)
@pytest.mark.parametrize("pp_size", PIPELINE_PARALLELS)
@pytest.mark.parametrize("distributed_executor_backend", DIST_EXECUTOR_BACKEND)
@wait_until_npu_memory_free(target_free_percentage=0.6)
def test_models_pp2_dp2(model: str, dp_size: int, pp_size: int, distributed_executor_backend: str) -> None:
    with DPVllmRunner(
        model,
        data_parallel_size=dp_size,
        pipeline_parallel_size=pp_size,
        cudagraph_capture_sizes=[1, 2, 4],
        distributed_executor_backend=distributed_executor_backend,
        gpu_memory_utilization=0.7,
        enable_expert_parallel=model in MOE_MODELS,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, 16)
        check_outputs_equal(
            outputs_0_lst=outputs,
            outputs_1_lst=GOLDEN,
            name_0=f"{model}-dp{dp_size}pp{pp_size}",
            name_1="GOLDEN",
        )
