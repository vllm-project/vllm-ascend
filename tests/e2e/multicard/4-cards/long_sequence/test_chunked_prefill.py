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
import os
import random
import string
from unittest.mock import patch

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


def generate_prompts(input_len, batchsize):
    prompts = [
        " ".join([
            f"{random.choice(string.ascii_letters)}" for _ in range(input_len)
        ]) for _ in range(batchsize)
    ]
    return prompts


@patch.dict(
    os.environ, {
        "HCCL_BUFFSIZE": "768",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"
    })
def test_models_chunked_prefill_mixed_length_prompts_including_1_token():
    TEST_ROPE_PARAMETERS = {
        "rope_theta": 1000000,
        "rope_type": "yarn",
        "factor": 4,
        "original_max_position_embeddings": 32768
    }
    prompts = [
        generate_prompts(128 * 1024, 1)[0],
        generate_prompts(1, 1)[0],
        generate_prompts(9104, 1)[0],
    ]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

    model = "vllm-ascend/Qwen3-30B-A3B-W8A8"
    with VllmRunner(
            model,
            enforce_eager=True,
            max_num_seqs=2,
            max_num_batched_tokens=131000,
            max_model_len=132000,
            tensor_parallel_size=2,
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            enable_expert_parallel=True,
            block_size=128,
            quantization="ascend",
            hf_overrides={"rope_parameters": TEST_ROPE_PARAMETERS},
    ) as runner:
        runner.model.generate(prompts, sampling_params)
