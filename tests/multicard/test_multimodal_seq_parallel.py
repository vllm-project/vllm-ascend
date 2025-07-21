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
#
"""
Compare the outputs of qwen25-vl with and without seq parallel.
Run `pytest tests/multicard/test_multimodal_seq_parallel.py`.
"""

import os

import pytest

from tests.model_utils import check_outputs_equal
from vllm.assets.image import ImageAsset

MODELS = ["Qwen/Qwen2.5-VL-3B-Instruct"]

@pytest.mark.skipif(os.getenv("VLLM_USE_V1") == "0",
                    reason="Qwen2.5-VL Seq parallel only support on v1")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [16])
def test_multimodal_seq_parallel_correctness(
    model: str,
    max_tokens: int,
    vllm_runner,
    prompt_template
) -> None:
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn' 
    image = ImageAsset("cherry_blossom") \
        .pil_image.convert("RGB")
    img_questions = [
        "What is the content of this image?",
        "Describe the content of this image in detail.",
        "What's in the image?",
        "Where is this image taken?",
    ]
    images = [image] * len(img_questions)
    prompts = prompt_template(img_questions)

    with vllm_runner(model_name=model,
                    max_model_len=4096,
                    max_num_seqs=16,
                    tensor_parallel_size=2,
                    distributed_executor_backend="mp",
                    mm_processor_kwargs={
                         "min_pixels": 28 * 28,
                         "max_pixels": 1280 * 28 * 28,
                         "fps": 1,
                     }) as vllm_model:
        vllm_sp_outputs = vllm_model.generate_greedy(prompts=prompts,
                                                     images=images,
                                                     max_tokens=max_tokens)

    with vllm_runner(
            model_name=model,
            max_model_len=4096,
            max_num_seqs=16,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": 1,
            }) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(prompts=prompts,
                                                  images=images, 
                                                  max_tokens=max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_outputs,
        outputs_1_lst=vllm_sp_outputs,
        name_0="vllm_outputs",
        name_1="vllm_sp_outputs",
    )