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
from unittest.mock import patch

import pytest
from vllm.assets.image import ImageAsset

from tests.e2e.conftest import VllmRunner, qwen_prompt, wait_until_npu_memory_free

MODEL = "Qwen/Qwen3.6-27B"


@pytest.mark.e2e_features("multimodal", "eager_mode", "fia", "multi_instance")
@pytest.mark.e2e_model("Qwen/Qwen3.6-27B")
@patch.dict(os.environ, {"HCCL_BUFFSIZE": "1024"})
@wait_until_npu_memory_free()
def test_qwen3_6_27b_multimodel_fia_eager():
    """Verify multimodal generation with FIA op and eager mode."""
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    questions = [
        "What is the content of this image?",
        "Describe the content of this image in detail.",
        "What's in the image?",
        "Where is this image taken?",
    ]

    images = [image] * len(questions)
    prompts = qwen_prompt(questions)

    with VllmRunner(
        MODEL,
        max_model_len=4096,
        tensor_parallel_size=2,
        language_model_only=False,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            prompts=prompts,
            images=images,
            max_tokens=64,
        )

    assert outputs[0][1]


@pytest.mark.e2e_features("multimodal", "piecewise_graph", "fia", "multi_instance")
@pytest.mark.e2e_model("Qwen/Qwen3.6-27B")
@patch.dict(os.environ, {"HCCL_BUFFSIZE": "1024"})
@wait_until_npu_memory_free()
def test_qwen3_6_27b_multimodel_fia_acl_graph():
    """Verify multimodal generation with FIA op and FULL_AND_piecewise_graph mode."""
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    questions = [
        "What is the content of this image?",
        "Describe the content of this image in detail.",
        "What's in the image?",
        "Where is this image taken?",
    ]

    images = [image] * len(questions)
    prompts = qwen_prompt(questions)

    with VllmRunner(
        MODEL,
        max_model_len=4096,
        tensor_parallel_size=2,
        language_model_only=False,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        compilation_config={
            "cudagraph_mm_encoder": True,
            "cudagraph_capture_sizes": [1],
            "encoder_cudagraph_token_budgets": [128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            prompts=prompts,
            images=images,
            max_tokens=64,
        )

    assert outputs[0][1]
