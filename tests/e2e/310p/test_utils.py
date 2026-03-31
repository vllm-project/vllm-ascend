#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import os

from PIL import Image

from tests.e2e.conftest import VllmRunner


def get_test_image():
    """Get the image object for testing"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "data", "qwen.png")
    return Image.open(image_path)


def get_test_prompts():
    """Get the prompts for testing"""
    return ["<|image_pad|>Describe this image in detail."]


def run_vl_model_test(
    model_name: str, tensor_parallel_size: int, max_tokens: int, dtype: str = "float16", enforce_eager: bool = True
):
    """
    Generic visual language model test function

    Args:
        model_name: Model name, e.g., "Qwen/Qwen3-VL-4B"
        tensor_parallel_size: Tensor parallel size
        max_tokens: Maximum number of generated tokens
        dtype: Data type, default is float16
        enforce_eager: Whether to enforce eager mode
    """
    image = get_test_image()
    images = [image]
    prompts = get_test_prompts()

    with VllmRunner(
        model_name, tensor_parallel_size=tensor_parallel_size, enforce_eager=enforce_eager, dtype=dtype
    ) as vllm_model:
        vllm_model.generate_greedy(prompts, max_tokens, images=images)
