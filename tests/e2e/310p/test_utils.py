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

from tests.e2e.conftest import VllmRunner
from PIL import Image
import os


def get_test_image():
    """获取测试用的图片对象"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "data", "qwen.png")
    return Image.open(image_path)


def get_test_prompts():
    """获取测试用的提示词"""
    return ["<|image_pad|>Describe this image in detail."]


def run_vl_model_test(model_name: str, 
                      tensor_parallel_size: int, 
                      max_tokens: int,
                      dtype: str = "float16",
                      enforce_eager: bool = True):
    """
    通用的视觉语言模型测试函数
    
    Args:
        model_name: 模型名称，如 "Qwen/Qwen3-VL-4B"
        tensor_parallel_size: 张量并行大小
        max_tokens: 最大生成 token 数
        dtype: 数据类型，默认 float16
        enforce_eager: 是否强制使用 eager 模式
    """
    image = get_test_image()
    images = [image]
    prompts = get_test_prompts()

    with VllmRunner(
            model_name,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            dtype=dtype
    ) as vllm_model:
        vllm_model.generate_greedy(prompts, max_tokens, images=images)