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

def test_qwen3_vl_4b_tp1_fp16():
    # 获取当前测试文件所在目录，然后定位到 data 目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path_str = os.path.join(current_dir, "..", "data", "qwen.png")
    
    # 使用 PIL.Image 加载图片，而不是直接传递路径字符串
    # vLLM 的 Processor 需要接收 PIL.Image.Image 对象
    image = Image.open(image_path_str)
    images = [image]
    
    # 在提示词中添加 <|image_pad|> 占位符，以便 vLLM 识别图片插入位置
    example_prompts = [
        "<|image_pad|>Describe this image in detail."
    ]
    max_tokens = 5

    with VllmRunner(
            "Qwen/Qwen3-VL-4B",
            tensor_parallel_size=1, # 4B模型较小，单卡即可运行；如需测试多卡可改为2
            enforce_eager=True,
            dtype="float16"
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens, images=images)

def test_qwen3_vl_8b_tp1_fp16():
    # 获取当前测试文件所在目录，然后定位到 data 目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path_str = os.path.join(current_dir, "..", "data", "qwen.png")
    
    # 使用 PIL.Image 加载图片，而不是直接传递路径字符串
    # vLLM 的 Processor 需要接收 PIL.Image.Image 对象
    image = Image.open(image_path_str)
    images = [image]
    
    # 在提示词中添加 <|image_pad|> 占位符，以便 vLLM 识别图片插入位置
    example_prompts = [
        "<|image_pad|>Describe this image in detail."
    ]
    max_tokens = 10

    with VllmRunner(
            "Qwen/Qwen3-VL-8B",
            tensor_parallel_size=1, 
            enforce_eager=True,
            dtype="float16"
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens, images=images)