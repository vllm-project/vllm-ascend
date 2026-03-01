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

import sys
import os

# 将 310p 目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 310p 目录
sys.path.insert(0, parent_dir)

from test_utils import run_vl_model_test


def test_qwen3_vl_4b_tp2_fp16():
    """Qwen3-VL-4B 双卡 FP16 测试"""
    run_vl_model_test(
        model_name="Qwen/Qwen3-VL-4B",
        tensor_parallel_size=2,
        max_tokens=5
    )


def test_qwen3_vl_8b_tp2_fp16():
    """Qwen3-VL-8B 双卡 FP16 测试"""
    run_vl_model_test(
        model_name="Qwen/Qwen3-VL-8B",
        tensor_parallel_size=2,
        max_tokens=10
    )

def test_qwen3_vl_32b_tp1_fp16():
    """Qwen3-VL-32B 四卡 FP16 测试"""
    run_vl_model_test(
        model_name="Qwen/Qwen3-VL-32B",
        tensor_parallel_size=4,
        max_tokens=10
    )