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
import sys

from tests.e2e.conftest import wait_until_npu_memory_free

current_dir = os.path.dirname(os.path.abspath(__file__))
full_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, full_dir)

# ruff: noqa: E402
import pytest

from tests.e2e.pull_request.utils_310p import run_vl_model_test


@pytest.mark.e2e_features("310P", "multimodal", "eager_mode")
@pytest.mark.e2e_model("Qwen/Qwen3-VL-8B-Instruct")
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_qwen3_vl_8b_tp1_fp16():
    """Qwen3-VL-8B single-card FP16 test"""
    run_vl_model_test(model_name="Qwen/Qwen3-VL-8B-Instruct", tensor_parallel_size=1, max_tokens=5)
