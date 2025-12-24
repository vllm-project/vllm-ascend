#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
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

import os
from unittest.mock import patch

import torch
from vllm import SamplingParams
from vllm.utils.mem_constants import GiB_bytes

from tests.e2e.conftest import VllmRunner
from tests.e2e.utils import fork_new_process_for_each_test


@fork_new_process_for_each_test
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "0"})
def test_end_to_end():
    free, total = torch.npu.mem_get_info()
    used_bytes_baseline = total - free  # in case other process is running

    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)

    with VllmRunner("Qwen/Qwen3-0.6B", enable_sleep_mode=True) as runner:

        output = runner.model.generate(prompt, sampling_params)
        # the benefit of `llm.sleep(level=2)` is mainly CPU memory usage,
        # which is difficult to measure in the test. therefore, we only
        # test sleep level 1 here.
        runner.model.sleep(level=1)

        free_gpu_bytes_after_sleep, total = torch.npu.mem_get_info()
        used_bytes = total - free_gpu_bytes_after_sleep - used_bytes_baseline
        # now the memory usage should be less than the model weights
        # (0.5B model, 1GiB weights)
        assert used_bytes < 1 * GiB_bytes

        runner.model.wake_up()
        output2 = runner.model.generate(prompt, sampling_params)

    # cmp output
    assert output[0].outputs[0].text == output2[0].outputs[0].text
