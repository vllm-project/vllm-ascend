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

import torch
from vllm import LLM, SamplingParams
from vllm.utils import GiB_bytes

from tests.utils import fork_new_process_for_each_test
from vllm_ascend.device_allocator.camem import CaMemAllocator

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")


@fork_new_process_for_each_test
def test_basic_camem():
    # some tensors from default memory pool
    shape = (1024, 1024)
    x = torch.empty(shape, device='npu:0')
    x.zero_()

    # some tensors from custom memory pool
    allocator = CaMemAllocator.get_instance()
    with allocator.use_memory_pool():
        # custom memory pool
        y = torch.empty(shape, device='npu:0')
        y.zero_()
        y += 1
        z = torch.empty(shape, device='npu:0')
        z.zero_()
        z += 2

    # they can be used together
    output = x + y + z
    assert torch.allclose(output, torch.ones_like(output) * 3)

    free_bytes = torch_npu.npu.mem_get_info()[0]
    allocator.sleep()
    free_bytes_after_sleep = torch_npu.npu.mem_get_info()[0]
    assert free_bytes_after_sleep > free_bytes
    allocator.wake_up()

    # they can be used together
    output = x + y + z
    assert torch.allclose(output, torch.ones_like(output) * 3)


@fork_new_process_for_each_test
def test_end_to_end():
    os.environ["VLLM_USE_V1"] = "0"
    free, total = torch_npu.npu.mem_get_info()
    used_bytes_baseline = total - free  # in case other process is running
    llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", enable_sleep_mode=True)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    # the benefit of `llm.sleep(level=2)` is mainly CPU memory usage,
    # which is difficult to measure in the test. therefore, we only
    # test sleep level 1 here.
    llm.sleep(level=1)

    free_gpu_bytes_after_sleep, total = torch_npu.npu.mem_get_info()
    used_bytes = total - free_gpu_bytes_after_sleep - used_bytes_baseline
    # now the memory usage should be less than the model weights
    # (0.5B model, 1GiB weights)
    assert used_bytes < 1 * GiB_bytes

    llm.wake_up()
    output2 = llm.generate(prompt, sampling_params)

    # cmp output
    assert output[0].outputs[0].text == output2[0].outputs[0].text
