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

import multiprocessing
import os
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.utils import fork_new_process_for_each_test
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

MODELS = ["Qwen/Qwen3-0.6B", "vllm-ascend/DeepSeek-V2-Lite-W8A8"]
ACLGRAPH_CAPTURE_SIZES = [1, 2, 4]


def full_and_piecewise_compilation_config() -> dict[str, object]:
    return {
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": ACLGRAPH_CAPTURE_SIZES.copy(),
    }


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [4])
@fork_new_process_for_each_test
@patch.dict(os.environ, {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
@patch.dict(os.environ, {"VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": "0"})
@patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"})
def test_aclgraph_mem_use(model: str, max_tokens: int) -> None:
    capture_called = multiprocessing.Value("i", 0)  # int, 0 or 1
    capture_mem_before = multiprocessing.Value("q", -1)  # long long (64-bit)
    capture_mem_after = multiprocessing.Value("q", -1)  # long long
    captured_graph_mem = multiprocessing.Value("q", -1)  # long long

    def capture_model_wrapper(original_method):
        def wrapped(self):
            mem_before = torch.npu.mem_get_info()[0]  # free memory
            result = original_method(self)
            mem_after = torch.npu.mem_get_info()[0]
            with capture_called.get_lock():
                capture_called.value = 1
                capture_mem_before.value = mem_before
                capture_mem_after.value = mem_after
                captured_graph_mem.value = int(result or 0)
            return result

        return wrapped

    original_capture = NPUModelRunner.capture_model

    with patch.object(NPUModelRunner, "capture_model", new=capture_model_wrapper(original_capture)):
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8":
            with VllmRunner(
                model,
                compilation_config=full_and_piecewise_compilation_config(),
                gpu_memory_utilization=0.7,
                max_model_len=1024,
                quantization="ascend",
            ) as vllm_model:
                _ = vllm_model.generate(prompts, sampling_params)
        else:
            with VllmRunner(
                model,
                compilation_config=full_and_piecewise_compilation_config(),
                gpu_memory_utilization=0.7,
            ) as vllm_model:
                _ = vllm_model.generate(prompts, sampling_params)

    assert capture_called.value == 1, "capture_model was not called during test"
    assert capture_mem_before.value != -1, "capture_mem_before not set"
    assert capture_mem_after.value != -1, "capture_mem_after not set"
    assert captured_graph_mem.value != -1, "captured_graph_mem not set"
    assert captured_graph_mem.value > 0, "capture_model did not report graph pool memory"

    print("capture_mem_before =", capture_mem_before.value)
    print("capture_mem_after =", capture_mem_after.value)
    print("captured_graph_mem =", captured_graph_mem.value)

    mem_used_by_capture = capture_mem_before.value - capture_mem_after.value
    # Empirical observation: capturing ACL graphs for Qwen3-0.6B uses ~0.20 GiB of NPU memory.
    # DeepSeek-V2-Lite-W8A8 uses ~0.68 GiB of NPU memory
    # FULL_AND_PIECEWISE captures one FULL decode path and one PIECEWISE mixed-batch path,
    # so allow up to 2x the historical PIECEWISE-only baseline plus runtime variance.
    if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8":
        baseline_capture_mem = 0.68
        capture_mem_tolerance = 1.5
    else:
        baseline_capture_mem = 0.20
        capture_mem_tolerance = 1.3
    max_capture_mem_gib = baseline_capture_mem * 2 * capture_mem_tolerance
    max_mem_expected = max_capture_mem_gib * (1024**3)
    assert captured_graph_mem.value < max_mem_expected, (
        f"capture_model graph pool used more memory than expected. "
        f"Used: {captured_graph_mem.value / (1024**3):.2f} GiB, "
        f"Free memory delta: {mem_used_by_capture / (1024**3):.2f} GiB, "
        f"Expected: < {max_capture_mem_gib:.2f} GiB"
    )
