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
from modelscope import snapshot_download  # type: ignore
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

MODELS = ["Qwen/Qwen3-0.6B", "vllm-ascend/DeepSeek-V2-Lite-W8A8"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [4])
@patch.dict(os.environ, {"VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": "0"})
@patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"})
def test_aclgraph_mem_use(model: str, max_tokens: int) -> None:
    del os.environ["VLLM_WORKER_MULTIPROC_METHOD"]
    capture_called = multiprocessing.Value("i", 0)  # int, 0 or 1
    capture_mem_before = multiprocessing.Value("q", -1)  # long long (64-bit)
    capture_mem_after = multiprocessing.Value("q", -1)  # long long

    def capture_model_wrapper(original_method):

        def wrapped(self):
            mem_before = torch.npu.mem_get_info()[0]  # free memory
            result = original_method(self)
            mem_after = torch.npu.mem_get_info()[0]
            with capture_called.get_lock():
                capture_called.value = 1
                capture_mem_before.value = mem_before
                capture_mem_after.value = mem_after
            return result

        return wrapped

    original_capture = NPUModelRunner._capture_model

    with patch.object(NPUModelRunner,
                      '_capture_model',
                      new=capture_model_wrapper(original_capture)):
        prompts = [
            "Hello, my name is", "The president of the United States is",
            "The capital of France is", "The future of AI is"
        ]
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         temperature=0.0)
        if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8":
            vllm_model = VllmRunner(snapshot_download(model),
                                    max_model_len=1024,
                                    quantization="ascend")
        else:
            vllm_model = VllmRunner(snapshot_download(model))
        _ = vllm_model.generate(prompts, sampling_params)

    assert capture_called.value == 1, "_capture_model was not called during test"
    assert capture_mem_before.value != -1, "capture_mem_before not set"
    assert capture_mem_after.value != -1, "capture_mem_after not set"

    print("capture_mem_before =", capture_mem_before.value)
    print("capture_mem_after =", capture_mem_after.value)

    mem_used_by_capture = capture_mem_before.value - capture_mem_after.value
    # Empirical observation: capturing ACL graphs for Qwen3-0.6B uses ~0.20 GiB of NPU memory.
    # DeepSeek-V2-Lite-W8A8 uses ~0.68 GiB of NPU memory
    # a 1.3x tolerance is applied to account for runtime variance.
    if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8":
        baseline_capture_mem = 0.68
        capture_mem_tolerance = 1.5
    else:
        baseline_capture_mem = 0.20
        capture_mem_tolerance = 1.3
    max_capture_mem_gib = baseline_capture_mem * capture_mem_tolerance
    max_mem_expected = max_capture_mem_gib * (1024**3)
    assert mem_used_by_capture < max_mem_expected, (
        f"_capture_model used more memory than expected. "
        f"Used: {mem_used_by_capture / (1024**3):.2f} GiB, "
        f"Expected: < {max_capture_mem_gib:.2f} GiB")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = 'spawn'
