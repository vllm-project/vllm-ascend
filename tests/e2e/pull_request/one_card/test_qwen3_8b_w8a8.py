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
# This file is a part of the vllm-ascend project.
#

import pytest
from vllm import SamplingParams
from vllm.config import CompilationConfig

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory, wait_until_npu_memory_free
from tests.e2e.pull_request.utils import PROMPTS_SHORT


@pytest.mark.e2e_features("multimodal", "full_decode_only", "w8a8", "eagle_3", "async_scheduling")
@pytest.mark.e2e_model("RedHatAI/Qwen3-8B-speculator.eagle3", "vllm-ascend/Qwen3-8B-W8A8")
@wait_until_npu_memory_free()
def test_dense_w8a8_eagle3_full_graph():
    """Verify dense W8A8 inference with Eagle-3 speculative decoding."""
    example_prompts = PROMPTS_SHORT
    sampling_params = SamplingParams(
        max_tokens=300,
        temperature=0.0,
        ignore_eos=False,
    )

    with VllmRunner(
        "vllm-ascend/Qwen3-8B-W8A8",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        disable_log_stats=False,
        max_model_len=4096,
        seed=1024,
        async_scheduling=False,
        quantization="ascend",
        speculative_config={
            "disable_padded_drafter_batch": False,
            "method": "eagle3",
            "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
            "num_speculative_tokens": 2,
            "draft_tensor_parallel_size": 1,
            "max_model_len": 128,
        },
        compilation_config=CompilationConfig(cudagraph_mode="FULL", cudagraph_capture_sizes=[5, 12]),
    ) as runner:
        spec_outputs = runner.generate(example_prompts, sampling_params)
        cleanup_dist_env_and_memory()
        del runner

    with VllmRunner(
        "vllm-ascend/Qwen3-8B-W8A8",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        disable_log_stats=False,
        max_model_len=4096,
        seed=1024,
        async_scheduling=False,
        quantization="ascend",
        compilation_config=CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[12]),
    ) as runner:
        ref_outputs = runner.generate(example_prompts, sampling_params)
        cleanup_dist_env_and_memory()
        del runner

    matches = 0
    threshold = 0.66
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        ref_token_ids = ref_output[0][0]
        spec_token_ids = spec_output[0][0]
        if ref_token_ids == spec_token_ids[: len(ref_token_ids)]:
            matches += 1
        else:
            print(f"ref_output: {ref_output[1][0]}")
            print(f"spec_output: {spec_output[1][0]}")

    assert matches > int(threshold * len(ref_outputs))
