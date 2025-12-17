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
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/e2e/multicard/test_qwen3_moe.py`.
"""

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


# def test_pcp_dcp_basic():
#     prompts = [
#         "The capital of France is",
#         "Hello, my name is Tom, I am",
#         "The president of United States is",
#         "AI future is"
#     ]
#     model = "/mnt/nfs/l00889328/DeepSeek-V2-Lite"
#     sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
#     with VllmRunner(model,
#                     enforce_eager=True,
#                     max_model_len=1024,
#                     tensor_parallel_size=2,
#                     prefill_context_parallel_size=2,
#                     decode_context_parallel_size=2,
#                     max_num_batched_tokens=1024,
#                     enable_expert_parallel=True,
#                     block_size=128
#                     ) as runner:
#         vllm_fullgraph_outputs = runner.model.generate(prompts,
#                                                        sampling_params)
#         for i, output in enumerate(vllm_fullgraph_outputs):
#             generated_text = output.outputs[0].text
#             print(f"req_num: {i}\nGenerated text: {generated_text!r}")
    
#     model = "/mnt/nfs/weights/Qwen3-30B-A3B-W8A8"
#     with VllmRunner(
#             model,
#             enforce_eager=True,
#             max_model_len=1024,
#             tensor_parallel_size=8,
#             prefill_context_parallel_size=2,
#             decode_context_parallel_size=2,
#             enable_expert_parallel=True,
#             block_size=128,
#             quantization="ascend",
#     ) as runner:
#         vllm_eager_outputs = runner.model.generate(prompts, sampling_params)
#         for i, output in enumerate(vllm_eager_outputs):
#             generated_text = output.outputs[0].text
#             print(f"req_num: {i}\nGenerated text: {generated_text!r}")

        
def test_pcp_dcp_full_graph():
    prompts = [
        "The capital of France is",
        "Hello, my name is Tom, I am",
        "The president of United States is",
        "AI future is"
    ]
    model = "/mnt/nfs/l00889328/DeepSeek-V2-Lite"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    # with VllmRunner(
    #         model,
    #         enforce_eager=False,
    #         max_model_len=1024,
    #         tensor_parallel_size=2,
    #         prefill_context_parallel_size=2,
    #         decode_context_parallel_size=2,
    #         max_num_batched_tokens=1024,
    #         enable_expert_parallel=True,
    #         block_size=128,
    #         compilation_config={
    #             "cudagraph_mode": "FULL_DECODE_ONLY",
    #             "cudagraph_capture_sizes": [4, 8, 24, 48, 60]}
    #     ) as runner:
    #     vllm_fullgraph_outputs = runner.model.generate(prompts,
    #                                                    sampling_params)
    #     for i, output in enumerate(vllm_fullgraph_outputs):
    #         generated_text = output.outputs[0].text
    #         print(f"req_num: {i}\nGenerated text: {generated_text!r}")
    
    model = "/mnt/nfs/weights/Qwen3-30B-A3B-W8A8"
    with VllmRunner(
            model,
            enforce_eager=False,
            max_model_len=1024,
            tensor_parallel_size=8,
            prefill_context_parallel_size=2,
            decode_context_parallel_size=2,
            enable_expert_parallel=True,
            block_size=128,
            quantization="ascend",
            compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4, 8, 24, 48, 60]}
        ) as runner:
        vllm_eager_outputs = runner.model.generate(prompts, sampling_params)
        for i, output in enumerate(vllm_eager_outputs):
            generated_text = output.outputs[0].text
            print(f"req_num: {i}\nGenerated text: {generated_text!r}")
            

def test_pcp_dcp_piece_wise():
    prompts = [
        "The capital of France is",
        "Hello, my name is Tom, I am",
        "The president of United States is",
        "AI future is"
    ]
    model = "/mnt/nfs/l00889328/DeepSeek-V2-Lite"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    # with VllmRunner(
    #         model,
    #         enforce_eager=False,
    #         max_model_len=1024,
    #         tensor_parallel_size=2,
    #         prefill_context_parallel_size=2,
    #         decode_context_parallel_size=2,
    #         max_num_batched_tokens=1024,
    #         enable_expert_parallel=True,
    #         block_size=128
    #     ) as runner:
    #     vllm_fullgraph_outputs = runner.model.generate(prompts,
    #                                                    sampling_params)
    #     for i, output in enumerate(vllm_fullgraph_outputs):
    #         generated_text = output.outputs[0].text
    #         print(f"req_num: {i}\nGenerated text: {generated_text!r}")
    
    model = "/mnt/nfs/weights/Qwen3-30B-A3B-W8A8"
    with VllmRunner(
            model,
            enforce_eager=False,
            max_model_len=1024,
            tensor_parallel_size=8,
            prefill_context_parallel_size=2,
            decode_context_parallel_size=2,
            enable_expert_parallel=True,
            block_size=128,
            quantization="ascend"
        ) as runner:
        vllm_eager_outputs = runner.model.generate(prompts, sampling_params)
        for i, output in enumerate(vllm_eager_outputs):
            generated_text = output.outputs[0].text
            print(f"req_num: {i}\nGenerated text: {generated_text!r}")
