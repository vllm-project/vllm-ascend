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
import os
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

os.environ["HCCL_BUFFSIZE"] = "768"


@wait_until_npu_memory_free()
def test_models_pcp_dcp_basic():
    prompts = [
        "The capital of France is", "Hello, my name is Tom, I am",
        "The president of United States is", "AI future is"
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(model,
                    enforce_eager=True,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=2,
                    max_num_batched_tokens=1024,
                    enable_expert_parallel=True,
                    block_size=128) as runner:
        runner.model.generate(prompts, sampling_params)

    model = "vllm-ascend/Qwen3-30B-A3B-W8A8"
    with VllmRunner(model,
                    enforce_eager=True,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=1,
                    enable_expert_parallel=True,
                    block_size=128,
                    quantization="ascend",
    ) as runner:
        runner.model.generate(prompts, sampling_params)
    
    model = "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning"
    with VllmRunner(
            model,
            max_model_len=1024,
            tensor_parallel_size=2,
            prefill_context_parallel_size=2,
            decode_context_parallel_size=2,
            enable_expert_parallel=True,
            gpu_memory_utilization=0.2,
            block_size=128,
            quantization="ascend",
    ) as runner:
        runner.model.generate(prompts, sampling_params)

    model = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    with VllmRunner(model,
                    enforce_eager=True,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=1,
                    max_num_batched_tokens=1024,
                    enable_expert_parallel=True,
                    long_prefill_token_threshold=4,
                    gpu_memory_utilization=0.8,
                    block_size=128) as runner:
        runner.model.generate(prompts, sampling_params)


@wait_until_npu_memory_free()
def test_models_pcp_dcp_full_graph():
    prompts = [
        "The capital of France is", "Hello, my name is Tom, I am",
        "The president of United States is", "AI future is"
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(model,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=2,
                    max_num_batched_tokens=1024,
                    enable_expert_parallel=True,
                    block_size=128,
                    compilation_config={
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [4, 8, 24, 48, 60]
                    }) as runner:
        runner.model.generate(prompts, sampling_params)

    model = "vllm-ascend/Qwen3-30B-A3B-W8A8"
    with VllmRunner(model,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=1,
                    enable_expert_parallel=True,
                    block_size=128,
                    quantization="ascend",
                    compilation_config={
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [4, 8, 24, 48, 60]
                    }) as runner:
        runner.model.generate(prompts, sampling_params)


@wait_until_npu_memory_free()
def test_models_pcp_dcp_piece_wise():
    prompts = [
        "The capital of France is", "Hello, my name is Tom, I am",
        "The president of United States is", "AI future is"
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(model,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=2,
                    max_num_batched_tokens=1024,
                    enable_expert_parallel=True,
                    cudagraph_capture_sizes=[1, 2, 4, 8],
                    block_size=128) as runner:
        runner.model.generate(prompts, sampling_params)

    model = "vllm-ascend/Qwen3-30B-A3B-W8A8"
    with VllmRunner(model,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=1,
                    enable_expert_parallel=True,
                    cudagraph_capture_sizes=[1, 2, 4, 8],
                    block_size=128,
                    quantization="ascend") as runner:
        runner.model.generate(prompts, sampling_params)


@wait_until_npu_memory_free()
def test_pcp_basic():
    prompts = [
        "The capital of France is", "Hello, my name is Tom, I am",
        "The president of United States is", "AI future is"
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(model,
                    enforce_eager=True,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=1,
                    max_num_batched_tokens=1024,
                    enable_expert_parallel=True,
                    block_size=128) as runner:
        runner.model.generate(prompts, sampling_params)


@wait_until_npu_memory_free()
def test_pcp_full_graph():
    prompts = [
        "The capital of France is", "Hello, my name is Tom, I am",
        "The president of United States is", "AI future is"
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(model,
                    enforce_eager=False,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=1,
                    max_num_batched_tokens=1024,
                    enable_expert_parallel=True,
                    block_size=128,
                    compilation_config={
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [4, 8, 24, 48, 60]
                    }) as runner:
        runner.model.generate(prompts, sampling_params)


@wait_until_npu_memory_free()
def test_pcp_piece_wise():
    prompts = [
        "The capital of France is", "Hello, my name is Tom, I am",
        "The president of United States is", "AI future is"
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(model,
                    enforce_eager=False,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=1,
                    max_num_batched_tokens=1024,
                    enable_expert_parallel=True,
                    block_size=128) as runner:
        runner.model.generate(prompts, sampling_params)


@wait_until_npu_memory_free()
def test_dcp_basic():
    prompts = [
        "The capital of France is", "Hello, my name is Tom, I am",
        "The president of United States is", "AI future is"
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(model,
                    enforce_eager=True,
                    max_model_len=1024,
                    tensor_parallel_size=4,
                    prefill_context_parallel_size=1,
                    decode_context_parallel_size=2,
                    max_num_batched_tokens=1024,
                    enable_expert_parallel=True,
                    block_size=128,
                    compilation_config={"pass_config": {"enable_sp": True}}) as runner:
        runner.model.generate(prompts, sampling_params)

@wait_until_npu_memory_free()
def test_dcp_full_graph():
    prompts = [
        "The capital of France is", "Hello, my name is Tom, I am",
        "The president of United States is", "AI future is"
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(model,
                    enforce_eager=False,
                    max_model_len=1024,
                    tensor_parallel_size=4,
                    prefill_context_parallel_size=1,
                    decode_context_parallel_size=2,
                    max_num_batched_tokens=1024,
                    enable_expert_parallel=True,
                    block_size=128,
                    compilation_config={
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [4, 8, 24, 48, 60]
                    }) as runner:
        runner.model.generate(prompts, sampling_params)


@wait_until_npu_memory_free()
def test_dcp_piece_wise():
    prompts = [
        "The capital of France is", "Hello, my name is Tom, I am",
        "The president of United States is", "AI future is"
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(model,
                    enforce_eager=False,
                    max_model_len=1024,
                    tensor_parallel_size=4,
                    prefill_context_parallel_size=1,
                    decode_context_parallel_size=2,
                    max_num_batched_tokens=1024,
                    enable_expert_parallel=True,
                    block_size=128) as runner:
        runner.model.generate(prompts, sampling_params)


@patch.dict(
    os.environ,
    {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "OMP_NUM_THREADS": "1",
        "OMP_PROC_BIND": "false",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free()
@pytest.mark.skipif(
    torch.npu.device_count() < 4,
    reason="Kimi-K2.5-W4A8 multimodal test requires at least 4 NPUs.",
)
def test_kimi_k25_multimodal_basic():
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../310p/data/qwen.png"))
    image = Image.open(image_path).convert("RGB")
    prompts = [
        (
            "<|im_user|>user<|media_begin|>image<|media_content|><|media_pad|>"
            "<|media_end|>What is the content of this image?<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|>"
        ),
        (
            "<|im_user|>user<|media_begin|>image<|media_content|><|media_pad|>"
            "<|media_end|>Describe the content of this image in detail.<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|>"
        ),
    ]

    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"vision_chunk": {"type": "image", "image": image}},
            "multi_modal_uuids": {"vision_chunk": f"kimi_k25_image_{i}"},
        }
        for i, prompt in enumerate(prompts)
    ]

    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    model = "Eco-Tech/Kimi-K2.5-w4a8"
    with VllmRunner(
        model,
        enforce_eager=True,
        max_model_len=1024,
        tensor_parallel_size=4,
        prefill_context_parallel_size=4,
        decode_context_parallel_size=1,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.8,
        enable_expert_parallel=True,
        limit_mm_per_prompt={"vision_chunk": 1},
        block_size=128,
        quantization="ascend",
    ) as runner:
        outputs = runner.model.generate(inputs, sampling_params=sampling_params)
        assert len(outputs) == len(prompts)
        for output in outputs:
            assert output.outputs and output.outputs[0].text.strip()
