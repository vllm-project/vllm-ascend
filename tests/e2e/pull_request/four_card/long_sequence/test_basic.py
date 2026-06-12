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
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

os.environ["HCCL_BUFFSIZE"] = "768"

E2E_ROOT = Path(__file__).resolve().parents[3]
QWEN_IMAGE_PATH = E2E_ROOT / "prompts" / "qwen.png"


@wait_until_npu_memory_free()
def test_models_pcp_dcp_basic():
    prompts = [
        "The capital of France is",
        "Hello, my name is Tom, I am",
        "The president of United States is",
        "AI future is",
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(
        model,
        enforce_eager=True,
        max_model_len=1024,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=2,
        max_num_batched_tokens=1024,
        enable_expert_parallel=True,
        block_size=128,
    ) as runner:
        runner.model.generate(prompts, sampling_params)

    model = "vllm-ascend/Qwen3-30B-A3B-W8A8"
    with VllmRunner(
        model,
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
    with VllmRunner(
        model,
        enforce_eager=True,
        max_model_len=1024,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=1,
        max_num_batched_tokens=1024,
        enable_expert_parallel=True,
        long_prefill_token_threshold=4,
        gpu_memory_utilization=0.8,
        block_size=128,
    ) as runner:
        outputs = runner.generate_greedy(prompts, 5)
        results = [item[1] for item in outputs]
        golden = [
            "The capital of France is Paris.\nThe capital",
            "Hello, my name is Tom, I am a 20 years",
            "The president of United States is the head of state and",
            "AI future is not just about technology,",
        ]
        res_percent = calculate_total_char_match_percent(results, golden)
        assert res_percent > 80


@patch.dict(
    os.environ,
    {
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free()
@pytest.mark.skipif(
    torch.npu.device_count() < 4,
    reason="DeepSeek V4 DSA CP e2e test requires at least 4 NPUs.",
)
def test_deepseek_v4_w4a8_dsa_cp_basic_greedy():
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "What is the meaning of life?",
    ]
    max_tokens = 5

    with VllmRunner(
        "gdydems/DeepSeek-V4-Flash-w4a8-mtp",
        max_model_len=8192,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        dtype="auto",
        tensor_parallel_size=4,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.9,
        quantization="ascend",
        tokenizer_mode="deepseek_v4",
        block_size=128,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
        additional_config={
            "enable_flashcomm1": True,
            "enable_dsa_cp": True,
            "multistream_dsa_preprocess": True,
        },
    ) as runner:
        outputs = runner.generate_greedy(prompts, max_tokens)

    assert len(outputs) == len(prompts)
    for output_ids, output_str in outputs:
        assert len(output_str) > 0
        assert len(output_ids) > 0


@wait_until_npu_memory_free()
def test_models_pcp_dcp_full_graph():
    prompts = [
        "The capital of France is",
        "Hello, my name is Tom, I am",
        "The president of United States is",
        "AI future is",
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=2,
        max_num_batched_tokens=1024,
        enable_expert_parallel=True,
        block_size=128,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8, 24, 48, 60]},
    ) as runner:
        runner.model.generate(prompts, sampling_params)

    model = "vllm-ascend/Qwen3-30B-A3B-W8A8"
    with VllmRunner(
        model,
        max_model_len=1024,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        block_size=128,
        quantization="ascend",
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8, 24, 48, 60]},
    ) as runner:
        runner.model.generate(prompts, sampling_params)


@wait_until_npu_memory_free()
def test_models_pcp_dcp_piece_wise():
    prompts = [
        "The capital of France is",
        "Hello, my name is Tom, I am",
        "The president of United States is",
        "AI future is",
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=2,
        max_num_batched_tokens=1024,
        enable_expert_parallel=True,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        block_size=128,
    ) as runner:
        runner.model.generate(prompts, sampling_params)

    model = "vllm-ascend/Qwen3-30B-A3B-W8A8"
    with VllmRunner(
        model,
        max_model_len=1024,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        block_size=128,
        quantization="ascend",
    ) as runner:
        runner.model.generate(prompts, sampling_params)


@wait_until_npu_memory_free()
def test_pcp_basic():
    prompts = [
        "The capital of France is",
        "Hello, my name is Tom, I am",
        "The president of United States is",
        "AI future is",
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(
        model,
        enforce_eager=True,
        max_model_len=1024,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=1,
        max_num_batched_tokens=1024,
        enable_expert_parallel=True,
        block_size=128,
    ) as runner:
        runner.model.generate(prompts, sampling_params)


@wait_until_npu_memory_free()
def test_pcp_full_graph():
    prompts = [
        "The capital of France is",
        "Hello, my name is Tom, I am",
        "The president of United States is",
        "AI future is",
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(
        model,
        enforce_eager=False,
        max_model_len=1024,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=1,
        max_num_batched_tokens=1024,
        enable_expert_parallel=True,
        block_size=128,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8, 24, 48, 60]},
    ) as runner:
        runner.model.generate(prompts, sampling_params)


@wait_until_npu_memory_free()
def test_pcp_piece_wise():
    prompts = [
        "The capital of France is",
        "Hello, my name is Tom, I am",
        "The president of United States is",
        "AI future is",
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(
        model,
        enforce_eager=False,
        max_model_len=1024,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=1,
        max_num_batched_tokens=1024,
        enable_expert_parallel=True,
        block_size=128,
    ) as runner:
        runner.model.generate(prompts, sampling_params)


@wait_until_npu_memory_free()
def test_dcp_basic():
    prompts = [
        "The capital of France is",
        "Hello, my name is Tom, I am",
        "The president of United States is",
        "AI future is",
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(
        model,
        enforce_eager=True,
        max_model_len=1024,
        tensor_parallel_size=4,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=2,
        max_num_batched_tokens=1024,
        enable_expert_parallel=True,
        block_size=128,
        compilation_config={"pass_config": {"enable_sp": True}},
    ) as runner:
        runner.model.generate(prompts, sampling_params)


@wait_until_npu_memory_free()
def test_dcp_full_graph():
    prompts = [
        "The capital of France is",
        "Hello, my name is Tom, I am",
        "The president of United States is",
        "AI future is",
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(
        model,
        enforce_eager=False,
        max_model_len=1024,
        tensor_parallel_size=4,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=2,
        max_num_batched_tokens=1024,
        enable_expert_parallel=True,
        block_size=128,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8, 24, 48, 60]},
    ) as runner:
        runner.model.generate(prompts, sampling_params)


@wait_until_npu_memory_free()
def test_dcp_piece_wise():
    prompts = [
        "The capital of France is",
        "Hello, my name is Tom, I am",
        "The president of United States is",
        "AI future is",
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(
        model,
        enforce_eager=False,
        max_model_len=1024,
        tensor_parallel_size=4,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=2,
        max_num_batched_tokens=1024,
        enable_expert_parallel=True,
        block_size=128,
    ) as runner:
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
    reason="Qwen3-VL-8B-Instruct multimodal test requires at least 4 NPUs.",
)
def test_qwen3_vl_8b_multimodal_single_and_multi_image():
    image = Image.open(QWEN_IMAGE_PATH).convert("RGB")

    single_image_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "Describe this image in detail.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    multi_image_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "Compare these two images and describe similarities.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = [
        {
            "prompt": single_image_prompt,
            "multi_modal_data": {"image": image},
        },
        {
            "prompt": multi_image_prompt,
            "multi_modal_data": {"image": [image, image.copy()]},
        },
    ]

    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    model = "Qwen/Qwen3-VL-8B-Instruct"
    with VllmRunner(
        model,
        enforce_eager=False,
        max_model_len=4096,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=1,
        max_num_batched_tokens=1024,
        block_size=128,
        limit_mm_per_prompt={"image": 2},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4, 8, 24, 48, 60],
        },
    ) as runner:
        outputs = runner.model.generate(inputs, sampling_params=sampling_params)
        assert len(outputs) == len(inputs)
        for output in outputs:
            assert output.outputs and output.outputs[0].text.strip()


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
    reason="Qwen3.5-4B multimodal test requires at least 4 NPUs.",
)
def test_qwen3_5_4b_multimodal_single_and_multi_image():
    image_1 = Image.open(QWEN_IMAGE_PATH).convert("RGB")
    image_2 = Image.open(QWEN_IMAGE_PATH).convert("RGB")

    single_image_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "Describe this image in detail.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    multi_image_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Image 1: <|vision_start|><|image_pad|><|vision_end|>\n"
        "Image 2: <|vision_start|><|image_pad|><|vision_end|>\n"
        "Compare these two images and describe one similarity.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = [
        {
            "prompt": single_image_prompt,
            "multi_modal_data": {"image": image_1},
        },
        {
            "prompt": multi_image_prompt,
            "multi_modal_data": {"image": [image_1, image_2]},
        },
    ]

    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    model = "Qwen/Qwen3.5-4B"
    with VllmRunner(
        model,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=4096,
        tensor_parallel_size=1,
        prefill_context_parallel_size=4,
        decode_context_parallel_size=1,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 2},
        block_size=128,
    ) as runner:
        outputs = runner.model.generate(inputs, sampling_params=sampling_params)
        assert len(outputs) == len(inputs)
        for output in outputs:
            assert output.outputs and output.outputs[0].text.strip()


def calculate_total_char_match_percent(pred_list: list[str], target_list: list[str]) -> float:
    if len(pred_list) != len(target_list):
        raise ValueError("list length not same")

    total_matched = 0
    total_checked = 0
    for pred_str, target_str in zip(pred_list, target_list):
        check_len = min(len(pred_str), len(target_str))
        if check_len == 0:
            continue
        matched = sum(1 for a, b in zip(pred_str, target_str) if a == b)
        total_matched += matched
        total_checked += check_len

    if total_checked == 0:
        return 0.0

    percent = (total_matched / total_checked) * 100
    return round(percent, 2)
