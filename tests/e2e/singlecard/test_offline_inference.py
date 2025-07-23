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

Run `pytest tests/test_offline_inference.py`.
"""
import os
from unittest.mock import patch

import pytest
import vllm  # noqa: F401
from modelscope import snapshot_download  # type: ignore[import-untyped]
from vllm import SamplingParams
from vllm.assets.image import ImageAsset

import vllm_ascend  # noqa: F401
from tests.e2e.conftest import VllmRunner

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B-Base",
]
MULTIMODALITY_MODELS = ["Qwen/Qwen2.5-VL-3B-Instruct"]

QUANTIZATION_MODELS = [
    "vllm-ascend/Qwen2.5-0.5B-Instruct-W8A8",
]
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half", "float16"])
@pytest.mark.parametrize("max_tokens", [5])
def test_models(model: str, dtype: str, max_tokens: int) -> None:
    # 5042 tokens for gemma2
    # gemma2 has alternating sliding window size of 4096
    # we need a prompt with more than 4096 tokens to test the sliding window
    prompt = "The following numbers of the sequence " + ", ".join(
        str(i) for i in range(1024)) + " are:"
    example_prompts = [prompt]

    with VllmRunner(model,
                    max_model_len=8192,
                    dtype=dtype,
                    enforce_eager=True,
                    gpu_memory_utilization=0.7) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@pytest.mark.parametrize("model", QUANTIZATION_MODELS)
@pytest.mark.parametrize("max_tokens", [5])
def test_quantization_models(model: str, max_tokens: int) -> None:
    prompt = "The following numbers of the sequence " + ", ".join(
        str(i) for i in range(1024)) + " are:"
    example_prompts = [prompt]

    # NOTE: Using quantized model repo id from modelscope encounters an issue,
    # this pr (https://github.com/vllm-project/vllm/pull/19212) fix the issue,
    # after it is being merged, there's no need to download model explicitly.
    model_path = snapshot_download(model)

    with VllmRunner(model_path,
                    max_model_len=8192,
                    enforce_eager=True,
                    dtype="auto",
                    gpu_memory_utilization=0.7,
                    quantization="ascend") as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@pytest.mark.parametrize("model", MULTIMODALITY_MODELS)
def test_multimodal(model, prompt_template, vllm_runner):
    image = ImageAsset("cherry_blossom") \
        .pil_image.convert("RGB")
    img_questions = [
        "What is the content of this image?",
        "Describe the content of this image in detail.",
        "What's in the image?",
        "Where is this image taken?",
    ]
    images = [image] * len(img_questions)
    prompts = prompt_template(img_questions)
    with vllm_runner(model,
                     max_model_len=4096,
                     mm_processor_kwargs={
                         "min_pixels": 28 * 28,
                         "max_pixels": 1280 * 28 * 28,
                         "fps": 1,
                     }) as vllm_model:
        vllm_model.generate_greedy(prompts=prompts,
                                   images=images,
                                   max_tokens=64)


@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION": "1"})
def test_models_topk() -> None:
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(max_tokens=5,
                                     temperature=0.0,
                                     top_k=50,
                                     top_p=0.9)

    with VllmRunner("Qwen/Qwen2.5-0.5B-Instruct",
                    max_model_len=8192,
                    dtype="float16",
                    enforce_eager=True,
                    gpu_memory_utilization=0.7) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)
