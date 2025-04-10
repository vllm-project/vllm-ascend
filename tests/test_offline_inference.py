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
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/test_offline_inference.py`.
"""
import os

import pytest
import vllm  # noqa: F401
from conftest import VllmRunner
from model_utils import PROMPT_TEMPLATES
from vllm.assets.image import ImageAsset

import vllm_ascend  # noqa: F401

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]
MULTIMODALITY_MODELS = ["Qwen/Qwen2.5-VL-3B-Instruct"]

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
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
                    enforce_eager=False,
                    gpu_memory_utilization=0.7) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@pytest.mark.multinpu
@pytest.mark.parametrize("model, distributed_executor_backend", [
    ("Qwen/QwQ-32B", "mp"),
])
def test_models_distributed(vllm_runner, model: str,
                            distributed_executor_backend: str) -> None:
    example_prompts = [
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
        "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.",
        "Compare and contrast artificial intelligence with human intelligence in terms of processing information.",
    ]
    dtype = "half"
    max_tokens = 5
    with vllm_runner(
            model,
            dtype=dtype,
            tensor_parallel_size=4,
            distributed_executor_backend=distributed_executor_backend,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@pytest.fixture(params=list(PROMPT_TEMPLATES.keys()))
def prompt_template(request):
    return PROMPT_TEMPLATES[request.param]


@pytest.mark.parametrize("model", MULTIMODALITY_MODELS)
def test_multimodal(vllm_runner, model: str, prompt_template) -> None:
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
                     max_num_seqs=5,
                     mm_processor_kwargs={
                         "min_pixels": 28 * 28,
                         "max_pixels": 1280 * 28 * 28,
                         "fps": 1,
                     }) as vllm_model:
        vllm_model.generate_greedy(prompts=prompts,
                                   images=images,
                                   max_tokens=64)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
