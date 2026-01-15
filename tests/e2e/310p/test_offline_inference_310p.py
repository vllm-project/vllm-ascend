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
import pytest
from vllm.assets.image import ImageAsset

from tests.e2e.conftest import VllmRunner


@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("max_tokens", [5])
def test_llm_models(dtype: str, max_tokens: int) -> None:
    example_prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]

    with VllmRunner("Qwen/Qwen3-0.6B",
                    tensor_parallel_size=1,
                    dtype=dtype,
                    max_model_len=2048,
                    enforce_eager=True) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_multimodal_vl():
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")

    img_questions = [
        "What is the content of this image?",
        "Describe the content of this image in detail.",
        "What's in the image?",
        "Where is this image taken?",
    ]

    images = [image] * len(img_questions)
    placeholder = "<|image_pad|>"
    prompts = [
        ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
         f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
         f"{q}<|im_end|>\n<|im_start|>assistant\n") for q in img_questions
    ]

    with VllmRunner("Qwen/Qwen2.5-VL-3B-Instruct",
                    mm_processor_kwargs={
                        "min_pixels": 28 * 28,
                        "max_pixels": 1280 * 28 * 28,
                        "fps": 1,
                    },
                    max_model_len=8192,
                    enforce_eager=True,
                    limit_mm_per_prompt={"image": 1}) as vllm_model:
        outputs = vllm_model.generate_greedy(
            prompts=prompts,
            images=images,
            max_tokens=64,
        )

        assert len(outputs) == len(prompts)

        for _, output_str in outputs:
            assert output_str, "Generated output should not be empty."
