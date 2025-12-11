#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from tests.e2e.model_utils import check_outputs_equal
from tests.e2e.conftest import VllmRunner
from vllm import SamplingParams
import pytest
import os

# Set spawn method before any torch/NPU imports to avoid fork issues
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

from vllm.assets.image import ImageAsset


MODELS = ["DeepSeek-ai/DeepSeek-OCR",]


@pytest.mark.parametrize("model", MODELS)
def test_deepseek_ocr(model: str):
    # Load test image:waiting to modify
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")

    # DeepSeek-OCR uses chat template format
    # Format: <image>\nQUESTION\n
    questions = ["Convert the document to markdown.",]

    # Build prompts with DeepSeek-OCR chat template
    prompts = [
        f"<image>\n{q}\n"
        for q in questions
    ]
    images = [image] * len(prompts)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        skip_special_tokens=False,
    )

    outputs = {}
    
    with VllmRunner(
            model,
            max_model_len=8192,
            enforce_eager=False,
            dtype="bfloat16",
    ) as vllm_model:
        generated_outputs = vllm_model.generate(
            prompts=prompts,
            images=images,
            sampling_params=sampling_params
        )

        assert len(generated_outputs) == len(prompts), \
            f"Expected {len(prompts)} outputs, got {len(generated_outputs)} in graph mode"

        for i, (_, outputs_list) in enumerate(generated_outputs):

            assert len(outputs_list) > 0, f"No outputs generated for prompt {i}"
        
