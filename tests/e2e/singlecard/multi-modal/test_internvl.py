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

import os

# Set spawn method before any torch/NPU imports to avoid fork issues
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

import pytest
from vllm.assets.image import ImageAsset

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODELS = [
    "OpenGVLab/InternVL2-8B",
    "OpenGVLab/InternVL2_5-8B",
    "OpenGVLab/InternVL3-8B",
    "OpenGVLab/InternVL3_5-8B",
]


@pytest.mark.parametrize("model", MODELS)
def test_internvl_basic(model: str):
    """Test basic InternVL2 inference with single image."""
    # Load test image
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")

    # InternVL uses chat template format
    # Format: <|im_start|>user\n<image>\nQUESTION<|im_end|>\n<|im_start|>assistant\n
    questions = [
        "What is the content of this image?",
        "Describe this image in detail.",
    ]

    # Build prompts with InternVL2 chat template
    prompts = [
        f"<|im_start|>user\n<image>\n{q}<|im_end|>\n<|im_start|>assistant\n"
        for q in questions
    ]
    images = [image] * len(prompts)

    outputs = {}
    for enforce_eager, mode in [(False, "eager"), (True, "graph")]:
        with VllmRunner(
                model,
                max_model_len=8192,
                limit_mm_per_prompt={"image": 4},
                enforce_eager=enforce_eager,
                dtype="bfloat16",
        ) as vllm_model:
            generated_outputs = vllm_model.generate_greedy(
                prompts=prompts,
                images=images,
                max_tokens=128,
            )

            assert len(generated_outputs) == len(prompts), \
                f"Expected {len(prompts)} outputs, got {len(generated_outputs)} in {mode} mode"

            for i, (_, output_str) in enumerate(generated_outputs):
                assert output_str, \
                    f"{mode.capitalize()} mode output {i} should not be empty. Prompt: {prompts[i]}"
                assert len(output_str.strip()) > 0, \
                    f"{mode.capitalize()} mode Output {i} should have meaningful content"

            outputs[mode] = generated_outputs

    eager_outputs = outputs["eager"]
    graph_outputs = outputs["graph"]

    check_outputs_equal(outputs_0_lst=eager_outputs,
                        outputs_1_lst=graph_outputs,
                        name_0="eager mode",
                        name_1="graph mode")
