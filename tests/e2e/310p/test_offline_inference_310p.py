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
import vllm  # noqa: F401
from vllm import SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

import vllm_ascend  # noqa: F401
from tests.e2e.conftest import VllmRunner

MODELS = ["/home/data/Qwen3-8B", "/home/data/Qwen2.5-7B-Instruct"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("max_tokens", [5])
def test_models(model: str, dtype: str, max_tokens: int) -> None:
    example_prompts = [
        "The future of AI is",
    ]

    with VllmRunner(model,
                    tensor_parallel_size=1,
                    dtype=dtype,
                    max_model_len=2048,
                    enforce_eager=True,
                    compilation_config={
                        "custom_ops":
                        ["none", "+rms_norm", "+rotary_embedding"]
                    }) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


VL_MODELS = ["/home/c00939328/vllm-ascend-csx/Qwen2.5-VL-3B-Instruct"]

def build_vl_req(model: str, img_uri: str):
    processor = AutoProcessor.from_pretrained(model)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": img_uri,
             "min_pixels": 224 * 224, "max_pixels": 1280 * 28 * 28},
            {"type": "text", "text": "Please provide a detailed description of this image"},
        ]},
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

@pytest.mark.parametrize("model", VL_MODELS)
@pytest.mark.parametrize("dtype", ["float16"])
def test_vl_model_with_image(model: str, dtype: str) -> None:
    img_uri = "/home/c00939328/image1.jpg"

    req = build_vl_req(model, img_uri)
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)

    with VllmRunner(
        model,
        tensor_parallel_size=1,
        dtype=dtype,
        max_model_len=16384,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none", "+rms_norm", "+rotary_embedding"]},
        limit_mm_per_prompt={"image": 10},
    ) as vllm_model:
        outs = vllm_model.generate([req], sampling_params)
        assert outs[0].outputs[0].text.strip() != ""
