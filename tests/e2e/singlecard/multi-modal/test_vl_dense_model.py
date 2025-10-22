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
#

import pytest
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODELS = [
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
]

mm_messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role":
        "user",
        "content": [
            {
                "type": "image",
                "image":
                "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png",
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
            },
            {
                "type": "text",
                "text": "Please provide a detailed description of this image"
            },
        ],
    },
]


def process_mm_messages(model: str):
    # Process text inputs
    processor = AutoProcessor.from_pretrained(model)
    prompt = processor.apply_chat_template(
        mm_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Process image inputs
    image_inputs, _, _ = process_vision_info(mm_messages,
                                             return_video_kwargs=True)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }
    return llm_inputs


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
def test_models_with_aclgraph(
    model: str,
    max_tokens: int,
) -> None:
    llm_inputs = process_mm_messages(model)
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    with VllmRunner(
            model,
            max_model_len=1024,
            limit_mm_per_prompt={"image": 10},
            enforce_eager=False,
    ) as runner:
        vllm_aclgraph_outputs = runner.model.generate(llm_inputs,
                                                      sampling_params)

    with VllmRunner(
            model,
            max_model_len=1024,
            limit_mm_per_prompt={"image": 10},
            enforce_eager=True,
    ) as runner:
        vllm_eager_outputs = runner.model.generate(llm_inputs, sampling_params)

    vllm_aclgraph_outputs_list = []
    for output in vllm_aclgraph_outputs:
        vllm_aclgraph_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    vllm_eager_outputs_list = []
    for output in vllm_eager_outputs:
        vllm_eager_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=vllm_aclgraph_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="vllm_aclgraph_outputs",
    )
