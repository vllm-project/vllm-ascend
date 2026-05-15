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
#
"""Light tests for quick feature coverage.

Coverage:
- MoE inference with TP, EP, EPLB, and graph capture.
- Multimodal reasoning with PP and graph capture.
"""

import json

import requests
from vllm.assets.image import ImageAsset
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, VllmRunner, qwen_prompt, wait_until_npu_memory_free


@wait_until_npu_memory_free()
def test_moe_tp_ep_eplb_full_graph():
    """Verify MoE serving with TP, EP, EPLB, and graph capture."""
    model = "Qwen/Qwen3-30B-A3B"
    port = get_open_port()
    env_dict = {
        "DYNAMIC_EPLB": "true",
        "HCCL_BUFFSIZE": "1024",
    }
    server_args = [
        "--max_model_len",
        "8192",
        "--tensor_parallel_size",
        "2",
        "--enable_expert_parallel",
        "--port",
        str(port),
        "--compilation-config",
        json.dumps({"cudagraph_capture_sizes": [8]}),
        "--additional-config",
        json.dumps(
            {
                "eplb_config": {
                    "dynamic_eplb": True,
                    "expert_heat_collection_interval": 100,
                    "algorithm_execution_interval": 20,
                    "num_redundant_experts": 2,
                }
            }
        ),
    ]

    with RemoteOpenAIServer(model, server_args, server_port=port, auto_port=False, env_dict=env_dict) as server:
        response = requests.post(
            server.url_for("v1", "completions"),
            json={
                "model": model,
                "prompt": "What is deeplearning?",
                "max_tokens": 400,
                "temperature": 0.0,
                "top_p": 1.0,
                "n": 1,
            },
            timeout=600,
        )
        response.raise_for_status()
        output = response.json()

        assert output["choices"][0]["text"]


@wait_until_npu_memory_free()
def test_multimodal_reasoning_pp_full_graph():
    """Verify multimodal generation with PP and graph capture."""
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")

    img_questions = [
        "What is the content of this image?",
        "Describe the content of this image in detail.",
        "What's in the image?",
        "Where is this image taken?",
    ]

    images = [image] * len(img_questions)
    prompts = qwen_prompt(img_questions)

    with VllmRunner(
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        pipeline_parallel_size=2,
        max_model_len=4096,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        hf_overrides={"text_config": {"architectures": ["Qwen3MoeForCausalLM"]}},
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4],
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            prompts=prompts,
            images=images,
            max_tokens=64,
        )

        assert len(outputs) == len(prompts)

        for _, output_str in outputs:
            assert output_str, "Generated output should not be empty."
