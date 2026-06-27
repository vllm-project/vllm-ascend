#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import huggingface_hub
import pytest
from huggingface_hub import snapshot_download as hf_snapshot_download
from vllm.assets.image import ImageAsset

from tests.e2e.conftest import VllmRunner, qwen_prompt, wait_until_npu_memory_free


@pytest.mark.e2e_features("multimodal", "full_decode_only", "mtp", "mamba_ssm")
@pytest.mark.e2e_model("Qwen/Qwen3.5-0.8B")
@wait_until_npu_memory_free()
def test_mamba_ssm_multimodal_reasoning_mtp_full_decode_only():
    """Verify Mamba/SSM Multimodal with MTP and full decode only."""
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    img_questions = [
        "What is the content of this image?",
        "Describe the content of this image in detail.",
        "What's in the image?",
        "Where is this image taken?",
    ]
    images = [image] * len(img_questions)
    prompts = qwen_prompt(img_questions)

    model_path = hf_snapshot_download(
        "Qwen/Qwen3.5-0.8B",
        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
    )
    with VllmRunner(
        model_path,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_batched_tokens=1024,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [2, 4, 6, 8],
        },
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 1,
        },
    ) as runner:
        outputs = runner.generate_greedy(
            prompts=prompts,
            images=images,
            max_tokens=64,
        )

        assert len(outputs) == len(prompts)
        for _, output_str in outputs:
            assert output_str, "Generated output should not be empty."
