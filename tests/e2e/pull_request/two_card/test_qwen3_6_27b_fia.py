#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import os
from unittest.mock import patch

from vllm.assets.image import ImageAsset

from tests.e2e.conftest import VllmRunner, qwen_prompt, wait_until_npu_memory_free

MODEL = "Qwen/Qwen3.6-27B"


def _get_test_images():
    """Exercise varying encoder sequence lengths within a multimodal batch."""
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    return [image.resize((size, size)) for size in (224, 448, 672, 896)]


@patch.dict(os.environ, {"HCCL_BUFFSIZE": "1024"})
@wait_until_npu_memory_free()
def test_qwen3_6_27b_multimodel_fia_eager():
    """Verify multimodal generation with FIA op and eager mode."""
    images = _get_test_images()
    questions = [
        "What is the content of this image?",
        "Describe the content of this image in detail.",
        "What's in the image?",
        "Where is this image taken?",
    ]

    prompts = qwen_prompt(questions)

    with VllmRunner(
        MODEL,
        max_model_len=4096,
        tensor_parallel_size=2,
        language_model_only=False,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            prompts=prompts,
            images=images,
            max_tokens=64,
        )

    assert len(outputs) == len(prompts)
    assert all(text for _, text in outputs)


@patch.dict(
    os.environ,
    {"HCCL_BUFFSIZE": "1024", "VLLM_USE_V2_MODEL_RUNNER": "1"},
)
@wait_until_npu_memory_free()
def test_qwen3_6_27b_multimodel_encoder_acl_graph_mrv2():
    """Verify MRV2 encoder ACL graph while decoder graph capture is disabled."""
    images = _get_test_images()
    questions = [
        "What is the content of this image?",
        "Describe the content of this image in detail.",
        "What's in the image?",
        "Where is this image taken?",
    ]

    prompts = qwen_prompt(questions)

    with VllmRunner(
        MODEL,
        max_model_len=4096,
        tensor_parallel_size=2,
        language_model_only=False,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        compilation_config={
            "cudagraph_mm_encoder": True,
            "cudagraph_mode": "NONE",
            "encoder_cudagraph_max_vision_items_per_batch": 4,
            # The four test images require 1920 encoder tokens in total. One
            # fitting budget is sufficient to validate capture and replay and
            # avoids capturing nine unrelated graphs during this E2E test.
            "encoder_cudagraph_token_budgets": [2048],
        },
    ) as vllm_model:
        graph_stats_before = vllm_model.model.llm_engine.collective_rpc("get_encoder_cudagraph_stats")
        outputs = vllm_model.generate_greedy(
            prompts=prompts,
            images=images,
            max_tokens=64,
        )

        graph_stats_after = vllm_model.model.llm_engine.collective_rpc("get_encoder_cudagraph_stats")

    assert len(outputs) == len(prompts)
    assert all(text for _, text in outputs)
    assert len(graph_stats_before) == len(graph_stats_after) == 2
    assert all(stats is not None for stats in graph_stats_before + graph_stats_after)
    assert all(stats["manager"] == "EncoderAclGraphManager" for stats in graph_stats_after)
    assert all(stats["captured"] for stats in graph_stats_after)
    assert all(
        after["graph_hits"] > before["graph_hits"] for before, after in zip(graph_stats_before, graph_stats_after)
    ), f"Encoder ACL graph did not replay on every TP rank: {graph_stats_before=} {graph_stats_after=}"
