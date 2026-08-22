# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from PIL import Image

from tests.e2e.conftest import VllmRunner

HYBRID_MODELS = {
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
}

FULL_DECODE_ONLY_GRAPH = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [1, 2, 4],
}


def hybrid_runner_kwargs(model: str) -> dict:
    """Qwen3.5 hybrid models require fp16 Mamba/GDN state on 310P."""
    if model in HYBRID_MODELS:
        return {"mamba_ssm_cache_dtype": "float16"}
    return {}


def get_test_image() -> Image.Image:
    """Build a deterministic image without relying on an external test asset."""
    return Image.new("RGB", (224, 224), color=(32, 96, 160))


def get_test_prompts():
    return ["<|image_pad|>Describe this image in detail."]


def run_vl_model_test(
    model_name: str,
    tensor_parallel_size: int,
    max_tokens: int,
    dtype: str = "float16",
    enforce_eager: bool = True,
    **runner_kwargs,
):
    image = get_test_image()
    images = [image]
    prompts = get_test_prompts()

    with VllmRunner(
        model_name,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        dtype=dtype,
        **runner_kwargs,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens, images=images)
        follow_up_outputs = vllm_model.generate_greedy(prompts, max_tokens, images=images)

    assert outputs and all(output[0] for output in outputs)
    assert follow_up_outputs and all(output[0] for output in follow_up_outputs)
