# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from modelscope import snapshot_download  # type: ignore[import-untyped]

import pytest
import vllm
from vllm.assets.image import ImageAsset
from vllm.lora.request import LoRARequest
from vllm.sampling_params import BeamSearchParams
from vllm.utils.torch_utils import set_default_torch_num_threads

from tests.e2e.conftest import (VllmRunner, cleanup_dist_env_and_memory,
                                wait_until_npu_memory_free)
from vllm_ascend.ascend_config import clear_ascend_config
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    "\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "What is in the image?<|im_end|>\n"
    "<|im_start|>assistant\n")


@dataclass
class TestConfig:
    model_path: str
    lora_path: str
    max_num_seqs: int = 2
    max_loras: int = 2
    max_lora_rank: int = 32
    enable_tower_connector_lora: bool = False
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.85
    mm_processor_kwargs: dict[str, int] | None = None
    mm_processor_cache_gb: float = 4

    def __post_init__(self):
        if self.mm_processor_kwargs is None:
            self.mm_processor_kwargs = {
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            }


def run_test(
    llm: vllm.LLM,
    config: TestConfig,
    images: list[ImageAsset],
    expected_outputs: list[str],
    lora_id: int | None = None,
    lora_name: str | None = None,
    temperature: float = 0,
    max_tokens: int = 5,
):
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    inputs = [{
        "prompt": PROMPT_TEMPLATE,
        "multi_modal_data": {
            "image": asset.pil_image
        },
    } for asset in images]

    lora_request = LoRARequest(lora_name if lora_name else str(lora_id),
                               lora_id, config.lora_path)
    outputs = llm.generate(inputs, sampling_params, lora_request=lora_request)
    generated_texts = [output.outputs[0].text.strip() for output in outputs]
    # Validate outputs
    for generated, expected in zip(generated_texts, expected_outputs):
        print(generated)
        assert expected.startswith(generated), (
            f"Generated text {generated} doesn't ")
        f"match expected pattern {expected}"


def run_beam_search_test(
    llm: vllm.LLM,
    config: TestConfig,
    images: list[ImageAsset],
    expected_outputs: list[list[str]],
    lora_id: int | None = None,
    temperature: float = 0,
    beam_width: int = 2,
    max_tokens: int = 5,
):
    beam_search_params = BeamSearchParams(beam_width=beam_width,
                                          max_tokens=max_tokens,
                                          temperature=temperature)

    inputs = [{
        "prompt": PROMPT_TEMPLATE,
        "multi_modal_data": {
            "image": asset.pil_image
        },
    } for asset in images]

    lora_request = LoRARequest(str(lora_id), lora_id, config.lora_path)
    outputs = llm.beam_search(inputs,
                              beam_search_params,
                              lora_request=lora_request)

    for output_obj, expected_outs in zip(outputs, expected_outputs):
        output_texts = [seq.text for seq in output_obj.sequences]
        assert output_texts == expected_outs, (
            f"Generated texts {output_texts} do not match expected {expected_outs}"
        )  # noqa: E501


TEST_IMAGES = [
    ImageAsset("stop_sign"),
    ImageAsset("cherry_blossom"),
]

EXPECTED_OUTPUTS = [
    "A red stop sign stands prominently in the foreground, with a traditional Chinese gate and a black SUV in the background, illustrating a blend of modern and cultural elements.",  # noqa: E501
    "A majestic skyscraper stands tall, partially obscured by a vibrant canopy of cherry blossoms, against a clear blue sky.",  # noqa: E501
]

EXPECTED_OUTPUTS_LANGUAGE = [
    "A stop sign is shown in an Asian city, with buildings and a car in the "
    "background.",
    "The Tokyo Skytree can be seen behind the pink blossoms of the cherry trees.",
]

EXPECTED_OUTPUTS_VISION = [
    "A stop sign in front of oriental buildings.",
    "A tree with pink flowers in front of it and a blue sky behind the flowers.",
]

EXPECTED_OUTPUTS_VISION_NO_CONNECTOR = [
    "A stop sign is located on the street of a Chinese neighborhood.",
    "A closeup shot of the Tokyo Skytree with pink flowers in the foreground.",
]

# NOTE - beam search .text contains the whole text
EXPECTED_BEAM_SEARCH_OUTPUTS = [
    [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is in the image?<|im_end|>\n<|im_start|>assistant\nA majestic skyscraper stands",  # noqa: E501
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is in the image?<|im_end|>\n<|im_start|>assistant\nA majestic tower stands tall",  # noqa: E501
    ],
]

QWEN2VL_MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"
QWEN25VL_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN3VL_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"

# @wait_until_npu_memory_free()
# def test_qwen2vl_lora(qwen2vl_lora_files):
#     """Test Qwen 2.0 VL model with LoRA"""
#     config = TestConfig(model_path=QWEN2VL_MODEL_PATH,
#                         lora_path=qwen2vl_lora_files)

#     # Test with different LoRA IDs
#     with VllmRunner(
#             config.model_path,
#             max_num_seqs=config.max_num_seqs,
#             enable_lora=True,
#             max_loras=config.max_loras,
#             max_lora_rank=config.max_lora_rank,
#             enable_tower_connector_lora=config.enable_tower_connector_lora,
#             gpu_memory_utilization=config.gpu_memory_utilization,
#             mm_processor_kwargs=config.mm_processor_kwargs,
#             mm_processor_cache_gb=config.mm_processor_cache_gb,
#             max_model_len=config.max_model_len,
#     ) as vllm_model:
#         llm = vllm_model.model
#         run_test(llm,
#                  config,
#                  TEST_IMAGES,
#                  expected_outputs=EXPECTED_OUTPUTS,
#                  lora_id=1)

# @wait_until_npu_memory_free()
# def test_qwen2vl_lora_beam_search(qwen2vl_lora_files):
#     """Test Qwen 2.0 VL model with LoRA through beam search."""
#     config = TestConfig(model_path=QWEN2VL_MODEL_PATH,
#                         lora_path=qwen2vl_lora_files)
#     with VllmRunner(
#             config.model_path,
#             max_num_seqs=config.max_num_seqs,
#             enable_lora=True,
#             max_loras=config.max_loras,
#             max_lora_rank=config.max_lora_rank,
#             enable_tower_connector_lora=config.enable_tower_connector_lora,
#             gpu_memory_utilization=config.gpu_memory_utilization,
#             mm_processor_kwargs=config.mm_processor_kwargs,
#             mm_processor_cache_gb=config.mm_processor_cache_gb,
#             max_model_len=config.max_model_len,
#     ) as vllm_model:
#         llm = vllm_model.model

#         # NOTE currently, we only test cherry blossom since stop sign
#         # output is slightly different for v1; - the root cause is likely
#         # independent of the intent of this test, which is to ensure beam
#         # search passes through lora through correctly.
#         run_beam_search_test(
#             llm,
#             config,
#             [ImageAsset("cherry_blossom")],
#             expected_outputs=EXPECTED_BEAM_SEARCH_OUTPUTS,
#             lora_id=1,
#         )

# @wait_until_npu_memory_free()
# def test_qwen25vl_lora(qwen25vl_lora_files):
#     """Test Qwen 2.5 VL model with LoRA"""
#     config = TestConfig(model_path=QWEN25VL_MODEL_PATH,
#                         lora_path=qwen25vl_lora_files)

#     # Test with different LoRA IDs
#     with VllmRunner(
#             config.model_path,
#             max_num_seqs=config.max_num_seqs,
#             enable_lora=True,
#             max_loras=config.max_loras,
#             max_lora_rank=config.max_lora_rank,
#             enable_tower_connector_lora=config.enable_tower_connector_lora,
#             gpu_memory_utilization=config.gpu_memory_utilization,
#             mm_processor_kwargs=config.mm_processor_kwargs,
#             mm_processor_cache_gb=config.mm_processor_cache_gb,
#             max_model_len=config.max_model_len,
#     ) as vllm_model:
#         llm = vllm_model.model
#         # with set_default_torch_num_threads(1):
#         run_test(llm,
#                  config,
#                  TEST_IMAGES,
#                  expected_outputs=EXPECTED_OUTPUTS,
#                  lora_id=1)

# @wait_until_npu_memory_free()
# def test_qwen25vl_vision_lora(qwen25vl_vision_lora_files):
#     config = TestConfig(
#         model_path=QWEN25VL_MODEL_PATH,
#         lora_path=qwen25vl_vision_lora_files,
#         # Currently, tower_connector_lora is incompatible with
#         # the multi-modal processor cache.
#         # TODO: Remove this restriction
#         mm_processor_cache_gb=0,
#         enable_tower_connector_lora=True,
#     )
#     with VllmRunner(
#             config.model_path,
#             max_num_seqs=config.max_num_seqs,
#             enable_lora=True,
#             max_loras=config.max_loras,
#             max_lora_rank=config.max_lora_rank,
#             enable_tower_connector_lora=config.enable_tower_connector_lora,
#             gpu_memory_utilization=config.gpu_memory_utilization,
#             mm_processor_kwargs=config.mm_processor_kwargs,
#             mm_processor_cache_gb=config.mm_processor_cache_gb,
#             max_model_len=config.max_model_len,
#     ) as vllm_model:
#         llm = vllm_model.model

#         run_test(
#             llm,
#             config,
#             TEST_IMAGES,
#             expected_outputs=EXPECTED_OUTPUTS,
#             lora_id=1,
#         )


@wait_until_npu_memory_free()
def test_qwen3vl_vision_lora(qwen3vl_vision_lora_files):
    config = TestConfig(
        model_path=QWEN3VL_MODEL_PATH,
        lora_path=qwen3vl_vision_lora_files,
        # Currently, tower_connector_lora is incompatible with
        # the multi-modal processor cache.
        # TODO: Remove this restriction
        mm_processor_cache_gb=0,
        enable_tower_connector_lora=True,
    )
    with VllmRunner(
            snapshot_download(config.model_path),
            max_num_seqs=config.max_num_seqs,
            enable_lora=True,
            max_loras=config.max_loras,
            max_lora_rank=config.max_lora_rank,
            enable_tower_connector_lora=config.enable_tower_connector_lora,
            gpu_memory_utilization=config.gpu_memory_utilization,
            mm_processor_kwargs=config.mm_processor_kwargs,
            mm_processor_cache_gb=config.mm_processor_cache_gb,
            max_model_len=config.max_model_len,
    ) as vllm_model:
        llm = vllm_model.model

        run_test(
            llm,
            config,
            TEST_IMAGES,
            expected_outputs=EXPECTED_OUTPUTS,
            lora_id=1,
        )


# @wait_until_npu_memory_free()
# def test_qwen2vl_multiple_lora_types(
#     qwen2vl_language_lora_files,
#     qwen2vl_vision_tower_connector_lora_files,
#     qwen2vl_vision_tower_lora_files,
# ):
#     """
#     Test multiple LoRA adapter types (language, vision tower + connector,
#     vision tower only) using the same LLM instance to verify mm_encoder_cache
#     behavior with different LoRA requests.

#     By reusing the same LLM instance across different LoRA requests, we ensure that
#     the multimodal encoder cache correctly manages state transitions between
#     language-only and vision-enabled LoRA adapters.
#     """
#     config = TestConfig(
#         model_path=QWEN2VL_MODEL_PATH,
#         # We'll override the lora_path for each specific test, but need to provide
#         # an initial path for initialization
#         lora_path=qwen2vl_language_lora_files,
#         # Currently, tower_connector_lora is incompatible with
#         # the multi-modal processor cache.
#         # TODO: Remove this restriction
#         mm_processor_cache_gb=0,
#         enable_tower_connector_lora=True,
#     )
#     with VllmRunner(
#             config.model_path,
#             max_num_seqs=config.max_num_seqs,
#             enable_lora=True,
#             max_loras=config.max_loras,
#             max_lora_rank=config.max_lora_rank,
#             enable_tower_connector_lora=config.enable_tower_connector_lora,
#             gpu_memory_utilization=config.gpu_memory_utilization,
#             mm_processor_kwargs=config.mm_processor_kwargs,
#             mm_processor_cache_gb=config.mm_processor_cache_gb,
#             max_model_len=config.max_model_len,
#     ) as vllm_model:
#         llm = vllm_model.model

#         # Test 1: Language-only LoRA adapter
#         config.lora_path = qwen2vl_language_lora_files
#         for lora_id in [1, 2]:
#             run_test(
#                 llm,
#                 config,
#                 TEST_IMAGES,
#                 expected_outputs=EXPECTED_OUTPUTS_LANGUAGE,
#                 lora_id=lora_id,
#                 lora_name="language_only",
#             )

#         # Test 2: Vision tower + connector LoRA adapter
#         config.lora_path = qwen2vl_vision_tower_connector_lora_files
#         for lora_id in [3, 4]:
#             run_test(
#                 llm,
#                 config,
#                 TEST_IMAGES,
#                 expected_outputs=EXPECTED_OUTPUTS_VISION,
#                 lora_id=lora_id,
#                 lora_name="vision_tower_connector",
#             )

#         # Test 3: Vision tower only LoRA adapter (no connector)
#         config.lora_path = qwen2vl_vision_tower_lora_files
#         for lora_id in [5, 6]:
#             run_test(
#                 llm,
#                 config,
#                 TEST_IMAGES,
#                 expected_outputs=EXPECTED_OUTPUTS_VISION_NO_CONNECTOR,
#                 lora_id=lora_id,
#                 lora_name="vision_tower",
#             )
