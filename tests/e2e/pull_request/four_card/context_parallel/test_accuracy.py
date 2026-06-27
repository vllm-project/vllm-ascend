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
"""PCP/DCP long-sequence accuracy guards.

Run `pytest tests/e2e/pull_request/four_card/context_parallel/test_accuracy.py`.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from PIL import Image
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, get_e2e_model, wait_until_npu_memory_free

MAX_NUM_SEQS = 4
E2E_ROOT = Path(__file__).resolve().parents[3]
QWEN_IMAGE_PATH = E2E_ROOT / "prompts" / "qwen.png"

FULL_DECODE_GRAPH = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [MAX_NUM_SEQS],
}

COMMON_PROMPTS = [
    "The capital of France is",
    "Hello, my name is Tom, I am",
    "The president of United States is",
]

DSV2_PROMPTS = [
    "The president of the United States is",
    "The capital of France is",
]

DSV2_PCP_GOLDEN = [
    "The president of the United States is a man who is not only a liar, but",
    "The capital of France is Paris.\nThe currency of France is the Euro",
]

DSV2_DCP_GOLDEN = [
    "The president of the United States is a man who is not only a liar, but",
    "The capital of France is Paris.\nThe currency of France is the Euro",
]

QWEN3_GOLDEN = [
    "The capital of France is Paris. The capital of",
    "Hello, my name is Tom, I am 12 years old",
    "The president of United States is the head of state and",
]

QWEN3_NEXT_GOLDEN = [
    "The capital of France is Paris. The capital of",
    "Hello, my name is Tom, I am 12 years old",
    "The president of United States is the head of state and",
]

DSV3_2_GOLDEN = [
    "The capital of France isbearerhaloce梗",
    "Hello, my name is Tom, I am" + "ERIC slicpacelike Chop",
    "The president of United States is平行astra unbehetroni",
]

DSV3_2_GOLDEN_BACKUPS = (
    [
        "The capital of France isbearerhaloce梗",
        "Hello, my name is Tom, I am" + "ERIC slicpacelike Chop",
        "The president of United States isoint054 Rund959arki",
    ],
    [
        "The capital of France isbearerdenomorthal",
        "Hello, my name is Tom, I am" + "ERIC slicpacelike Chop",
        "The president of United States is平行astra unbehetroni",
    ],
    [
        "The capital of France isbearerdenomorthal",
        "Hello, my name is Tom, I am" + "ERIC slicpacelike Chop",
        "The president of United States isoint054 Rund959arki",
    ],
)

DEEPSEEK_MTP3_GOLDEN = [
    "The capital of France is Salmonella团团 elsewhereッγκ",
    "Hello, my name is Tom, I amEiSlowukt Analysis sprouts",
    "The president of United States is Salmonella团团 elsewhereッγκ",
]

DEEPSEEK_V4_PROMPTS = [
    "Hello, my name is",
    "What is the meaning of life?",
]

DEEPSEEK_V4_GOLDEN = ["Hello, my name is {name} and I", 'What is the meaning of life?",\n    "What is']


@dataclass(frozen=True)
class AccuracyCase:
    name: str
    prompts: Sequence[str]
    expected_outputs: Sequence[str] | Sequence[Sequence[str]]
    max_tokens: int
    runner_kwargs: dict[str, Any]


def _run_accuracy_case(model: str, case: AccuracyCase) -> None:
    with VllmRunner(model, **case.runner_kwargs) as runner:
        outputs = runner.generate_greedy(list(case.prompts), case.max_tokens)

    if isinstance(case.expected_outputs[0], str):
        expected_outputs = cast(Sequence[str], case.expected_outputs)
        match_outputs_with_goldens(outputs, expected_outputs)
    else:
        # If multiple expected output sets are provided, the output is considered correct if it matches any of the sets.
        multi_expected_outputs = cast(Sequence[Sequence[str]], case.expected_outputs)
        tries = []
        for expected in multi_expected_outputs:
            try:
                match_outputs_with_goldens(outputs, expected)
            except AssertionError as exc:
                tries.append(f"Output did not match expected set:\n{exc}")
            else:
                break
        if len(tries) == len(multi_expected_outputs):
            failure_details = "\n\n".join(tries)
            raise AssertionError(f"Output did not match any of the expected output sets:\n{failure_details}")


def match_outputs_with_goldens(outputs: list[tuple[list[int], str]], goldens: Sequence[str]) -> None:
    """Helper function to compare output with golden output, ignoring whitespace differences."""
    outputs_str: Sequence[str] = [output[1] for output in outputs]
    assert len(outputs_str) == len(goldens)
    for index, (output, golden) in enumerate(zip(outputs_str, goldens)):
        assert isinstance(output, str) and isinstance(golden, str), "Both output and golden must be strings"
        assert output and golden, "Output and golden should not be empty"
        assert output.strip() == golden.strip()


@patch.dict(
    os.environ,
    {
        "HCCL_BUFFSIZE": "768",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "OMP_NUM_THREADS": "1",
        "OMP_PROC_BIND": "false",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
@pytest.mark.e2e_features("multimodal", "gqa", "tp", "pcp", "full_decode_only")
@pytest.mark.e2e_model("Qwen/Qwen3-VL-8B-Instruct")
def test_qwen3_vl_multimodal_pcp_accuracy_guard(request) -> None:
    image = Image.open(QWEN_IMAGE_PATH).convert("RGB")
    single_image_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "Describe this image in detail.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    multi_image_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "Compare these two images and describe similarities.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = [
        {
            "prompt": single_image_prompt,
            "multi_modal_data": {"image": image},
        },
        {
            "prompt": multi_image_prompt,
            "multi_modal_data": {"image": [image, image.copy()]},
        },
    ]
    sampling_params = SamplingParams(max_tokens=16, temperature=0.0)
    model = get_e2e_model(request)
    with VllmRunner(
        model,
        enforce_eager=False,
        max_model_len=4096,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=1,
        max_num_batched_tokens=1024,
        block_size=128,
        limit_mm_per_prompt={"image": 2},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [MAX_NUM_SEQS],
        },
    ) as runner:
        outputs = runner.model.generate(inputs, sampling_params=sampling_params)

    assert len(outputs) == len(inputs)
    for output in outputs:
        assert output.outputs and output.outputs[0].text.strip()


DSV2_COMMON_KWARGS: dict[str, Any] = {
    "max_model_len": 1024,
    "max_num_seqs": MAX_NUM_SEQS,
    "max_num_batched_tokens": 1024,
    "enable_expert_parallel": True,
    "enable_chunked_prefill": True,
    "enable_prefix_caching": True,
    "block_size": 128,
    "quantization": "ascend",
    "compilation_config": FULL_DECODE_GRAPH,
    "additional_config": {"enable_flashcomm1": True},
}

DSV2_PARALLEL_CASES = [
    AccuracyCase(
        name="dsv2_pcp_dcp_full_features",
        prompts=DSV2_PROMPTS,
        expected_outputs=DSV2_PCP_GOLDEN,
        max_tokens=10,
        runner_kwargs={
            **DSV2_COMMON_KWARGS,
            "tensor_parallel_size": 2,
            "prefill_context_parallel_size": 2,
            "decode_context_parallel_size": 2,
            "cp_kv_cache_interleave_size": 128,
            "long_prefill_token_threshold": 4,
        },
    ),
    AccuracyCase(
        name="dsv2_pcp_only_full_features",
        prompts=DSV2_PROMPTS,
        expected_outputs=DSV2_PCP_GOLDEN,
        max_tokens=10,
        runner_kwargs={
            **DSV2_COMMON_KWARGS,
            "tensor_parallel_size": 2,
            "prefill_context_parallel_size": 2,
            "decode_context_parallel_size": 1,
            "cp_kv_cache_interleave_size": 128,
            "long_prefill_token_threshold": 4,
        },
    ),
    AccuracyCase(
        name="dsv2_dcp_only_full_features",
        prompts=DSV2_PROMPTS,
        expected_outputs=DSV2_DCP_GOLDEN,
        max_tokens=10,
        runner_kwargs={
            **DSV2_COMMON_KWARGS,
            "tensor_parallel_size": 2,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 2,
            "long_prefill_token_threshold": 64,
            "compilation_config": {
                **FULL_DECODE_GRAPH,
                "pass_config": {"enable_sp": True},
            },
        },
    ),
]

FULL_FEATURE_MODEL_CASES = [
    AccuracyCase(
        name="qwen3_pcp_dcp_full_features",
        prompts=COMMON_PROMPTS,
        expected_outputs=QWEN3_GOLDEN,
        max_tokens=5,
        runner_kwargs={
            "max_model_len": 1024,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_num_batched_tokens": 1024,
            "tensor_parallel_size": 2,
            "prefill_context_parallel_size": 2,
            "decode_context_parallel_size": 1,
            "enable_expert_parallel": True,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "block_size": 128,
            "quantization": "ascend",
            "long_prefill_token_threshold": 4,
            "compilation_config": FULL_DECODE_GRAPH,
            "additional_config": {"enable_flashcomm1": True},
        },
    ),
    AccuracyCase(
        name="qwen3_next_pcp_dcp_full_features",
        prompts=COMMON_PROMPTS,
        expected_outputs=QWEN3_NEXT_GOLDEN,
        max_tokens=5,
        runner_kwargs={
            "enforce_eager": True,
            "max_model_len": 1024,
            "tensor_parallel_size": 2,
            "prefill_context_parallel_size": 2,
            "decode_context_parallel_size": 1,
            "max_num_batched_tokens": 1024,
            "enable_expert_parallel": True,
            # TODO(qcs): We should set `long_prefill_token_threshold` to 4
            # when chunked prefill with PCP is stable.
            "long_prefill_token_threshold": 128,
            "gpu_memory_utilization": 0.8,
            "block_size": 128,
            # FlashComm1 is disabled for qwen3_next until the PCP decode path is fixed.
            "additional_config": {"enable_flashcomm1": False},
        },
    ),
    AccuracyCase(
        name="dsv3_2_pcp_dcp_full_features",
        prompts=COMMON_PROMPTS,
        # TODO(qcs): Remove multi-expected_outputs after the first request output is stable.
        expected_outputs=(DSV3_2_GOLDEN, *DSV3_2_GOLDEN_BACKUPS),
        max_tokens=5,
        runner_kwargs={
            "max_model_len": 1024,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_num_batched_tokens": 1024,
            "tensor_parallel_size": 2,
            "prefill_context_parallel_size": 2,
            "decode_context_parallel_size": 2,
            "enable_expert_parallel": True,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.2,
            "cp_kv_cache_interleave_size": 128,
            "block_size": 128,
            "quantization": "ascend",
            # TODO(qcs): We should set `long_prefill_token_threshold` to 4
            # when chunked prefill with PCP is stable.
            "long_prefill_token_threshold": 128,
            "compilation_config": FULL_DECODE_GRAPH,
            "additional_config": {"enable_flashcomm1": True},
            # graph_mode is disabled for dsv32 until the PCP gatherv3 out of index issue is fixed.
            "enforce_eager": True,
        },
    ),
    AccuracyCase(
        name="deepseek_mtp3_pcp_dcp_full_features",
        prompts=COMMON_PROMPTS,
        expected_outputs=DEEPSEEK_MTP3_GOLDEN,
        max_tokens=5,
        runner_kwargs={
            "max_model_len": 1024,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_num_batched_tokens": 1024,
            "tensor_parallel_size": 2,
            "prefill_context_parallel_size": 2,
            "decode_context_parallel_size": 2,
            "enable_expert_parallel": True,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "block_size": 128,
            "long_prefill_token_threshold": 4,
            "speculative_config": {
                "method": "mtp",
                "num_speculative_tokens": 3,
            },
            "compilation_config": FULL_DECODE_GRAPH,
            "additional_config": {"enable_flashcomm1": True},
        },
    ),
    AccuracyCase(
        name="deepseek_v4_w4a8_dsa_cp_full_features",
        prompts=DEEPSEEK_V4_PROMPTS,
        expected_outputs=DEEPSEEK_V4_GOLDEN,
        max_tokens=5,
        runner_kwargs={
            "max_model_len": 8192,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 4096,
            "dtype": "auto",
            "tensor_parallel_size": 4,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "enable_expert_parallel": True,
            "gpu_memory_utilization": 0.9,
            "quantization": "ascend",
            "tokenizer_mode": "deepseek_v4",
            "block_size": 128,
            "compilation_config": {
                "cudagraph_mode": "FULL_DECODE_ONLY",
            },
            "additional_config": {
                "enable_flashcomm1": True,
                "enable_dsa_cp": True,
            },
        },
    ),
]


FULL_FEATURE_SKIP_REASONS: dict[str, str] = {
    "deepseek_mtp3_pcp_dcp_full_features": "Temporarily skip MTP with PCP/DCP until the token layout issue is fixed.",
}


@patch.dict(
    os.environ,
    {
        "HCCL_BUFFSIZE": "768",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
@pytest.mark.parametrize("case", DSV2_PARALLEL_CASES, ids=lambda case: case.name)
@pytest.mark.e2e_features("moe", "tp", "ep", "pcp", "dcp", "full_decode_only", "w8a8", "flash_comm1")
@pytest.mark.e2e_model("vllm-ascend/DeepSeek-V2-Lite-W8A8")
def test_dsv2_lite_parallel_config_accuracy(request, case: AccuracyCase) -> None:
    model = get_e2e_model(request)
    _run_accuracy_case(model, case)


@patch.dict(
    os.environ,
    {
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
@pytest.mark.parametrize(
    "index, case",
    [
        pytest.param(
            i,
            c,
            marks=pytest.mark.skip(reason=reason) if (reason := FULL_FEATURE_SKIP_REASONS.get(c.name)) else [],
            id=c.name,
        )
        for i, c in enumerate(FULL_FEATURE_MODEL_CASES)
    ],
)
@pytest.mark.e2e_features(
    "moe",
    "tp",
    "ep",
    "pcp",
    "dcp",
    "full_decode_only",
    "eager_mode",
    "w8a8",
    "w4a8",
    "chunked_prefill",
    "prefix_caching",
    "sfa",
    "dsa",
    "flash_comm1",
    "mtp",
)
@pytest.mark.e2e_model(
    "vllm-ascend/Qwen3-30B-A3B-W8A8",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning",
    "wemaster/deepseek_mtp_main_random_bf16",
    "gdydems/DeepSeek-V4-Flash-w4a8-mtp",
)
def test_models_pcp_dcp_full_feature_accuracy(request, index: int, case: AccuracyCase) -> None:
    """Cross-mode parametrized test spanning both eager and graph cases.

    This single test function is parametrized over multiple models whose
    ``AccuracyCase`` runner_kwargs mix ``enforce_eager=True`` (eager_mode)
    and ``compilation_config=FULL_DECODE_GRAPH`` (full_decode_only).
    Because ``@pytest.mark.e2e_features`` is per-function, both modes are
    listed together even though they are mutually exclusive per-case.
    """
    model = get_e2e_model(request, index)
    _run_accuracy_case(model, case)
