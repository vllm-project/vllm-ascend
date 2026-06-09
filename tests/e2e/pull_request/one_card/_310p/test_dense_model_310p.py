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


from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.model_utils import check_outputs_equal

QWEN3_5_PREFIX_MAMBA_PROMPT = (
    "You are reading a compact synthetic operations ledger. "
    "Use only the rows below when answering the final question.\n"
    + "\n".join(
        f"Row {i}: route R{i:03d} moves cargo from zone {i % 11} to zone {(i * 7) % 13}; priority is {i % 5}."
        for i in range(64)
    )
    + "\n"
)

QWEN3_5_PREFIX_MAMBA_PROMPTS = [
    QWEN3_5_PREFIX_MAMBA_PROMPT + "Question: What route is listed in row 17? Answer briefly.",
    QWEN3_5_PREFIX_MAMBA_PROMPT + "Question: What priority is listed in row 42? Answer briefly.",
]


def _generate_qwen3_5_prefix_mamba_outputs(enable_prefix_caching: bool) -> list[tuple[list[int], str]]:
    outputs: list[tuple[list[int], str]] = []

    if enable_prefix_caching:
        with VllmRunner(
            "Qwen/Qwen3.5-4B",
            tensor_parallel_size=1,
            enforce_eager=True,
            dtype="float16",
            max_model_len=2048,
            max_num_batched_tokens=2048,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            mamba_ssm_cache_dtype="float16",
        ) as vllm_model:
            for prompt in QWEN3_5_PREFIX_MAMBA_PROMPTS:
                outputs.extend(vllm_model.generate_greedy([prompt], max_tokens=8))
    else:
        with VllmRunner(
            "Qwen/Qwen3.5-4B",
            tensor_parallel_size=1,
            enforce_eager=True,
            dtype="float16",
            max_model_len=2048,
            max_num_batched_tokens=2048,
            enable_prefix_caching=False,
            mamba_ssm_cache_dtype="float16",
        ) as vllm_model:
            for prompt in QWEN3_5_PREFIX_MAMBA_PROMPTS:
                outputs.extend(vllm_model.generate_greedy([prompt], max_tokens=8))
    return outputs


def test_qwen3_dense_tp1_fp16():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
        "Qwen/Qwen3-8B",
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float16",
        max_model_len=16384,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@wait_until_npu_memory_free(0.7)
def test_qwen3_dense_tp1_fp16_aclgraph():
    example_prompts = [
        "Hello, my name is",
    ] * 8
    max_tokens = 2
    with VllmRunner(
        "Qwen/Qwen3-8B",
        tensor_parallel_size=1,
        dtype="float16",
        max_num_seqs=16,
        max_model_len=16384,
        gpu_memory_utilization=0.80,
        additional_config={"ascend_compilation_config": {"fuse_norm_quant": False}},
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
        },
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_qwen3_dense_tp1_w8a8():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
        "vllm-ascend/Qwen3-8B-W8A8",
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float16",
        quantization="ascend",
        max_model_len=16384,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_qwen3_5_dense_tp1_fp16():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
        "Qwen/Qwen3.5-4B",
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float16",
        max_model_len=16384,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@wait_until_npu_memory_free(0.7)
def test_qwen3_5_dense_prefix_mamba_cache_tp1_fp16():
    prefix_cache_outputs = _generate_qwen3_5_prefix_mamba_outputs(enable_prefix_caching=True)
    no_prefix_cache_outputs = _generate_qwen3_5_prefix_mamba_outputs(enable_prefix_caching=False)

    assert len(prefix_cache_outputs) == len(no_prefix_cache_outputs) == len(QWEN3_5_PREFIX_MAMBA_PROMPTS)
    check_outputs_equal(
        outputs_0_lst=no_prefix_cache_outputs,
        outputs_1_lst=prefix_cache_outputs,
        name_0="no_prefix_cache_outputs",
        name_1="prefix_cache_outputs",
    )


@wait_until_npu_memory_free(0.7)
def test_qwen3_5_dense_tp1_fp16_aclgraph():
    example_prompts = [
        "Hello, my name is",
    ] * 8
    max_tokens = 2
    with VllmRunner(
        "Qwen/Qwen3.5-4B",
        tensor_parallel_size=1,
        dtype="float16",
        max_num_seqs=16,
        max_model_len=16384,
        gpu_memory_utilization=0.80,
        additional_config={"ascend_compilation_config": {"fuse_norm_quant": False}},
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
        },
        mamba_ssm_cache_dtype="float16",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
