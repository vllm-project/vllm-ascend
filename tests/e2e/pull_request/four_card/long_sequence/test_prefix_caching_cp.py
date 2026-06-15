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
"""Qwen3.5 prefix-cache CP guard.

Run `pytest tests/e2e/pull_request/four_card/long_sequence/test_prefix_caching_cp.py`.
"""

from tests.e2e.conftest import VllmRunner

MODEL = "Qwen/Qwen3.5-4B"
MAX_NUM_SEQS = 2

QWEN3_5_PREFIX_MAMBA_PROMPT = (
    "You are reading a compact synthetic operations ledger. "
    "Use only the rows below when answering the final question.\n"
    + "\n".join(
        f"Row {i}: route R{i:03d} moves cargo from zone {i % 11} to zone {(i * 7) % 13}; priority is {i % 5}."
        for i in range(64)
    )
    + "\n"
)

INPUT_PROMPTS = [
    QWEN3_5_PREFIX_MAMBA_PROMPT + "Question: What route is listed in row 17? Answer briefly.",
    QWEN3_5_PREFIX_MAMBA_PROMPT + "Question: What priority is listed in row 42? Answer briefly.",
]


def test_qwen3_5_prefix_cache_with_pcp() -> None:
    with VllmRunner(
        MODEL,
        dtype="float16",
        block_size=128,
        max_model_len=2048,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=2048,
        tensor_parallel_size=1,
        prefill_context_parallel_size=4,
        decode_context_parallel_size=1,
        enforce_eager=True,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        mamba_ssm_cache_dtype="float16",
    ) as vllm_model:
        prefix_cache_outputs = vllm_model.generate_greedy(INPUT_PROMPTS, 8)

    assert len(prefix_cache_outputs) == len(INPUT_PROMPTS)
    for output_ids, output_text in prefix_cache_outputs:
        assert output_ids
        assert output_text
