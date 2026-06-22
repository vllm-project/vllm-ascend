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
"""DeepSeek-V2-Lite prefix-cache CP guard.

Run `pytest tests/e2e/pull_request/four_card/context_parallel/test_prefix_caching_cp.py`.
"""

import os
from unittest.mock import patch

from tests.e2e.conftest import VllmRunner

# TODO(qcs): We should use Qwen3.5 for this test when it's fixed and available,
# so that we can test the hybrid kv cache with both full attention and linear attention.
MODEL = "vllm-ascend/DeepSeek-V2-Lite-W8A8"
MAX_NUM_SEQS = 2
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

DSV2_LITE_PREFIX_PROMPT = (
    "You are reading a compact synthetic operations ledger. "
    "Use only the rows below when answering the final question.\n"
    + "\n".join(
        f"Row {i}: route R{i:03d} moves cargo from zone {i % 11} to zone {(i * 7) % 13}; priority is {i % 5}."
        for i in range(64)
    )
    + "\n"
)

INPUT_PROMPTS = [
    DSV2_LITE_PREFIX_PROMPT + "Question: What route is listed in row 17? Answer briefly.",
    DSV2_LITE_PREFIX_PROMPT + "Question: What priority is listed in row 42? Answer briefly.",
]


@patch.dict(os.environ, THREAD_ENV)
def test_dsv2_lite_prefix_cache_with_pcp() -> None:
    with VllmRunner(
        MODEL,
        block_size=128,
        max_model_len=2048,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=2048,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=1,
        enforce_eager=True,
        enable_expert_parallel=True,
        enable_prefix_caching=True,
        quantization="ascend",
    ) as vllm_model:
        prefix_cache_outputs = vllm_model.generate_greedy(INPUT_PROMPTS, 8)

    assert len(prefix_cache_outputs) == len(INPUT_PROMPTS)
    for output_ids, output_text in prefix_cache_outputs:
        assert output_ids
        assert output_text
