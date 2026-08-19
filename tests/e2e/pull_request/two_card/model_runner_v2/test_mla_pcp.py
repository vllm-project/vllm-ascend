# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 MLA PCP correctness on two Ascend NPUs.

Run `pytest -sv tests/e2e/pull_request/two_card/model_runner_v2/test_mla_pcp.py`.
"""

import os
from unittest.mock import patch

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = "vllm-ascend/DeepSeek-V2-Lite-W8A8"
PROMPTS = [
    "Context parallel inference must preserve every token. " * 48 + "The result is",
    "A short request can begin decoding while a longer request is still being prefetched. " * 7 + "Therefore",
]


def _generate(*, pcp_size: int, dcp_size: int = 1) -> list[tuple[list[int], str]]:
    with VllmRunner(
        MODEL,
        tensor_parallel_size=1,
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        distributed_executor_backend="mp",
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_model_len=1024,
        max_num_batched_tokens=256,
        max_num_seqs=len(PROMPTS),
        quantization="ascend",
    ) as runner:
        return runner.generate_greedy(PROMPTS, max_tokens=16)


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_mla_pcp_matches_single_rank_for_uneven_chunked_prefills() -> None:
    baseline = _generate(pcp_size=1)
    pcp_outputs = _generate(pcp_size=2)

    assert [token_ids for token_ids, _ in pcp_outputs] == [token_ids for token_ids, _ in baseline]


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_mla_pcp_dcp_equal_axis_matches_baseline_tokens() -> None:
    baseline = _generate(pcp_size=1, dcp_size=1)
    combined_outputs = _generate(pcp_size=2, dcp_size=2)

    assert [token_ids for token_ids, _ in combined_outputs] == [token_ids for token_ids, _ in baseline]
