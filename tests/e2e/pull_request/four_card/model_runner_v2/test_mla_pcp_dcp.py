# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 dense MLA PCP+DCP correctness on four Ascend NPUs."""

import os
from unittest.mock import patch

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = "vllm-ascend/DeepSeek-V2-Lite-W8A8"
PROMPTS = [
    "Combined context parallel inference must preserve every token. " * 48 + "The result is",
    "Tensor and prefill parallel ranks must agree while a long prompt is chunked. " * 9 + "Therefore",
]


def _generate(*, pcp_size: int, dcp_size: int) -> list[tuple[list[int], str]]:
    with VllmRunner(
        MODEL,
        tensor_parallel_size=2,
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
def test_mla_pcp_dcp_full_axis_matches_tp_baseline_tokens() -> None:
    baseline = _generate(pcp_size=1, dcp_size=1)
    combined_outputs = _generate(pcp_size=2, dcp_size=4)

    assert [token_ids for token_ids, _ in combined_outputs] == [token_ids for token_ids, _ in baseline]
