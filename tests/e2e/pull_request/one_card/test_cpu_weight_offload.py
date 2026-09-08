# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""End-to-end tests for CPU weight offloading on Ascend NPU.

Covers the prefetch backend (AscendPrefetchOffloader).
Tests verify that offloading produces the same outputs
as the baseline (no offloading).
"""

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request import utils as e2e_utils
from tests.e2e.pull_request.utils import PROMPTS_SHORT

MODEL = "Qwen/Qwen3-0.6B"
# Default Ascend weight_nz_mode. ND (mode 0) is omitted to keep this file
# near 10 minutes; each Qwen3-0.6B engine is ~1.5-2 min of cold start.
_BASE_KWARGS = {
    "model_name": MODEL,
    "max_model_len": 512,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 256,
    "additional_config": {"weight_nz_mode": 1},
}
_CUDAGRAPH_CAPTURE_SIZES = [1, 2, 4, 8]


def _generate(**runner_kwargs):
    with VllmRunner(**{**_BASE_KWARGS, **runner_kwargs}) as runner:
        return runner.model.generate(
            prompts=PROMPTS_SHORT,
            sampling_params=e2e_utils._LOGPROB_SAMPLING_PARAMS,
        )


def _assert_offload_logprobs(baseline_outputs, offload_outputs, atol: float = 0.0689) -> None:
    decode_atol = 2 * atol
    for prompt_idx, (base_out, offload_out) in enumerate(zip(baseline_outputs, offload_outputs)):
        base_seq = base_out.outputs[0]
        offload_seq = offload_out.outputs[0]

        assert base_seq.logprobs is not None and offload_seq.logprobs is not None, (
            f"logprobs not returned for prompt {prompt_idx}"
        )
        assert len(base_seq.token_ids) == len(offload_seq.token_ids) == 3, (
            f"Expected 3 tokens for prompt {prompt_idx}, "
            f"got baseline={len(base_seq.token_ids)}, offload={len(offload_seq.token_ids)}"
        )

        e2e_utils._check_prefill_token(base_seq, offload_seq, prompt_idx, atol)
        for token_idx in range(1, 3):
            e2e_utils._check_decode_token(base_seq, offload_seq, token_idx, prompt_idx, decode_atol)


@wait_until_npu_memory_free()
def test_prefetch_offload_accuracy() -> None:
    """Eager / graph / selective-MLP prefetch vs one NZ no-offload baseline."""
    baseline_outputs = _generate(enforce_eager=True)

    eager_outputs = _generate(
        enforce_eager=True,
        offload_backend="prefetch",
        offload_group_size=4,
        offload_num_in_group=1,
    )
    _assert_offload_logprobs(baseline_outputs, eager_outputs)

    graph_outputs = _generate(
        offload_backend="prefetch",
        offload_group_size=4,
        offload_num_in_group=1,
        cudagraph_capture_sizes=_CUDAGRAPH_CAPTURE_SIZES,
    )
    _assert_offload_logprobs(baseline_outputs, graph_outputs)

    selective_outputs = _generate(
        enforce_eager=True,
        offload_backend="prefetch",
        offload_group_size=8,
        offload_num_in_group=2,
        offload_prefetch_step=1,
        offload_params={"gate_up_proj", "down_proj"},
    )
    _assert_offload_logprobs(baseline_outputs, selective_outputs)
