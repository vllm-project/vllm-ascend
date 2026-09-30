# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# 310P MRV2 e2e smoke for temperature / top-k / top-p post-process.
# Style mirrors MRV1 ``tests/e2e/pull_request/one_card/test_sampler.py``
# (single generate with SamplingParams; no distributional asserts).

from __future__ import annotations

import os
from unittest.mock import patch

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

# CI uses hub id; local verify can override (e.g. /home/weights/Qwen3-0.6B).
QWEN3_06B_MODEL = os.environ.get("QWEN3_06B_MODEL", "Qwen/Qwen3-0.6B")


@wait_until_npu_memory_free(0.7)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_qwen3_mrv2_310p_topk() -> None:
    """MRV1 ``test_qwen3_topk`` counterpart on 310P MRV2.

    MRV1 uses ``temperature=0.0, top_k=50, top_p=0.9`` (vLLM clears top_k/top_p
    when temperature==0). Here temperature>0 so Ascend310PSampler's
    temperature / top-k / top-p path is actually exercised under ACLGraph.
    """
    example_prompts = ["Hello, my name is"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.8, top_k=50, top_p=0.9)
    with VllmRunner(
        QWEN3_06B_MODEL,
        tensor_parallel_size=1,
        enforce_eager=False,
        dtype="float16",
        max_model_len=2048,
        max_num_seqs=8,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=True,
        additional_config={"ascend_compilation_config": {"fuse_norm_quant": False, "enable_npugraph_ex": False}},
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 8],
        },
    ) as vllm_model:
        assert vllm_model.model.llm_engine.vllm_config.use_v2_model_runner
        vllm_model.generate(example_prompts, sampling_params)
