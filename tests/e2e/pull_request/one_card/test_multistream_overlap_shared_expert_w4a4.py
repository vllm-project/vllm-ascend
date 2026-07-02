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
"""
Compare the outputs of vLLM with multistream_overlap_shared_expert
enabled and disabled for W4A4_MXFP4 quantized models.

This exercises the W4A4 4-stage multistream overlap path (the MXFP4 branch
in ``_forward_shared_experts``), which uses small ops
(``npu_dynamic_mx_quant`` + ``act_fn`` + matmul) instead of the fused
``npu_dequant_swiglu_quant`` used by the W8A8 path. It verifies that:
  1. multistream ON (eager)  == baseline (multistream OFF, eager)
  2. multistream ON (aclgraph) == baseline (multistream OFF, eager)

W4A4_MXFP4 model weights may not be publicly available, so this test is
skipped by default. To run it:

    RUN_W4A4_E2E=1 VLLM_ASCEND_W4A4_MODEL=<model_id_or_path> \
    pytest tests/e2e/pull_request/one_card/test_multistream_overlap_shared_expert_w4a4.py -v
"""

import os

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

# W4A4_MXFP4 quantized model. Override via env var with a local path or HF id.
DEFAULT_W4A4_MODEL = "vllm-ascend/DeepSeek-V2-Lite-W4A4-MXFP4"
W4A4_MODEL = os.environ.get("VLLM_ASCEND_W4A4_MODEL", DEFAULT_W4A4_MODEL)

# Skipped unless the user explicitly opts in (W4A4 weights may not be public).
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_W4A4_E2E") != "1",
    reason="W4A4_MXFP4 e2e test needs a W4A4-quantized model; "
    "set RUN_W4A4_E2E=1 and VLLM_ASCEND_W4A4_MODEL=<model> to enable.",
)


@pytest.mark.parametrize("max_tokens", [32])
def test_models_with_multistream_overlap_shared_expert_w4a4(max_tokens: int) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    # 1. multistream ON + eager
    with VllmRunner(
        W4A4_MODEL,
        max_model_len=1024,
        enforce_eager=True,
        cudagraph_capture_sizes=[4, 8, 16, 32],
        additional_config={
            "multistream_overlap_shared_expert": True,
        },
        quantization="ascend",
    ) as runner:
        ms_eager_outputs = runner.model.generate(prompts, sampling_params)

    # 2. multistream ON + aclgraph
    with VllmRunner(
        W4A4_MODEL,
        max_model_len=1024,
        cudagraph_capture_sizes=[4, 8, 16, 32],
        additional_config={
            "multistream_overlap_shared_expert": True,
        },
        quantization="ascend",
    ) as runner:
        ms_aclgraph_outputs = runner.model.generate(prompts, sampling_params)

    # 3. baseline: multistream OFF + eager
    with VllmRunner(
        W4A4_MODEL,
        max_model_len=1024,
        enforce_eager=True,
        cudagraph_capture_sizes=[4, 8, 16, 32],
        quantization="ascend",
    ) as runner:
        baseline_outputs = runner.model.generate(prompts, sampling_params)

    def to_list(outputs):
        return [(o.outputs[0].index, o.outputs[0].text) for o in outputs]

    baseline = to_list(baseline_outputs)

    check_outputs_equal(
        outputs_0_lst=baseline,
        outputs_1_lst=to_list(ms_eager_outputs),
        name_0="baseline_eager",
        name_1="ms_eager",
    )

    check_outputs_equal(
        outputs_0_lst=baseline,
        outputs_1_lst=to_list(ms_aclgraph_outputs),
        name_0="baseline_eager",
        name_1="ms_aclgraph",
    )
