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
#
"""End-to-end NPU support test for DiffusionGemma-26B-A4B-it.

DiffusionGemma is a discrete text-diffusion model (block-diffusion, fixed
256-token canvas, MoE 8/128 experts) served through vLLM's V2 model runner
(``VLLM_USE_V2_MODEL_RUNNER=1``). This test exercises the offline ``LLM(...)``
path on Ascend NPU with tensor parallelism.

Two paths are covered:

* ``max_num_seqs=1`` (homogeneous single-canvas batches) - the baseline that
  already generates coherent text (~94.5% GSM8K eager).
* ``max_num_seqs=2`` (the batching path THE PR FIXES) - guards the mixed-phase
  canvas-batching regression where a mixed batch (some requests in the
  causal/commit phase, others in the bidirectional denoise phase) previously
  collapsed to a single batch-wide attention phase. This asserts batched
  generation still produces coherent, non-empty output.

The test requires 2 NPUs (``tensor_parallel_size=2``) and the model weights on
disk; it SKIPS cleanly when either is unavailable so it is safe to collect
off-hardware.
"""

import os
from unittest.mock import patch

import pytest
import torch

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

# Model weights location. Overridable via env for CI mirrors, matching the
# ``os.getenv(...)`` convention used elsewhere in tests/e2e/.
DIFFUSION_GEMMA_MODEL_PATH = os.getenv(
    "DIFFUSION_GEMMA_MODEL_PATH",
    "/data/public_models/diffusiongemma-26B-A4B-it",
)

# Small arithmetic prompt with a deterministic greedy answer. DiffusionGemma
# denoises this to coherent text such as "17 + 26 = 43".
ARITHMETIC_PROMPT = "17 + 26 ="
EXPECTED_SUBSTRING = "43"

# DiffusionGemma denoises a fixed 256-token canvas; keep max_model_len modest.
MAX_MODEL_LEN = 512
MAX_OUTPUT_TOKENS = 64
TENSOR_PARALLEL_SIZE = 2

# Skip gates (evaluated at collection time so the file is safe off-hardware).
_MODEL_MISSING = not os.path.isdir(DIFFUSION_GEMMA_MODEL_PATH)


def _available_npus() -> int:
    """NPU count, tolerant of environments without torch_npu / a driver."""
    try:
        return torch.npu.device_count()
    except Exception:  # noqa: BLE001 - collection must never crash off-hardware
        return 0


_NOT_ENOUGH_NPUS = _available_npus() < TENSOR_PARALLEL_SIZE


@pytest.mark.skipif(
    _MODEL_MISSING,
    reason=f"DiffusionGemma weights not found at {DIFFUSION_GEMMA_MODEL_PATH}.",
)
@pytest.mark.skipif(
    _NOT_ENOUGH_NPUS,
    reason="DiffusionGemma e2e test requires at least 2 NPUs (tensor_parallel_size=2).",
)
@pytest.mark.parametrize(
    "max_num_seqs",
    [
        pytest.param(1, id="single_canvas"),
        # The batching path THE PR FIXES: mixed-phase canvas batching.
        pytest.param(2, id="canvas_batching"),
    ],
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free()
def test_diffusion_gemma_generation(max_num_seqs: int) -> None:
    """DiffusionGemma generates coherent text on NPU with canvas batching."""
    # For max_num_seqs=2 send two prompts so the mixed-phase batching path
    # (the PR's headline fix) is actually exercised in one forward batch.
    prompts = [ARITHMETIC_PROMPT] * max_num_seqs

    with VllmRunner(
        DIFFUSION_GEMMA_MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=max_num_seqs,
        # DiffusionGemma is text-only in this test; disable image inputs.
        limit_mm_per_prompt={"image": 0},
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            prompts=prompts,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

    assert len(outputs) == len(prompts)

    for _, output_str in outputs:
        # Non-empty coherent generation on every request in the (possibly
        # batched) canvas. Under the pre-fix regression a mixed batch silently
        # dropped the causal mask and degraded output; we require the greedy
        # arithmetic answer to survive on each request.
        assert output_str, "DiffusionGemma produced empty output."
        assert EXPECTED_SUBSTRING in output_str, (
            f"Expected '{EXPECTED_SUBSTRING}' in greedy output for prompt '{ARITHMETIC_PROMPT}', got: {output_str!r}"
        )
