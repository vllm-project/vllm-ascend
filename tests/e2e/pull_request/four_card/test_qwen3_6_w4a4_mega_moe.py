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
"""End-to-end test for the fused W4A4 INT4 MoE "mega" kernel on Qwen3.x-MoE.

Loads the INT4 + block-diagonal-Hadamard checkpoint via the
("W4A4_DYNAMIC", "moe") scheme and verifies coherent generation on 4 cards.
The kernel JIT-compiles with ``bisheng`` against a PTO-ISA toolchain, so the
test self-skips where ``PTO_LIB_PATH`` is not available.
"""

import os

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = "Mocchibird/Qwen3.6-35B-A3B-W4A4-Hadamard64"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.environ.get("PTO_LIB_PATH", "/sources/pto-isa")),
    reason="W4A4 mega kernel requires a PTO-ISA toolchain (set PTO_LIB_PATH)",
)


@wait_until_npu_memory_free()
def test_qwen3_6_w4a4_mega_moe_generation():
    """W4A4 fused mega-MoE end-to-end: the INT4 routed-expert kernel runs every
    MoE layer (no vendor fallback) and produces coherent text."""
    prompts = ["The capital of France is", "List three prime numbers:"]
    with VllmRunner(
        MODEL,
        quantization="ascend",
        tensor_parallel_size=4,
        max_model_len=2048,
        # The INT4 path is fp16-only on 910B and the gated-delta-net graph-compile
        # path is fragile on some CANN versions; eager is the validated config.
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens=32)

    assert len(outputs) == len(prompts)
    # Every prompt yields more than it was given (no empty/garbage output).
    for prompt, (_token_ids, text) in zip(prompts, outputs):
        assert len(text) > len(prompt)
    # Deterministic greedy correctness check on the factual prompt.
    assert "Paris" in outputs[0][1]
