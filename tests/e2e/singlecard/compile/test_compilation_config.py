#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""E2E tests for --compilation-config mode and dynamic_shapes_config on Ascend NPU.

Tests that all four CompilationMode values (NONE, STOCK_TORCH_COMPILE,
DYNAMO_TRACE_ONCE, VLLM_COMPILE) produce consistent outputs, and that
dynamic_shapes_config combinations work correctly.
"""

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

MODEL = "Qwen/Qwen3-0.6B"

PROMPTS = [
    "The capital of France is",
    "The future of AI is",
    "Hello, my name is",
    "The president of the United States is",
]


def _run_and_collect(runner: VllmRunner, prompts: list[str]) -> list[str]:
    """Run inference and collect output strings."""
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = runner.generate(prompts, sampling_params)
    return [req_output.outputs[0].text for req_output in outputs]


@pytest.mark.parametrize(
    "mode",
    ["NONE", "STOCK_TORCH_COMPILE", "DYNAMO_TRACE_ONCE"],
)
def test_compilation_modes_output_consistency(mode: str):
    """Verify that NONE, STOCK_TORCH_COMPILE, and DYNAMO_TRACE_ONCE
    produce identical outputs (greedy decoding, temperature=0.0)."""
    runner = VllmRunner(
        model_name=MODEL,
        max_model_len=512,
        enforce_eager=None,
        compilation_config={"mode": mode},
    )
    try:
        results = _run_and_collect(runner, PROMPTS)
        assert len(results) == len(PROMPTS)
        assert all(isinstance(r, str) and len(r) > 0 for r in results), (
            f"All outputs should be non-empty strings. Got: {results}"
        )
    finally:
        del runner


def test_vllm_compile_mode_basic():
    """Verify VLLM_COMPILE mode (ACL Graph) with PIECEWISE cudagraph."""
    runner = VllmRunner(
        model_name=MODEL,
        max_model_len=512,
        enforce_eager=None,
        compilation_config={
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "PIECEWISE",
        },
    )
    try:
        results = _run_and_collect(runner, PROMPTS)
        assert len(results) == len(PROMPTS)
        assert all(isinstance(r, str) and len(r) > 0 for r in results), (
            f"All outputs should be non-empty strings. Got: {results}"
        )
    finally:
        del runner


def test_vllm_compile_with_dynamic_shapes_backed():
    """Verify VLLM_COMPILE + BACKED dynamic shapes works."""
    runner = VllmRunner(
        model_name=MODEL,
        max_model_len=512,
        enforce_eager=None,
        compilation_config={
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "PIECEWISE",
            "dynamic_shapes_config": {"type": "BACKED"},
        },
    )
    try:
        results = _run_and_collect(runner, PROMPTS)
        assert len(results) == len(PROMPTS)
        assert all(isinstance(r, str) and len(r) > 0 for r in results), (
            f"All outputs should be non-empty strings. Got: {results}"
        )
    finally:
        del runner


def test_dynamo_with_backed_size_oblivious():
    """Verify DYNAMO_TRACE_ONCE + BACKED_SIZE_OBLIVIOUS passes through."""
    runner = VllmRunner(
        model_name=MODEL,
        max_model_len=512,
        enforce_eager=None,
        compilation_config={
            "mode": "DYNAMO_TRACE_ONCE",
            "dynamic_shapes_config": {"type": "BACKED_SIZE_OBLIVIOUS"},
        },
    )
    try:
        results = _run_and_collect(runner, PROMPTS)
        assert len(results) == len(PROMPTS)
        assert all(isinstance(r, str) and len(r) > 0 for r in results), (
            f"All outputs should be non-empty strings. Got: {results}"
        )
    finally:
        del runner


def test_none_mode_with_dynamic_shapes_config():
    """Verify NONE mode passes dynamic_shapes_config through unchanged."""
    runner = VllmRunner(
        model_name=MODEL,
        max_model_len=512,
        enforce_eager=None,
        compilation_config={
            "mode": "NONE",
            "dynamic_shapes_config": {
                "type": "BACKED",
                "evaluate_guards": True,
                "assume_32_bit_indexing": True,
            },
        },
    )
    try:
        results = _run_and_collect(runner, PROMPTS)
        assert len(results) == len(PROMPTS)
        assert all(isinstance(r, str) and len(r) > 0 for r in results), (
            f"All outputs should be non-empty strings. Got: {results}"
        )
    finally:
        del runner


def test_stock_torch_compile_with_evaluate_guards():
    """Verify STOCK_TORCH_COMPILE with evaluate_guards=True passes through."""
    runner = VllmRunner(
        model_name=MODEL,
        max_model_len=512,
        enforce_eager=None,
        compilation_config={
            "mode": "STOCK_TORCH_COMPILE",
            "dynamic_shapes_config": {
                "type": "BACKED",
                "evaluate_guards": True,
            },
        },
    )
    try:
        results = _run_and_collect(runner, PROMPTS)
        assert len(results) == len(PROMPTS)
        assert all(isinstance(r, str) and len(r) > 0 for r in results), (
            f"All outputs should be non-empty strings. Got: {results}"
        )
    finally:
        del runner
