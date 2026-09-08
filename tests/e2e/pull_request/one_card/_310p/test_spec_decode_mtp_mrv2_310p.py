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
"""MRv2 310P MTP smoke (extends MRv1 test_spec_decode_mtp_310p patterns)."""

from __future__ import annotations

import os

import pytest

from tests.e2e.conftest import VllmRunner


def _has_qwen35_4b() -> bool:
    return os.path.isdir("/home/weights/Qwen3.5-4B-W8A8") or os.path.isdir(
        os.path.expanduser("~/.cache/huggingface/hub")
    )


@pytest.mark.skipif(not _has_qwen35_4b(), reason="Qwen3.5-4B weights not available")
@pytest.mark.parametrize("num_speculative_tokens", [1, 2, 3])
def test_qwen3_5_mrv2_mtp_tp1_eager(num_speculative_tokens: int):
    """MRv2 + MTP eager: short greedy decode must be coherent (no garble)."""
    example_prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    model = "/home/weights/Qwen3.5-4B-W8A8" if os.path.isdir("/home/weights/Qwen3.5-4B-W8A8") else "Qwen/Qwen3.5-4B"
    quant_kwargs = {"quantization": "ascend"} if "W8A8" in model else {}
    with VllmRunner(
        model,
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float16",
        max_model_len=2048,
        max_num_seqs=4,
        mamba_ssm_cache_dtype="float16",
        mamba_cache_mode="align",
        skip_mm_profiling=True,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        },
        additional_config={
            "ascend_compilation_config": {
                "fuse_norm_quant": False,
                "enable_npugraph_ex": False,
            }
        },
        **quant_kwargs,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens=16)
        assert len(outputs) == len(example_prompts)
        for _token_ids, text in outputs:
            assert isinstance(text, str) and len(text) > 0
            # Reject classic 310P MTP corruption signatures.
            assert "\ufffd" not in text
            assert "0" * 20 not in text


@pytest.mark.skipif(not os.path.isdir("/home/weights/Qwen3.5-4B-W8A8"), reason="need W8A8")
def test_qwen3_5_mrv2_mtp_tp1_graph_k1():
    """MRv2 + MTP + FULL_DECODE_ONLY capture covering (K+1)*batch."""
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    with VllmRunner(
        "/home/weights/Qwen3.5-4B-W8A8",
        tensor_parallel_size=1,
        enforce_eager=False,
        dtype="float16",
        max_model_len=2048,
        max_num_seqs=4,
        mamba_ssm_cache_dtype="float16",
        mamba_cache_mode="align",
        skip_mm_profiling=True,
        quantization="ascend",
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [2, 8],
        },
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 1,
        },
        additional_config={
            "ascend_compilation_config": {
                "fuse_norm_quant": False,
                "enable_npugraph_ex": False,
            }
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(["Einstein was born in"], max_tokens=24)
        assert len(outputs) == 1
        _token_ids, text = outputs[0]
        assert len(text) > 0
        assert "0" * 20 not in text
