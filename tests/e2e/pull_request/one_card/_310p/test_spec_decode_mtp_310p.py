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

"""310P MTP e2e: MRv1 baseline + MRv2 eager smoke (1-card CI safe)."""

import os
from unittest.mock import patch

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

# CI uses hub id; local verify can override (e.g. /home/weights/Qwen3.5-4B-W8A8).
QWEN35_MTP_MODEL = os.environ.get("QWEN35_MTP_MODEL", "Qwen/Qwen3.5-4B")
QWEN35_MTP_QUANTIZATION = os.environ.get("QWEN35_MTP_QUANTIZATION")  # e.g. "ascend"


def _quant_kw():
    return {"quantization": QWEN35_MTP_QUANTIZATION} if QWEN35_MTP_QUANTIZATION else {}


def test_qwen3_5_mtp_tp1_eager():
    """MRv1 baseline (no V2 runner env)."""
    with VllmRunner(
        QWEN35_MTP_MODEL,
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float16",
        max_model_len=2048,
        mamba_ssm_cache_dtype="float16",
        speculative_config={
            "method": "qwen3_5_mtp",
            "num_speculative_tokens": 1,
        },
        **_quant_kw(),
    ) as vllm_model:
        vllm_model.generate_greedy(["Hello, my name is"], max_tokens=8)


@wait_until_npu_memory_free()
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_qwen3_5_mtp_mrv2_tp1_eager():
    """MRv2 MTP eager smoke.

    FULL_DECODE_ONLY is covered by local/nightly runs: 1-card PR CI OOMs during
    target+draft ACLGraph capture (Engine core init fails with empty Failed core
    proc(s)).
    """
    with VllmRunner(
        QWEN35_MTP_MODEL,
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float16",
        max_model_len=2048,
        mamba_ssm_cache_dtype="float16",
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 1,
        },
        **_quant_kw(),
    ) as vllm_model:
        assert vllm_model.model.llm_engine.vllm_config.use_v2_model_runner
        vllm_model.generate_greedy(["Hello, my name is"], max_tokens=8)
