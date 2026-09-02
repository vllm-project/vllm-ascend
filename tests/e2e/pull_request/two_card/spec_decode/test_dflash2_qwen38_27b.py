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
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run "pytest tests/e2e/pull_request/two_card/spec_decode/test_dflash2_qwen38_27b.py".
"""

import json
import os
from unittest.mock import patch

import pytest
from vllm.config import CompilationConfig

from tests.e2e.pull_request.utils import _run_speculative_decoding

MODELS = ["UploadWeight/Qwen3.8-27B"]
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize(
    ("expected_acceptance_length", "num_speculative_tokens", "additional_config"),
    [
        pytest.param(
            3.33,
            7,
            {"enable_cpu_binding": True},
            id="dflash-qwen38-27b",
        ),
    ],
)
@patch.dict(
    os.environ,
    {
        "OMP_NUM_THREADS": "10",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_BUFFSIZE": "1024",
        "TASK_QUEUE_ENABLE": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "LCCL_DETERMINISTI": "1",
        "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "HCCL_DETERMINISTIC": "true",
        "CLOSE_MATMUL_K_SHIFT": "1",
    },
)
def test_qwen38_27b_dflash_acceptance_tp2(
    model_name,
    expected_acceptance_length,
    num_speculative_tokens,
    additional_config,
):
    # The config.json file of the weight does not support sliding window,
    # which may cause accuracy problems.
    draft_model_path = "UploadWeight/Qwen3.8-27B-DFlash2"
    config_path = os.path.join(draft_model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        if "layer_types" in config and isinstance(config["layer_types"], list):
            config["layer_types"] = ["full_attention"] * len(config["layer_types"])
        config["sliding_window"] = None
        config["use_sliding_window"] = False
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    _run_speculative_decoding(
        model_name=model_name,
        speculative_config={
            "method": "dflash",
            "model": "UploadWeight/Qwen3.8-27B-DFlash2",
            "num_speculative_tokens": num_speculative_tokens,
            "enforce_eager": True,
        },
        expected_acceptance_length=expected_acceptance_length,
        runner_kwargs={
            "tensor_parallel_size": 2,
            "max_model_len": 8096, 
            "compilation_config": CompilationConfig(cudagraph_mode="NONE"),
            "additional_config": additional_config,
        },
        is_moe=False,
    )