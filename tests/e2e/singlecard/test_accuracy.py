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

import lm_eval
import pytest

MODEL_KWARGS = [
    {
        "pretrained": "Qwen/Qwen3-0.6B",
        "tensor_parallel_size": 1,
        "trust_remote_code": True,
        "max_model_len": 8192,
    },
    {
        "pretrained": "vllm-ascend/DeepSeek-V2-Lite-W8A8",
        "tensor_parallel_size": 1,
        "quantization": "ascend",
        "trust_remote_code": True,
        "max_model_len": 8192,
    },
]

MODEL_KWARGS_FULL_FRAPH = [
    {
        "pretrained": "Qwen/Qwen3-0.6B",
        "tensor_parallel_size": 1,
        "trust_remote_code": True,
        "max_model_len": 8192,
        "compilation_config": {
            "cudagraph_capture_sizes": [4, 8, 32, 64],
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
    },
    {
        "pretrained": "vllm-ascend/DeepSeek-V2-Lite-W8A8",
        "tensor_parallel_size": 1,
        "quantization": "ascend",
        "trust_remote_code": True,
        "max_model_len": 8192,
        "compilation_config": {
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
    },
]

MODEL_KWARGS_NPUGRAPH_EX = [
    {
        "pretrained": "Qwen/Qwen3-0.6B",
        "tensor_parallel_size": 1,
        "trust_remote_code": True,
        "max_model_len": 8192,
        "compilation_config": {
            "cudagraph_capture_sizes": [4, 8, 32, 64],
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
        "additional_config": {
            "enable_npugraph_ex": True
        }
    },
    {
        "pretrained": "vllm-ascend/DeepSeek-V2-Lite-W8A8",
        "tensor_parallel_size": 1,
        "quantization": "ascend",
        "trust_remote_code": True,
        "max_model_len": 8192,
        "compilation_config": {
            "cudagraph_capture_sizes": [4, 8, 32, 64],
            "cudagraph_mode": "FULL_DECODE_ONLY"
        },
        "additional_config": {
            "enable_npugraph_ex": True
        },
    },
]

TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
EXPECTED_VALUES = {
    "Qwen/Qwen3-0.6B": 0.42,
    "vllm-ascend/DeepSeek-V2-Lite-W8A8": 0.35,
}


def run_test(model_name: str, model_args: dict):
    """Run the end to end accuracy test."""
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks="gsm8k",
        batch_size="200",
        device="npu",
        random_seed=0,
    )

    measured_value = results["results"][TASK][FILTER]
    assert model_name in EXPECTED_VALUES, (
        f"Cannot find the expected value for the model {model_name=}")
    print(measured_value)
    expected_value = EXPECTED_VALUES[model_name]
    assert (measured_value - RTOL < expected_value
            and measured_value + RTOL > expected_value
            ), f"Expected: {expected_value} |  Measured: {measured_value}"


@pytest.mark.parametrize("model_args", MODEL_KWARGS)
def test_lm_eval_accuracy_with_piecewise(model_args):
    run_test(model_args["pretrained"], model_args)


@pytest.mark.parametrize("model_args", MODEL_KWARGS_FULL_FRAPH)
def test_lm_eval_accuracy_with_full_decode_only(model_args, monkeypatch):
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    run_test(model_args["pretrained"], model_args)


@pytest.mark.parametrize("model_args", MODEL_KWARGS_NPUGRAPH_EX)
def test_lm_eval_with_fullgraph_npugraph_ex(model_args, monkeypatch):
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    run_test(model_args["pretrained"], model_args)
