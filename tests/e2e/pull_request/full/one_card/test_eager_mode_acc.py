#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
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
This file test accuracy via LMEval.
It uses local-completions, which interacts with vLLM
through the OAI API with N concurrent connections.
This simulates real work usage of the API and makes
sure that the zmq frontend mp RPC message passing and
AsyncLLMEngine are working correctly.
"""

import lm_eval
import pytest

MODEL_NAMES = ["Qwen/Qwen3-0.6B", "vllm-ascend/DeepSeek-V2-Lite-W8A8"]
NUM_CONCURRENT = 500
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
EXPECTED_VALUES = {"Qwen/Qwen3-0.6B": 0.414, "vllm-ascend/DeepSeek-V2-Lite-W8A8": 0.34}


def run_test(model_name, more_args=None):
    """Run the end to end accuracy test."""

    # NOTE: Do not add any spaces to the string below, as this will cause parameter parsing errors.
    model_args = f"pretrained={model_name},max_model_len=4096,enforce_eager=True"

    if more_args is not None:
        model_args = "{},{}".format(model_args, more_args)

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks="gsm8k",
        batch_size="auto",
    )

    measured_value = results["results"][TASK][FILTER]
    assert model_name in EXPECTED_VALUES, f"Cannot find the expected value for the model {model_name=}"
    expected_value = EXPECTED_VALUES[model_name]
    assert measured_value - RTOL < expected_value and measured_value + RTOL > expected_value, (
        f"Expected: {expected_value} |  Measured: {measured_value}"
    )


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_lm_eval_accuracy(model):
    """Run with the V1 Engine."""
    more_args = None
    run_test(model, more_args)
