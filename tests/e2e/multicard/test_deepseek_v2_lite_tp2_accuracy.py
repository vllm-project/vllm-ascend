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
# Adapted from vllm-project/blob/main/tests/entrypoints/llm/test_accuracy.py
#

import gc
import multiprocessing
from multiprocessing import Queue

import lm_eval
import pytest
import torch

# pre-trained model path on Hugging Face.
MODELS = ["deepseek-ai/DeepSeek-V2-Lite"]
# Math reasoning benchmark (Grade School Math 8K).
TASK = "gsm8k"
# Answer validation requiring format consistency.
FILTER = "exact_match,strict-match"
# 3% relative tolerance for numerical accuracy.
RTOL = 0.03
# Baseline accuracy after VLLM optimization.
EXPECTED_VALUE = 0.3843821076573162


def run_test(model_name, queue, more_args=None):
    model_args = f"pretrained={model_name},max_model_len=4096,trust_remote_code=True,tensor_parallel_size=4,enforce_eager=True"
    if more_args is not None:
        model_args = f"{model_args},{more_args}"
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=TASK,
        batch_size="auto",
    )
    result = results["results"][TASK][FILTER]
    print(100 * "*", "\nThe accuracy test result:", result)
    queue.put(result)
    del results
    torch.npu.empty_cache()
    gc.collect()


@pytest.mark.parametrize("model", MODELS)
def test_lm_eval_accuracy(model, monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context():
        result_queue: Queue[float] = multiprocessing.Queue()
        p = multiprocessing.Process(target=run_test,
                                    args=(
                                        model,
                                        result_queue,
                                    ))
        p.start()
        p.join()
        result = result_queue.get()
        assert (EXPECTED_VALUE - RTOL < result < EXPECTED_VALUE + RTOL), \
            f"Expected: {EXPECTED_VALUE}±{RTOL} | Measured: {result}"
