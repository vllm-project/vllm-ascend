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
import sys
from multiprocessing import Queue

import lm_eval
import pytest
import torch

# pre-trained model path on Hugging Face.
MODEL_NAME = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"]
# Benchmark configuration mapping models to evaluation tasks:
# - Text model: GSM8K (grade school math reasoning)
# - Vision-language model: MMMU Art & Design validation (multimodal understanding)
TASK = {
    "Qwen/Qwen2.5-0.5B-Instruct": "gsm8k",
    "Qwen/Qwen2.5-VL-3B-Instruct": "mmmu_val_art_and_design"
}
# Answer validation requiring format consistency.
FILTER = {
    "Qwen/Qwen2.5-0.5B-Instruct": "exact_match,strict-match",
    "Qwen/Qwen2.5-VL-3B-Instruct": "acc,none"
}
# 3% relative tolerance for numerical accuracy.
RTOL = 0.03
# Baseline accuracy after VLLM optimization.
EXPECTED_VALUE = {
    "Qwen/Qwen2.5-0.5B-Instruct": 0.316,
    "Qwen/Qwen2.5-VL-3B-Instruct": 0.566
}
# Maximum context length configuration for each model.
MAX_MODEL_LEN = {
    "Qwen/Qwen2.5-0.5B-Instruct": 4096,
    "Qwen/Qwen2.5-VL-3B-Instruct": 8192
}
# Model types distinguishing text-only and vision-language models.
MODEL_TYPE = {
    "Qwen/Qwen2.5-0.5B-Instruct": "vllm",
    "Qwen/Qwen2.5-VL-3B-Instruct": "vllm-vlm"
}
# wrap prompts in a chat-style template.
APPLY_CHAT_TEMPLATE = {"vllm": False, "vllm-vlm": True}
# Few-shot examples handling as multi-turn dialogues.
FEWSHOT_AS_MULTITURN = {"vllm": False, "vllm-vlm": True}
# batch_size
BATCH_SIZE = {
    "Qwen/Qwen2.5-0.5B-Instruct": "auto",
    "Qwen/Qwen2.5-VL-3B-Instruct": 1
}

multiprocessing.set_start_method("spawn", force=True)


def run_test(queue, model, max_model_len, model_type):
    try:
        if model_type == "vllm-vlm":
            model_args = (f"pretrained={model},max_model_len={max_model_len},"
                          "tensor_parallel_size=1,dtype=auto,max_images=2")
        else:
            model_args = (f"pretrained={model},max_model_len={max_model_len},"
                          "tensor_parallel_size=1,dtype=auto")
        results = lm_eval.simple_evaluate(
            model=model_type,
            model_args=model_args,
            tasks=TASK[model],
            batch_size=BATCH_SIZE[model],
            apply_chat_template=APPLY_CHAT_TEMPLATE[model_type],
            fewshot_as_multiturn=FEWSHOT_AS_MULTITURN[model_type],
        )
        result = results["results"][TASK[model]][FILTER[model]]
        print("result:", result)
        queue.put(result)
    except Exception as e:
        queue.put(e)
        sys.exit(1)
    finally:
        gc.collect()
        torch.npu.empty_cache()


@pytest.mark.parametrize("model", MODEL_NAME)
@pytest.mark.parametrize("VLLM_USE_V1", ["0", "1"])
def test_lm_eval_accuracy(monkeypatch: pytest.MonkeyPatch, model, VLLM_USE_V1):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", VLLM_USE_V1)
        result_queue: Queue[float] = multiprocessing.Queue()
        p = multiprocessing.Process(target=run_test,
                                    args=(result_queue, model,
                                          MAX_MODEL_LEN[model],
                                          MODEL_TYPE[model]))
        p.start()
        p.join()
        result = result_queue.get()
        if isinstance(result, Exception):
            pytest.fail(f"Subprocess failed with exception: {str(result)}")
        print(result)
        assert (EXPECTED_VALUE[model] - RTOL < result < EXPECTED_VALUE[model] + RTOL), \
            f"Expected: {EXPECTED_VALUE[model]}±{RTOL} | Measured: {result}"
