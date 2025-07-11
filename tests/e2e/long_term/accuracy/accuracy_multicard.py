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
import signal
import subprocess
import sys
import time
from multiprocessing import Queue

import lm_eval
import pytest
import requests
import torch

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
COMPLETIONS_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/completions"

# pre-trained model path on Hugging Face.
# Qwen/Qwen2.5-0.5B-Instruct: accuracy test for DP.
# Qwen/Qwen3-30B-A3B: accuracy test for EP and ETP.
# deepseek-ai/DeepSeek-V2-Lite: accuracy test for TP.
MODEL_NAME = ["Qwen/Qwen3-30B-A3B", "deepseek-ai/DeepSeek-V2-Lite"]

# Benchmark configuration mapping models to evaluation tasks:
# - Text model: GSM8K (grade school math reasoning)
# - Vision-language model: MMMU Art & Design validation (multimodal understanding)
TASK = {
    "Qwen/Qwen2.5-0.5B-Instruct": "gsm8k",
    "Qwen/Qwen3-30B-A3B": "gsm8k",
    "deepseek-ai/DeepSeek-V2-Lite": "gsm8k"
}
# Answer validation requiring format consistency.
FILTER = {
    "Qwen/Qwen2.5-0.5B-Instruct": "exact_match,strict-match",
    "Qwen/Qwen3-30B-A3B": "exact_match,strict-match",
    "deepseek-ai/DeepSeek-V2-Lite": "exact_match,strict-match"
}
# 3% relative tolerance for numerical accuracy.
RTOL = 0.03
# Baseline accuracy after VLLM optimization.
EXPECTED_VALUE = {
    "Qwen/Qwen2.5-0.5B-Instruct": 0.316,
    "Qwen/Qwen3-30B-A3B": 0.888,
    "deepseek-ai/DeepSeek-V2-Lite": 0.375
}
# Maximum context length configuration for each model.
MAX_MODEL_LEN = {
    "Qwen/Qwen2.5-0.5B-Instruct": 4096,
    "Qwen/Qwen3-30B-A3B": 4096,
    "deepseek-ai/DeepSeek-V2-Lite": 4096
}
# Model types distinguishing text-only and vision-language models.
MODEL_TYPE = {
    "Qwen/Qwen2.5-0.5B-Instruct": "vllm",
    "Qwen/Qwen3-30B-A3B": "vllm",
    "deepseek-ai/DeepSeek-V2-Lite": "vllm"
}
# wrap prompts in a chat-style template.
APPLY_CHAT_TEMPLATE = {
    "Qwen/Qwen2.5-0.5B-Instruct": False,
    "Qwen/Qwen3-30B-A3B": False,
    "deepseek-ai/DeepSeek-V2-Lite": False
}
# Few-shot examples handling as multi-turn dialogues.
FEWSHOT_AS_MULTITURN = {
    "Qwen/Qwen2.5-0.5B-Instruct": False,
    "Qwen/Qwen3-30B-A3B": False,
    "deepseek-ai/DeepSeek-V2-Lite": False
}
# MORE_ARGS extra CLI args per model
MORE_ARGS = {
    "Qwen/Qwen2.5-0.5B-Instruct":
    None,
    "Qwen/Qwen3-30B-A3B":
    "tensor_parallel_size=4,enable_expert_parallel=True,enforce_eager=True",
    "deepseek-ai/DeepSeek-V2-Lite":
    "tensor_parallel_size=4,trust_remote_code=True,enforce_eager=True"
}

multiprocessing.set_start_method("spawn", force=True)


def run_test(queue, model, max_model_len, model_type, more_args):
    try:
        if model_type == "vllm-vlm":
            model_args = (f"pretrained={model},max_model_len={max_model_len},"
                          "dtype=auto,max_images=2")
        else:
            model_args = (f"pretrained={model},max_model_len={max_model_len},"
                          "dtype=auto")
        if more_args is not None:
            model_args = f"{model_args},{more_args}"
        results = lm_eval.simple_evaluate(
            model=model_type,
            model_args=model_args,
            tasks=TASK[model],
            batch_size="auto",
            apply_chat_template=APPLY_CHAT_TEMPLATE[model],
            fewshot_as_multiturn=FEWSHOT_AS_MULTITURN[model],
        )
        result = results["results"][TASK[model]][FILTER[model]]
        print("result:", result)
        queue.put(result)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        queue.put(error_msg)
        sys.exit(1)
    finally:
        gc.collect()
        torch.npu.empty_cache()


@pytest.mark.parametrize("model", MODEL_NAME)
def test_lm_eval_accuracy(monkeypatch: pytest.MonkeyPatch, model):
    with monkeypatch.context():
        result_queue: Queue[float] = multiprocessing.Queue()
        p = multiprocessing.Process(target=run_test,
                                    args=(result_queue, model,
                                          MAX_MODEL_LEN[model],
                                          MODEL_TYPE[model], MORE_ARGS[model]))
        p.start()
        p.join()
        result = result_queue.get()
        print(result)
        assert (EXPECTED_VALUE[model] - RTOL < result < EXPECTED_VALUE[model] + RTOL), \
            f"Expected: {EXPECTED_VALUE[model]}±{RTOL} | Measured: {result}"


@pytest.mark.parametrize("max_tokens", [10])
@pytest.mark.parametrize("model", ["Qwen/Qwen2.5-0.5B-Instruct"])
def test_lm_eval_accuracy_dp(model, max_tokens):
    log_file = open("accuracy_pd.log", "a+")
    cmd = [
        "vllm", "serve", model, "--max_model_len", "4096",
        "--tensor_parallel_size", "2", "--data_parallel_size", "2"
    ]
    server_proc = subprocess.Popen(cmd,
                                   stdout=log_file,
                                   stderr=subprocess.DEVNULL)

    try:
        for _ in range(300):
            try:
                r = requests.get(HEALTH_URL, timeout=1)
                if r.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        else:
            log_file.flush()
            log_file.seek(0)
            log_content = log_file.read()
            pytest.fail(
                f"vLLM serve did not become healthy after 300s: {HEALTH_URL}\n"
                f"==== vLLM Serve Log Start ===\n{log_content}\n==== vLLM Serve Log End ==="
            )

        prompt = "bejing is a"
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 123
            }
        }
        resp = requests.post(COMPLETIONS_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        generated = data["choices"][0]["text"].strip()
        expected = "city in north china, it has many famous attractions"
        assert generated == expected, f"Expected `{expected}`, got `{generated}`"

    finally:
        server_proc.send_signal(signal.SIGINT)
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()


@pytest.mark.parametrize("max_tokens", [10])
@pytest.mark.parametrize("model", ["Qwen/Qwen3-30B-A3B"])
def test_lm_eval_accuracy_etp(model, max_tokens):
    log_file = open("accuracy_etp.log", "a+")
    cmd = [
        "vllm", "serve", model, "--max_model_len", "4096",
        "--tensor_parallel_size", "4", "--enforce_eager",
        "--enable_expert_parallel", "--additional_config",
        '{"expert_tensor_parallel_size": "4"}'
    ]
    server_proc = subprocess.Popen(cmd,
                                   stdout=log_file,
                                   stderr=subprocess.DEVNULL)

    try:
        for _ in range(300):
            try:
                r = requests.get(HEALTH_URL, timeout=1)
                if r.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        else:
            log_file.flush()
            log_file.seek(0)
            log_content = log_file.read()
            pytest.fail(
                f"vLLM serve did not become healthy after 300s: {HEALTH_URL}\n"
                f"==== vLLM Serve Log Start ===\n{log_content}\n==== vLLM Serve Log End ==="
            )

        prompt = "bejing is a"
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 123
            }
        }
        resp = requests.post(COMPLETIONS_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        generated = data["choices"][0]["text"].strip()
        expected = "city in china. it is the capital city of"
        assert generated == expected, f"Expected `{expected}`, got `{generated}`"

    finally:
        server_proc.send_signal(signal.SIGINT)
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
