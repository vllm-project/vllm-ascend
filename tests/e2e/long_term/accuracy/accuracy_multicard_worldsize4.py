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
import multiprocessing
import signal
import subprocess
import time

import pytest
import requests

from tests.e2e.long_term.accuracy.accuracy_multicard_worldsize2 import (
    COMPLETIONS_URL, HEALTH_URL)

multiprocessing.set_start_method("spawn", force=True)


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
