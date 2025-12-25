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
#

from zhejing.send_reqs_with_pressure import run_zhejing  # type: ignore
from zhejing.test_h_postproc import run_postproc  # type: ignore

from tools.aisbench import run_aisbench_cases
from tools.send_request import (get_prompt_from_dataset,
                                send_v1_chat_completions, send_v1_completions)
from tools.vllm_bench import run_vllm_bench_case

server_ip = "0.0.0.0"
port = "1999"
model_path = "/mnt/share/Qwen2.5-VL-7B-Instruct"
model_name = "qwen25vl"


class MockServer:
    url_root = f"http://{server_ip}:{port}"

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)


server = MockServer()


def test_single_request():
    prompt = "San Francisco is a"
    api_keyword_args = {"max_tokens": 10}
    send_v1_chat_completions(prompt,
                             model_name,
                             server,
                             request_args=api_keyword_args)
    send_v1_completions(prompt,
                        model_name,
                        server,
                        request_args=api_keyword_args)


def test_aisbench_cases():
    aisbench_cases = [{
        "case_type": "accuracy",
        "dataset_path": "vllm-ascend/gsm8k-lite",
        "request_conf": "vllm_api_general_chat",
        "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_chat_prompt",
        "max_out_len": 10240,
        "batch_size": 32,
        "baseline": 50,
        "threshold": 50,
        "model_path": model_path,
        "dataset_path_local": ""
    }, {
        "case_type": "performance",
        "dataset_path": "vllm-ascend/GSM8K-in3500-bs400",
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_str_perf",
        "num_prompts": 8,
        "max_out_len": 2,
        "batch_size": 2,
        "baseline": 1,
        "threshold": 0.97,
        "model_path": model_path,
        "dataset_path_local": ""
    }]
    run_aisbench_cases(model_name, port, aisbench_cases, host_ip=server_ip)


def test_zhejing():
    config = {
        "IP": server_ip,
        "PORT": port,
        "model_name": model_name,
        "test_concurrent_workers": 16,
        "max_tokens": 10,
        "background_concurrent_workers": 0,
        "is_stream": False
    }
    run_zhejing(config)
    config["is_stream"] = True
    run_zhejing(config)


def test_postproc():
    run_postproc(server_ip, port, model_name)
    run_postproc(server_ip, port, model_name, is_long=True)


def test_single_long():
    prompt = get_prompt_from_dataset("vllm-ascend/GSM8K-in131072-bs1",
                                     "GSM8K-in131072-bs1.txt")
    send_v1_chat_completions(prompt, model_name, server)


def test_vllm_bench():
    config = {
        "input_len": 1024,
        "output_len": 2,
        "max_concurrency": 2,
        "request_rate": "5000",
        "model_path": model_path
    }
    res = run_vllm_bench_case(model_name, port, config)
    print("vllm bench result is: ", res)


def test_linear():
    config = {
        "input_len": 1024,
        "output_len": 2,
        "request_rate": "5000",
        "model_path": model_path
    }
    bs_list = [1, 2, 4, 8, 12, 16]
    TPOT_list = []
    for bs in bs_list:
        config["max_concurrency"] = bs
        res = run_vllm_bench_case(model_name, port, config)
        TPOT_list.append(res["mean_tpot_ms"])
    assert TPOT_list == sorted(TPOT_list), "linear test failed"
    print("TPOT is linear.")
