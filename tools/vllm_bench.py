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
import json
import logging
import os
import subprocess
from datetime import datetime

from .aisbench import maybe_download_from_modelscope


class VllmbenchRunner:

    def _run_vllm_bench_task(self):
        vllm_bench_cmd = [
            'vllm', 'bench', 'serve', '--backend', 'openai-chat',
            '--trust-remote-code', '--served-model-name',
            str(self.model_name), '--model', self.model_path, '--tokenizer',
            self.model_path, '--metric-percentiles', '50,90,99', '--host',
            self.host_ip, '--port',
            str(self.port), '--save-result', '--result-filename',
            self.result_filename, '--endpoint', '/v1/chat/completions',
            '--ready-check-timeout-sec', '0'
        ]
        self._concat_config_args(vllm_bench_cmd)
        print(f"running vllm_bench cmd: {' '.join(vllm_bench_cmd)}")
        self.proc: subprocess.Popen = subprocess.Popen(vllm_bench_cmd,
                                                       stdout=subprocess.PIPE,
                                                       stderr=subprocess.PIPE,
                                                       text=True)

    def __init__(self,
                 model_name: str,
                 port: int,
                 config: dict,
                 model_path: str = "",
                 host_ip: str = "localhost"):
        self.model_name = model_name
        self.model_path = model_path
        if not self.model_path:
            self.model_path = maybe_download_from_modelscope(model_name)
        assert self.model_path is not None, \
            f"Failed to download model: model={self.model_path}"
        self.port = port
        self.host_ip = host_ip
        curr_time = datetime.now().strftime('%Y%m%d%H%M%S')
        self.result_filename = f"result_vllm_bench_{curr_time}.json"
        self.config = config

        self._run_vllm_bench_task()
        self._wait_for_task()
        self._get_result()

    def _concat_config_args(self, vllm_bench_cmd):
        if "ignore_eos" in self.config:
            if self.config["ignore_eos"]:
                self.config["ignore_eos"] = ""
            else:
                self.config.pop("ignore_eos")
        for key, value in self.config.items():
            key = "--" + key.replace("_", "-")
            vllm_bench_cmd += [key, str(value)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.proc.terminate()
        try:
            self.proc.wait(8)
        except subprocess.TimeoutExpired:
            # force kill if needed
            self.proc.kill()

    def _wait_for_task(self):
        result_msg = "========================="
        while True:
            line = self.proc.stdout.readline().strip()
            if line:
                print(line)
            if result_msg in line:
                return
            if "ERROR" in line:
                error_msg = f"Some errors happened to vllm_bench runtime, the first error is {line}"
                raise RuntimeError(error_msg) from None

    def _get_result(self):
        result_file = os.path.join(os.getcwd(), self.result_filename)
        print("Getting performance results from file: ", result_file)
        with open(result_file, 'r', encoding='utf-8') as f:
            self.result = json.load(f)


def run_vllm_bench_case(model_name,
                        port,
                        config,
                        model_path="",
                        host_ip="localhost"):
    try:
        with VllmbenchRunner(model_name,
                             port,
                             config,
                             model_path=model_path,
                             host_ip=host_ip) as vllm_bench:
            vllm_bench_result = vllm_bench.result
    except Exception as e:
        print(e)
        error_msg = f"vllm_bench run failed, reason is {e}"
        logging.error(error_msg)
        assert False, f"vllm_bench run failed, reason is {e}"
    return vllm_bench_result
