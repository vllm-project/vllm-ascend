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
            "vllm",
            "bench",
            "serve",
            "--backend",
            "openai-chat",
            "--trust-remote-code",
            "--served-model-name",
            str(self.model_name),
            "--model",
            self.model_path,
            "--tokenizer",
            self.model_path,
            "--metric-percentiles",
            "50,90,99",
            "--host",
            self.host_ip,
            "--port",
            str(self.port),
            "--save-result",
            "--result-filename",
            self.result_filename,
            "--endpoint",
            "/v1/chat/completions",
            "--ready-check-timeout-sec",
            "0",
        ]
        self._concat_config_args(vllm_bench_cmd)
        print(f"running vllm_bench cmd: {' '.join(vllm_bench_cmd)}")
        self.proc: subprocess.Popen = subprocess.Popen(
            vllm_bench_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

    def __init__(
        self,
        model_name: str,
        port: int,
        config: dict,
        baseline: float,
        threshold: float = 0.97,
        model_path: str = "",
        host_ip: str = "localhost",
    ):
        self.model_name = model_name
        self.model_path = model_path
        if not self.model_path:
            self.model_path = maybe_download_from_modelscope(model_name)
        assert self.model_path is not None, f"Failed to download model: model={self.model_path}"
        self.port = port
        self.host_ip = host_ip
        curr_time = datetime.now().strftime("%Y%m%d%H%M%S")
        self.result_filename = f"result_vllm_bench_{curr_time}.json"
        self.config = config
        self.baseline = baseline
        self.threshold = threshold

        self._run_vllm_bench_task()
        self._wait_for_task()
        self._performance_verify()

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
        """Wait for the vllm bench command to complete and check the execution result"""

        stdout, stderr = self.proc.communicate()

        if self.proc.returncode != 0:
            logging.error(f"vllm bench command failed, return code: {self.proc.returncode}")
            logging.error(f"Standard output: {stdout}")
            logging.error(f"Standard error: {stderr}")
            raise RuntimeError(f"vllm bench command execution failed: {stderr}")

        logging.info(f"vllm bench command completed, return code: {self.proc.returncode}")
        if stdout:
            lines = stdout.split("\n")
            last_lines = lines[-100:] if len(lines) > 100 else lines
            logging.info(f"Last {len(last_lines)} lines of standard output:")
            for line in last_lines:
                logging.info(line)
        else:
            logging.info("Standard output is empty")

    def _get_result(self):
        result_file = os.path.join(os.getcwd(), self.result_filename)
        print("Getting performance results from file: ", result_file)
        with open(result_file, encoding="utf-8") as f:
            self.result = json.load(f)

    def _performance_verify(self):
        self._get_result()
        output_throughput = self.result["output_throughput"]
        assert float(output_throughput) >= self.baseline * self.threshold, (
            "Performance verification failed. "
            f"The current Output Token Throughput is {output_throughput} token/s, "
            f"which is not greater than or equal to {self.threshold} * baseline {self.baseline}."
        )


def run_vllm_bench_case(model_name, port, config, baseline, threshold=0.97, model_path="", host_ip="localhost"):
    try:
        with VllmbenchRunner(
            model_name, port, config, baseline, threshold, model_path=model_path, host_ip=host_ip
        ) as vllm_bench:
            vllm_bench_result = vllm_bench.result
    except Exception as e:
        print(e)
        error_msg = f"vllm_bench run failed, reason is {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e
    return vllm_bench_result
