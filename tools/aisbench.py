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
import os
import re
import subprocess

import pandas as pd
from modelscope import snapshot_download  # type: ignore

DATASET_CONF_DIR = "benchmark/ais_bench/benchmark/configs/datasets"
REQUEST_CONF_DIR = "benchmark/ais_bench/benchmark/configs/models/vllm_api"
DATASET_DIR = "benchmark/ais_bench/datasets"


class AisbenchRunner:
    RESULT_MSG = {
        "performance": "Performance Result files locate in ",
        "accuracy": "write csv to "
    }
    DATASET_RENAME = {"aime2024": "aime"}

    def _run_aisbench_task(self):
        dataset_conf = self.dataset_conf.split('/')[-1]
        if self.task_type == "accuracy":
            aisbench_cmd = [
                'ais_bench', '--models', f'{self.request_conf}_custom',
                '--datasets', f'{dataset_conf}', '--debug'
            ]
        if self.task_type == "performance":
            aisbench_cmd = [
                'ais_bench', '--models', f'{self.request_conf}_custom',
                '--datasets', f'{dataset_conf}_custom', '--debug', '--mode',
                'perf'
            ]
            if self.num_prompts:
                aisbench_cmd.extend(['--num-prompts', str(self.num_prompts)])
        self.proc: subprocess.Popen = subprocess.Popen(aisbench_cmd,
                                                       stdout=subprocess.PIPE,
                                                       stderr=subprocess.PIPE,
                                                       text=True)

    def __init__(self,
                 model: str,
                 port: int,
                 aisbench_config: dict,
                 verify=True):
        self.result_line = None
        self.dataset_path = snapshot_download(aisbench_config["dataset_path"],
                                              repo_type='dataset')
        self.task_type = aisbench_config["case_type"]
        self.request_conf = aisbench_config["request_conf"]
        self.dataset_conf = aisbench_config.get("dataset_conf")
        self.num_prompts = aisbench_config.get("num_prompts")
        self.max_out_len = aisbench_config["max_out_len"]
        self.batch_size = aisbench_config["batch_size"]
        self.request_rate = aisbench_config.get("request_rate", 0)
        self.model = model
        self.model_path = snapshot_download(model)
        self.port = port
        self._init_dataset_conf()
        self._init_request_conf()
        self._run_aisbench_task()
        self._wait_for_task()
        if verify:
            self.baseline = aisbench_config.get("baseline", 1)
            if self.task_type == "accuracy":
                self.threshold = aisbench_config.get("threshold", 1)
                self._accuracy_verify()
            if self.task_type == "performance":
                self.threshold = aisbench_config.get("threshold", 0.97)
                self._performance_verify()

    def _init_dataset_conf(self):
        if self.task_type == "accuracy":
            dataset_name = os.path.basename(self.dataset_path)
            dataset_rename = self.DATASET_RENAME.get(dataset_name, "")
            dst_dir = os.path.join(DATASET_DIR, dataset_rename)
            command = ["cp", "-r", self.dataset_path, dst_dir]
            subprocess.call(command)
        if self.task_type == "performance":
            conf_path = os.path.join(DATASET_CONF_DIR,
                                     f'{self.dataset_conf}.py')
            with open(conf_path, 'r', encoding='utf-8') as f:
                content = f.read()
            content = re.sub(r'path=.*', f'path="{self.dataset_path}",',
                             content)
            conf_path_new = os.path.join(DATASET_CONF_DIR,
                                         f'{self.dataset_conf}_custom.py')
            with open(conf_path_new, 'w', encoding='utf-8') as f:
                f.write(content)

    def _init_request_conf(self):
        conf_path = os.path.join(REQUEST_CONF_DIR, f'{self.request_conf}.py')
        with open(conf_path, 'r', encoding='utf-8') as f:
            content = f.read()
        content = re.sub(r'model=.*', f'model="{self.model}",', content)
        content = re.sub(r'host_port.*', f'host_port = {self.port},', content)
        content = re.sub(r'max_out_len.*',
                         f'max_out_len = {self.max_out_len},', content)
        content = re.sub(r'batch_size.*', f'batch_size = {self.batch_size},',
                         content)
        content = content.replace("top_k", "#top_k")
        content = content.replace("seed", "#seed")
        content = content.replace("repetition_penalty", "#repetition_penalty")
        if self.task_type == "performance":
            content = re.sub(r'path=.*', f'path="{self.model_path}",', content)
            content = re.sub(r'request_rate.*',
                             f'request_rate = {self.request_rate},', content)
            content = re.sub(
                r"temperature.*",
                "temperature = 0,\n            ignore_eos = True,", content)
            content = content.replace("top_p", "#top_p")
        if self.task_type == "accuracy":
            content = re.sub(r"temperature.*", "temperature = 0.6,", content)
        conf_path_new = os.path.join(REQUEST_CONF_DIR,
                                     f'{self.request_conf}_custom.py')
        with open(conf_path_new, 'w', encoding='utf-8') as f:
            f.write(content)

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
        result_msg = self.RESULT_MSG[self.task_type]
        while True:
            line = self.proc.stdout.readline().strip()
            print(line)
            if result_msg in line:
                self.result_line = line
                return
            if "ERROR" in line:
                raise RuntimeError(
                    "Some errors happen to Aisbench task.") from None

    def _get_result_performance(self):
        result_dir = re.search(r'Performance Result files locate in (.*)',
                               self.result_line).group(1)[:-1]
        result_csv_file = os.path.join(result_dir, "gsm8kdataset.csv")
        result_json_file = os.path.join(result_dir, "gsm8kdataset.json")
        self.result_csv = pd.read_csv(result_csv_file)
        with open(result_json_file, 'r', encoding='utf-8') as f:
            self.result_json = json.load(f)

    def _get_result_accuracy(self):
        acc_file = re.search(r'write csv to (.*)', self.result_line).group(1)
        df = pd.read_csv(acc_file)
        return float(df.loc[0][-1])

    def _performance_verify(self):
        self._get_result_performance()
        output_throughput = self.result_json["Output Token Throughput"][
            "total"].replace("token/s", "")
        assert float(
            output_throughput
        ) >= self.threshold * self.baseline, f"Performance verification failed. The current Output Token Throughput is {output_throughput} token/s, which is not greater than or equal to {self.threshold} * baseline {self.baseline}."

    def _accuracy_verify(self):
        acc_value = self._get_result_accuracy()
        assert self.baseline - self.threshold <= acc_value <= self.baseline + self.threshold, f"Accuracy verification failed. The accuracy of {self.dataset_path} if not within {self.threshold} relative to baseline {self.baseline}."


def run_aisbench_cases(model, port, aisbench_cases):
    aisbench_errors = []
    for aisbench_case in aisbench_cases:
        try:
            with AisbenchRunner(model, port, aisbench_case):
                pass
        except Exception as e:
            aisbench_errors.append([aisbench_case, e])
            print(e)
    for failed_case, error_info in aisbench_errors:
        print(
            f"The following aisbench case failed: {failed_case}, reason is {error_info}."
        )
    assert not aisbench_errors, "some aisbench cases failed, info were shown above."
