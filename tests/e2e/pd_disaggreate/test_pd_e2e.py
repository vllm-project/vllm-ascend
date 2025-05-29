#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import os
import signal
import subprocess
import time

import psutil
import requests


def kill_process_and_children(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            print(f"Killing child process {child.pid}")
            child.kill()
        print(f"Killing parent process {pid}")
        parent.kill()
    except psutil.NoSuchProcess:
        pass


def kill_all_vllm_related():
    current_pid = os.getpid()

    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            if proc.pid == current_pid:
                continue
            cmd = ' '.join(proc.info['cmdline'])
            if "vllm" in cmd or "proxy" in cmd or "engine_worker" in cmd:
                kill_process_and_children(proc.pid)
        except Exception:
            continue


PROXY_PORT = 10102
DECODE_PORT = 8002

SCRIPT_PATH = os.path.abspath("./tests/e2e/run_disagg_pd.sh")


def wait_for_port(port, timeout=30):
    import socket
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    raise TimeoutError(f"Port {port} not ready after {timeout}s")


def start_and_test_pipeline():
    print("Launching bash script to run vLLM PD setup...")
    proc = subprocess.Popen(["bash", SCRIPT_PATH])
    try:
        print("Waiting for proxy port to be available...")
        wait_for_port(PROXY_PORT, 180)
        wait_for_port(DECODE_PORT, 600)

        # request
        payload = {
            "model": "Deepseek",
            "prompt": "The future of AI is",
            "max_tokens": 64,
            "temperature": 0,
        }
        response = requests.post(
            f"http://localhost:{PROXY_PORT}/v1/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10)
        assert response.status_code == 200, f"HTTP failed: {response.status_code}"
        result = response.json()
        print("Response:", result)
        assert "text" in result["choices"][0]
        assert len(result["choices"][0]["text"].strip()) > 0

    finally:
        # clean up subprocesses
        print("Cleaning up subprocess...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        kill_all_vllm_related()


def test_disaggregated_pd_pipeline():
    start_and_test_pipeline()
