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

"""Reusable vLLM OpenAI server process helper for end-to-end tests."""

import contextlib
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from typing import Any

import openai
import psutil
import requests
from requests.exceptions import RequestException

from tests.e2e.common.multi_node.internal_dp.multi_node_config import DisaggregatedPrefillCfg, NodeInfo

logger = logging.getLogger(__name__)


class RemoteOpenAIServer:
    DUMMY_API_KEY = "token-abc123"  # vLLM's OpenAI server does not need API key

    def _start_server(self, model: str, server_cmd: list[str], env_dict: dict[str, str] | None) -> None:
        """Subclasses override this method to customize server process launch"""
        env = os.environ.copy()
        # the current process might initialize npu,
        # to be safe, we should use spawn method
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if env_dict is not None:
            env.update(env_dict)
        logger.info("Starting server with command: %s", " ".join(server_cmd))
        self.proc: subprocess.Popen = subprocess.Popen(
            server_cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    def __init__(
        self,
        model: str,
        vllm_serve_args: list[str] | str,
        *,
        server_host: str = "0.0.0.0",
        server_port: int = 8080,
        env_dict: dict[str, str] | None = None,
        seed: int | None = None,
        auto_port: bool = True,
        nodes_info: list[NodeInfo] | None = None,
        disaggregated_prefill: DisaggregatedPrefillCfg | None = None,
        proxy_port: int | None = None,
        max_wait_seconds: float | None = None,
        override_hf_configs: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(vllm_serve_args, str):
            vllm_serve_args = shlex.split(vllm_serve_args)
        else:
            vllm_serve_args = ["vllm", "serve", model, *vllm_serve_args]
        if auto_port:
            from vllm.utils.network_utils import get_open_port

            if "-p" in vllm_serve_args or "--port" in vllm_serve_args:
                raise ValueError("You have manually specified the port when `auto_port=True`.")

            # No need for a port if using unix sockets
            if "--uds" not in vllm_serve_args:
                # Don't mutate the input args
                vllm_serve_args = vllm_serve_args + ["--port", str(get_open_port())]
        if seed is not None:
            if "--seed" in vllm_serve_args:
                raise ValueError(f"You have manually specified the seed when `seed={seed}`.")

            vllm_serve_args = vllm_serve_args + ["--seed", str(seed)]

        if override_hf_configs is not None:
            vllm_serve_args = vllm_serve_args + ["--hf-overrides", json.dumps(override_hf_configs)]

        self.host = str(server_host)
        self.port = int(server_port)
        # for multi-nodes test
        self.nodes_info = nodes_info
        self.disaggregated_prefill = disaggregated_prefill
        self.cur_index = os.getenv("LWS_WORKER_INDEX", 0)
        self.proxy_port = proxy_port

        self._start_server(model, vllm_serve_args, env_dict)
        max_wait_seconds = max_wait_seconds or 2800
        if self.disaggregated_prefill:
            assert proxy_port is not None, "for disaggregated_prefill, proxy port must be provided"
            self._wait_for_server_pd(timeout=max_wait_seconds)
        else:
            self._wait_for_multiple_servers([(self.host, self.url_for("health"))], timeout=max_wait_seconds)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._terminate_server()

    def _poll(self) -> int | None:
        """Subclasses override this method to customize process polling"""
        return self.proc.poll()

    def hang_until_terminated(self, url) -> None:
        """
        Wait until the server process terminates.
        This is for headless mode, where the api server
        process only exists in the leader node.
        """
        logger.info("Hanging until server process terminates...")
        client = requests
        try:
            while True:
                try:
                    resp = client.get(url, timeout=5)
                    if resp.status_code != 200:
                        break
                    time.sleep(5)
                except Exception:
                    break
        finally:
            self._terminate_server()

    def _wait_for_server_pd(self, timeout: float):
        # Wait for all api_server nodes ready
        assert self.nodes_info is not None, "cluster info must be provided"
        proxy_port = self.proxy_port

        def url_health(ip: str, port: int) -> str:
            return f"http://{ip}:{port}/health"

        targets = [
            (node_info.ip, url_health(node_info.ip, self.port))
            for node_info in self.nodes_info
            if not node_info.headless
        ]

        # Wait for proxy ready
        master_node = self.nodes_info[0]
        url_proxy = f"http://{master_node.ip}:{proxy_port}/healthcheck"

        # Wait for master node proxy first
        self._wait_for_multiple_servers([(master_node.ip, url_proxy)], timeout=timeout)

        # Then wait for all api_server nodes
        self._wait_for_multiple_servers(targets=targets, timeout=timeout)

    def _wait_for_multiple_servers(
        self, targets, timeout: float, log_interval: float = 30.0, always_check_nodes: bool = False
    ):
        """
        targets: List[(node_ip, url)]
        log_interval
        """
        start = time.time()
        client = requests

        # Key by (node_ip, url): several servers may share one IP (e.g. the
        # single-node PD/EPD modes start multiple engines on 127.0.0.1), so a
        # plain-IP key would collapse their readiness state together and make
        # the timeout message report an empty not-ready list.
        ready = {(node_ip, url): False for node_ip, url in targets}

        last_log_time = 0.0

        while True:
            now = time.time()
            all_ready = True
            should_log = (now - last_log_time) >= log_interval

            for node_ip, url in targets:
                if ready[(node_ip, url)] and not always_check_nodes:
                    continue

                try:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        ready[(node_ip, url)] = True
                        logger.info("[READY] Node %s: %s is ready.", node_ip, url)
                except RequestException:
                    all_ready = False
                    if should_log:
                        logger.debug("[WAIT] %s: connection failed", url)

                    # check unexpected exit
                    result = self._poll()
                    if result is not None and result != 0:
                        self._terminate_server()
                        raise RuntimeError(f"Server at {node_ip} exited unexpectedly.") from None

            if should_log:
                last_log_time = now

            if all_ready:
                break

            if now - start > timeout:
                not_ready_nodes = [url for (_, url), ok in ready.items() if not ok]
                self._terminate_server()
                raise RuntimeError(
                    f"Timeout: these nodes did not become ready: {not_ready_nodes} in time: {timeout}s"
                ) from None

            time.sleep(5)

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _terminate_server(self) -> None:
        """Subclasses override this method to customize server process termination"""
        self._terminate_process_tree(self.proc)

    def _terminate_process_tree(self, proc: subprocess.Popen) -> None:
        try:
            parent = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            return

        children = parent.children(recursive=True)

        try:
            parent.terminate()
            parent.wait(timeout=60)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            with contextlib.suppress(psutil.NoSuchProcess):
                parent.kill()

        for child in children:
            with contextlib.suppress(psutil.NoSuchProcess):
                child.terminate()

        _, still_alive = psutil.wait_procs(children, timeout=10)

        for child in still_alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def get_client(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.OpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def get_async_client(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.AsyncOpenAI(base_url=self.url_for("v1"), api_key=self.DUMMY_API_KEY, max_retries=0, **kwargs)
