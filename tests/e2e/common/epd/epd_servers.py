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

"""Reusable EPD (encoder-prefill-decode) test server helpers."""

import copy
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import traceback
from pathlib import Path

from tests.e2e.common.serving.remote_openai_server import RemoteOpenAIServer

DISAGG_EPD_PROXY_SCRIPT = (
    Path(__file__).parent.parent.parent.parent.parent / "examples" / "disaggregated_encoder" / "disagg_epd_proxy.py"
)


class RemoteEPDServer(RemoteOpenAIServer):
    def _start_server(self, model: str, server_cmd: list[str], env_dict: dict[str, str] | None) -> None:
        """Subclasses override this method to customize server process launch"""
        raise NotImplementedError("RemoteEPDServer should use _start_server_with_prefix instead")

    def __init__(
        self,
        vllm_serve_args: list[str] | list[list[str]],
        server_host: str = "0.0.0.0",
        env_dict: dict[str, str] | None = None,
        max_wait_seconds: float | None = 2800,
        per_instance_env: list[dict[str, str]] | None = None,
        assert_clean_exit: bool = True,
    ) -> None:
        self._proc_list = []
        self.exit_codes: list[int | None] = []
        self.assert_clean_exit = assert_clean_exit

        self.env_dict: dict[str, str] = {}
        if env_dict is not None:
            self.env_dict.update(env_dict)

        self.env_dict["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        self.env_dict["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        self.env_dict["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        self.vllm_serve_args_list = []
        self.health_url_list = []
        self.host = server_host

        if isinstance(vllm_serve_args, list):
            if not all(isinstance(item, list) for item in vllm_serve_args):
                args_copy = copy.deepcopy(vllm_serve_args)
                self.vllm_serve_args_list.append([str(arg) for arg in args_copy])
            else:
                self.vllm_serve_args_list = [
                    [str(arg) for arg in sublist] for sublist in copy.deepcopy(vllm_serve_args)
                ]
        else:
            raise RuntimeError("vllm_serves_args must be a list")

        if per_instance_env is not None and len(per_instance_env) != len(self.vllm_serve_args_list):
            raise ValueError(
                "per_instance_env must provide one dict per instance: "
                f"got {len(per_instance_env)}, expected {len(self.vllm_serve_args_list)}"
            )

        serve_arg_cmd = ["vllm", "serve"]

        for i, vllm_serve_arg in enumerate(self.vllm_serve_args_list):
            instance_env = dict(self.env_dict)
            instance_env["ASCEND_RT_VISIBLE_DEVICES"] = str(i)
            if per_instance_env is not None:
                instance_env.update({key: str(value) for key, value in per_instance_env[i].items()})
            if isinstance(vllm_serve_arg, list):
                if "--port" not in vllm_serve_arg:
                    raise ValueError("You have to manually specify the port")
                else:
                    port_arg = "--port"
                    try:
                        index = vllm_serve_arg.index(port_arg)
                    except ValueError:
                        raise ValueError(f"--port not found in args: {vllm_serve_arg}")
                    port_str = vllm_serve_arg[index + 1]
                    self.port = int(port_str)
            else:
                vllm_serve_arg_str = str(vllm_serve_arg)
                if "--port" not in vllm_serve_arg_str:
                    raise ValueError("You have to manually specify the port")
                else:
                    raise ValueError(f"Unexpected type for vllm_serve_arg: {type(vllm_serve_arg)}")

            self.health_url_list.append(super().url_for("health"))
            vllm_serve_arg = [*serve_arg_cmd, *vllm_serve_arg]
            proc = self._start_server_with_prefix(vllm_serve_arg, instance_env, f"[VLLM_{i}] ")
            self._proc_list.append(proc)

        timeout_value = float(max_wait_seconds) if max_wait_seconds is not None else 2800.0
        super()._wait_for_multiple_servers(
            [(self.host, url) for url in self.health_url_list], timeout=timeout_value, always_check_nodes=True
        )

    def _poll(self) -> int | None:
        return None

    def _delete_shm(self) -> list[str]:
        """Delete the EC shared-storage paths of all instances and return them.

        Idempotent: ``shutil.rmtree(..., ignore_errors=True)`` makes repeated
        calls safe, and an unparsable ``--ec-transfer-config`` only warns instead
        of crashing the context-manager exit path.
        """
        deleted: list[str] = []
        for i, arg in enumerate(getattr(self, "vllm_serve_args_list", [])):
            if "--ec-transfer-config" not in arg:
                continue
            index = arg.index("--ec-transfer-config")
            config_str = arg[index + 1]
            try:
                config_dict = json.loads(config_str)
            except (TypeError, ValueError) as exc:
                print(f"warning: failed to parse --ec-transfer-config for instance {i}: {exc}")
                continue
            extra_config = config_dict.get("ec_connector_extra_config") if isinstance(config_dict, dict) else None
            shm_path = extra_config.get("shared_storage_path") if isinstance(extra_config, dict) else None
            if not shm_path:
                continue
            print(f"delete shm_path is: {shm_path}")
            shutil.rmtree(str(shm_path), ignore_errors=True)
            deleted.append(str(shm_path))
        return deleted

    def _read_output(self, pipe, prefix):
        try:
            with pipe:
                for line in iter(pipe.readline, ""):
                    if line:
                        print(f"{prefix}: {line}", end="")

        except Exception as e:
            print(f"error: {e}")
            traceback.print_exc()

    def _start_server_with_prefix(self, server_cmd: list[str], env_dict: dict[str, str] | None, log_prefix: str):
        env = os.environ.copy()
        if env_dict is not None:
            env.update(env_dict)
        proc = subprocess.Popen(
            server_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )
        stdout_thread = threading.Thread(target=self._read_output, args=(proc.stdout, log_prefix), daemon=True)
        stderr_thread = threading.Thread(target=self._read_output, args=(proc.stderr, log_prefix), daemon=True)

        stdout_thread.start()
        stderr_thread.start()
        return proc

    def _terminate_server(self) -> list[int | None]:
        """Kill server processes and their children, returning their exit codes."""
        print("vllm instance is stopping")
        self.exit_codes = []
        for proc in self._proc_list:
            self._terminate_process_tree(proc)
            try:
                code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                code = None
            self.exit_codes.append(code)
        return self.exit_codes

    def _assert_clean_shutdown(self) -> None:
        """Assert that every server process exited with an expected code."""
        still_alive = [(index, proc.pid) for index, proc in enumerate(self._proc_list) if proc.poll() is None]
        if still_alive:
            raise AssertionError(f"server processes still alive after shutdown (index, pid): {still_alive}")
        allowed_exit_codes = {0, -signal.SIGTERM}
        unexpected = [(index, code) for index, code in enumerate(self.exit_codes) if code not in allowed_exit_codes]
        if unexpected:
            raise AssertionError(f"unexpected server exit codes (index, code): {unexpected}")

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: stop processes, clean EC storage, verify the exit."""
        self._terminate_server()
        self._delete_shm()
        # Never mask an exception raised inside the ``with`` block with a teardown assertion.
        if exc_type is None and getattr(self, "assert_clean_exit", True):
            self._assert_clean_shutdown()


class DisaggEpdProxy(RemoteEPDServer):
    def __init__(
        self,
        proxy_args: list[str] | str | None = None,
        env_dict: dict[str, str] | None = None,
        server_host: str = "0.0.0.0",
        max_wait_seconds: float | None = 2800,
        assert_clean_exit: bool = True,
    ) -> None:
        if proxy_args is None:
            proxy_args_list: list[str] = []
        elif isinstance(proxy_args, str):
            proxy_args_list = shlex.split(proxy_args)
        else:
            proxy_args_list = proxy_args

        self.proxy_args = proxy_args_list
        self.env_dict: dict[str, str] = {}
        if env_dict is not None:
            self.env_dict.update(env_dict)
        self._proc_list = list()
        self.exit_codes: list[int | None] = []
        self.assert_clean_exit = assert_clean_exit
        self.host = server_host

        print(f"proxy param is: {self.proxy_args}")
        proxy_cmd = [sys.executable, str(DISAGG_EPD_PROXY_SCRIPT), *self.proxy_args]
        proc = self._start_server_with_prefix(proxy_cmd, self.env_dict, "[PROXY] ")
        self._proc_list.append(proc)

        if "--port" not in self.proxy_args:
            raise ValueError("You have manually specified the port ")
        else:
            try:
                index = self.proxy_args.index("--port")
            except ValueError:
                raise ValueError("--port not found in proxy args")
            port_str = self.proxy_args[index + 1]
            self.port = int(port_str)

        timeout_value = float(max_wait_seconds) if max_wait_seconds is not None else 2800.0
        super()._wait_for_multiple_servers([(self.host, super().url_for("health"))], timeout=timeout_value)

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point - reuse the base teardown and verification."""
        super().__exit__(exc_type, exc_val, exc_tb)
