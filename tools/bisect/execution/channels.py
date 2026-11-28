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
"""Transport requests to explicit existing environments without selecting a cluster."""

import json
import subprocess
import tempfile
from pathlib import Path

from tools.bisect.execution.bundle import BOOTSTRAP
from tools.bisect.execution.process_output import run_streaming
from tools.bisect.execution.targets import ExecutionTarget


class ChannelError(RuntimeError):
    """A transport failed; target completion cannot be assumed."""


def parse_response(output: str, nonce: str) -> dict:
    prefix = "BISECT-RESULT-" + nonce + " "
    matches = [line[len(prefix) :] for line in output.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise ChannelError("Target did not return exactly one matching completion response")
    response = json.loads(matches[0])
    if not isinstance(response, dict) or response.get("status") not in {"ok", "error"}:
        raise ChannelError("Malformed target response")
    return response


class LocalChannel:
    def verify(self, target: ExecutionTarget) -> None:
        if not Path(target.repository).is_dir():
            raise ChannelError(f"Target repository does not exist: {target.name}")

    def invoke(self, target: ExecutionTarget, request: dict, output, timeout: float) -> dict:
        self.verify(target)
        with tempfile.TemporaryFile() as payload:
            payload.write(json.dumps(request).encode())
            payload.seek(0)
            response = run_streaming(
                [target.python, "-u", "-c", BOOTSTRAP],
                cwd=target.repository,
                env=None,
                output=output,
                timeout=timeout,
                stdin=payload,
                isolate_process_group=True,
            )
        if response.returncode:
            raise ChannelError(f"Target process exited {response.returncode}: {target.name}")
        return parse_response(response.stdout, request["nonce"])

    def close(self) -> None:
        pass


class KubernetesChannel:
    def _prefix(self, target: ExecutionTarget) -> list[str]:
        return ["kubectl", "--context", target.option("context"), "--namespace", target.option("namespace")]

    def verify(self, target: ExecutionTarget) -> None:
        command = self._prefix(target) + ["get", "pod", target.option("pod"), "-o", "json"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        pod = json.loads(result.stdout)
        if pod["metadata"].get("uid") != target.option("pod_uid") or pod["metadata"].get("deletionTimestamp"):
            raise ChannelError(f"Pod identity changed or is terminating: {target.name}")
        statuses = pod.get("status", {}).get("containerStatuses", [])
        status = next((item for item in statuses if item.get("name") == target.option("container")), None)
        if (
            pod.get("status", {}).get("phase") != "Running"
            or status is None
            or not status.get("state", {}).get("running")
            or status.get("containerID") != target.option("container_id")
        ):
            raise ChannelError(f"Container is not running or its identity changed: {target.name}")

    def invoke(self, target: ExecutionTarget, request: dict, output, timeout: float) -> dict:
        self.verify(target)
        command = self._prefix(target) + [
            "exec",
            "-i",
            target.option("pod"),
            "-c",
            target.option("container"),
            "--",
            target.python,
            "-u",
            "-c",
            BOOTSTRAP,
        ]
        with tempfile.TemporaryFile() as payload:
            payload.write(json.dumps(request).encode())
            payload.seek(0)
            response = run_streaming(
                command,
                cwd=Path.cwd(),
                env=None,
                output=output,
                timeout=timeout,
                stdin=payload,
                isolate_process_group=True,
            )
        if response.returncode:
            raise ChannelError(f"Container exec exited {response.returncode}: {target.name}")
        self.verify(target)
        return parse_response(response.stdout, request["nonce"])

    def close(self) -> None:
        pass
