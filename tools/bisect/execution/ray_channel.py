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
"""Pinned Ray execution against an existing cluster, without retry or soft placement."""

import codecs
import time

from tools.bisect.execution.bundle import BOOTSTRAP
from tools.bisect.execution.channels import ChannelError, parse_response
from tools.bisect.execution.targets import ExecutionTarget

# Functions are built in an isolated namespace so Ray serializes them by value.
# Remote workers need only Python's standard library to start our frozen tools.
_REMOTE_SOURCE = r"""
def run(python, bootstrap, request, timeout):
    import contextlib, json, os, pathlib, signal, subprocess, sys, tempfile
    request["ray_runtime_prefix"] = sys.prefix
    root = pathlib.Path(tempfile.gettempdir()) / ("bisect-ray-" + request["nonce"])
    root.mkdir(mode=0o700)
    with tempfile.TemporaryFile() as payload, (root / "output.log").open("wb") as output:
        payload.write(json.dumps(request).encode()); payload.seek(0)
        process = subprocess.Popen([python, "-u", "-c", bootstrap], stdin=payload, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ray target command timed out")
        finally:
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
            process.wait()

def read(nonce, offset):
    import pathlib, tempfile
    path = pathlib.Path(tempfile.gettempdir()) / ("bisect-ray-" + nonce) / "output.log"
    if not path.exists(): return b""
    with path.open("rb") as stream:
        stream.seek(offset)
        return stream.read(65536)
"""


class RayChannel:
    def __init__(self, address: str):
        try:
            import ray  # Optional transport dependency, loaded only when selected.
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        except ImportError as exc:
            raise ChannelError("Ray channel selected but Ray is not installed in the controller environment") from exc
        if ray.is_initialized():
            raise ChannelError(
                "Use a separate controller process for the explicit Ray address; implicit reuse is not allowed"
            )
        self.ray = ray
        self.strategy = NodeAffinitySchedulingStrategy
        ray.init(address=address, log_to_driver=False)
        namespace = {"__name__": "__main__"}
        exec(_REMOTE_SOURCE, namespace)
        self.run_task = ray.remote(num_cpus=0, max_retries=0)(namespace["run"])
        self.read_task = ray.remote(num_cpus=0, max_retries=0)(namespace["read"])

    def verify(self, target: ExecutionTarget) -> None:
        nodes = [node for node in self.ray.nodes() if node.get("NodeID") == target.option("node_id")]
        if len(nodes) != 1 or not nodes[0].get("Alive"):
            raise ChannelError(f"Explicit Ray node is missing or dead: {target.name}")

    def invoke(self, target: ExecutionTarget, request: dict, output, timeout: float) -> dict:
        self.verify(target)
        placement = self.strategy(node_id=target.option("node_id"), soft=False)
        reference = self.run_task.options(scheduling_strategy=placement).remote(
            target.python, BOOTSTRAP, request, timeout
        )
        offset = 0
        tail = ""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        deadline = time.monotonic() + timeout
        ready = False
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ChannelError(f"Ray transport deadline exceeded: {target.name}; target completion is unknown")
                chunk_ref = self.read_task.options(scheduling_strategy=placement).remote(request["nonce"], offset)
                chunk = self.ray.get(chunk_ref, timeout=min(30, remaining))
                if chunk:
                    offset += len(chunk)
                    text = decoder.decode(chunk)
                    print(text, end="", flush=True)
                    output.write(text)
                    output.flush()
                    tail = (tail + text)[-65536:]
                    continue
                if ready:
                    break
                completed, _ = self.ray.wait([reference], timeout=min(0.2, remaining))
                ready = bool(completed)
            return_code = self.ray.get(reference, timeout=max(0.1, deadline - time.monotonic()))
            if return_code:
                raise ChannelError(f"Ray target process exited {return_code}: {target.name}")
            self.verify(target)
            return parse_response(tail, request["nonce"])
        except BaseException:
            self.ray.cancel(reference, force=True)
            raise

    def close(self) -> None:
        self.ray.shutdown()
