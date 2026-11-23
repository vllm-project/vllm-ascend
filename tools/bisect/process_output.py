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
"""Stream child output to the console and the trial log."""

import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import TextIO


def run_streaming(
    command: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str] | None,
    output: TextIO | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Tee merged stdout/stderr without making output reads defeat the timeout."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    messages: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                messages.put(line)
        finally:
            process.stdout.close()
            messages.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    deadline = None if timeout is None else time.monotonic() + timeout
    tail = ""
    try:
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                line = messages.get(timeout=remaining)
            except queue.Empty:
                raise subprocess.TimeoutExpired(command, timeout) from None
            if line is None:
                break
            print(line, end="", flush=True)
            if output is not None:
                output.write(line)
                output.flush()
            tail = (tail + line)[-4000:]
        remaining = None if deadline is None else max(0, deadline - time.monotonic())
        return subprocess.CompletedProcess(command, process.wait(timeout=remaining), stdout=tail)
    except BaseException:
        process.kill()
        process.wait()
        raise
