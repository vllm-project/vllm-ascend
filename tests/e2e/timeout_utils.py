#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import contextlib
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence

COVERAGE_TIMEOUT_MULTIPLIER = 3


def coverage_scaled_timeout(timeout: float) -> float:
    """Allow instrumented E2E processes more time without changing normal CI."""
    if os.getenv("ENABLE_COVERAGE") == "true":
        return timeout * COVERAGE_TIMEOUT_MULTIPLIER
    return timeout


def run_subprocess_with_timeout(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[bytes]:
    """Run a subprocess and kill its process group if it times out."""
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if hasattr(os, "killpg"):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            cmd,
            timeout,
            output=stdout if stdout is not None else exc.output,
            stderr=stderr if stderr is not None else exc.stderr,
        ) from exc

    return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
