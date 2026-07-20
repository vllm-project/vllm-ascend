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

import os
import subprocess
import sys

import pytest

from tests.e2e.timeout_utils import (
    COVERAGE_TIMEOUT_MULTIPLIER,
    coverage_scaled_timeout,
    run_subprocess_with_timeout,
)


def test_coverage_scaled_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ENABLE_COVERAGE", raising=False)
    assert coverage_scaled_timeout(10) == 10

    monkeypatch.setenv("ENABLE_COVERAGE", "false")
    assert coverage_scaled_timeout(10) == 10

    monkeypatch.setenv("ENABLE_COVERAGE", "true")
    assert coverage_scaled_timeout(10) == 10 * COVERAGE_TIMEOUT_MULTIPLIER


def test_run_subprocess_with_timeout_success():
    proc = run_subprocess_with_timeout(
        [sys.executable, "-c", "print('ok')"],
        env=os.environ.copy(),
        timeout=5,
    )

    assert proc.returncode == 0
    assert proc.stdout.strip() == b"ok"


def test_run_subprocess_with_timeout_raises_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_subprocess_with_timeout(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            env=os.environ.copy(),
            timeout=0.1,
        )
