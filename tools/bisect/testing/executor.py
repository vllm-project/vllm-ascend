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
"""Run a fixed foreground test entry without nightly-specific behavior."""

import json
import os
import subprocess
from pathlib import Path

from tools.bisect.execution.process_output import run_streaming
from tools.bisect.testing.definition import TestEntry
from tools.bisect.testing.outcome import RunOutcome


def execute_test(entry: TestEntry, repository: Path, environment: dict[str, str] | None, log_path: Path) -> RunOutcome:
    directory = Path(entry.working_directory)
    if not directory.is_absolute():
        directory = repository / directory
    env = dict(environment if environment is not None else os.environ)
    env.update(entry.environment)
    with log_path.open("a", encoding="utf-8") as output:
        output.write("\n[test-entry] " + json.dumps(entry.command) + "\n")
        output.flush()
        try:
            process = run_streaming(
                list(entry.command),
                cwd=directory,
                env=env,
                output=output,
                timeout=entry.timeout_seconds,
                isolate_process_group=True,
            )
            return RunOutcome(exit_code=process.returncode)
        except subprocess.TimeoutExpired:
            return RunOutcome(exit_code=124, infra_error=True, skip_reason="test entry timed out")
        except OSError as exc:
            return RunOutcome(exit_code=127, infra_error=True, skip_reason=f"test entry could not start: {exc}")


def evaluate_test(outcome: RunOutcome) -> tuple[str, str]:
    if outcome.infra_error:
        return "SKIP", outcome.skip_reason
    if outcome.exit_code == 125:
        return "SKIP", "test entry could not judge this commit"
    return (
        ("PASS", "test entry exited 0")
        if outcome.exit_code == 0
        else ("FAIL", f"test entry exited {outcome.exit_code}")
    )
