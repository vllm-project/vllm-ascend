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
"""Fixed command entry loaded before checkout, independent of nightly conventions."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TestEntry:
    command: tuple[str, ...]
    working_directory: str = "."
    environment: tuple[tuple[str, str], ...] = ()
    timeout_seconds: float = 7200.0

    @classmethod
    def load(cls, path: Path) -> "TestEntry":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) - {"command", "working_directory", "environment", "timeout_seconds"}:
            raise ValueError("Unknown test-entry fields")
        command = data.get("command")
        environment = data.get("environment", {})
        directory = data.get("working_directory", ".")
        timeout = data.get("timeout_seconds", 7200.0)
        if (
            not isinstance(command, list)
            or not command
            or not command[0]
            or not all(isinstance(x, str) and "\x00" not in x for x in command)
        ):
            raise ValueError("command must be a string array with a nonempty executable and no NUL characters")
        if not isinstance(environment, dict) or not all(
            isinstance(k, str)
            and bool(k)
            and "=" not in k
            and "\x00" not in k
            and isinstance(v, str)
            and "\x00" not in v
            for k, v in environment.items()
        ):
            raise ValueError("environment must contain valid nonempty names and string values without NUL characters")
        if not isinstance(directory, str) or not directory or "\x00" in directory:
            raise ValueError("working_directory must be a nonempty string")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        return cls(tuple(command), directory, tuple(sorted(environment.items())), float(timeout))

    @property
    def identity(self) -> str:
        payload = json.dumps((self.command, self.working_directory, self.environment, self.timeout_seconds))
        return hashlib.sha256(payload.encode()).hexdigest()
