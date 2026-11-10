# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Explicitly known forward and inverse operations for build transitions."""

import shlex
from dataclasses import dataclass

from tools.bisect.build_analyzer.ir import Operation


@dataclass(frozen=True)
class ExplicitInverseRule:
    name: str
    operation_kind: str
    forward_tokens: tuple[str, ...]
    inverse_tokens: tuple[str, ...]
    forward_action_kind: str
    inverse_action_kind: str

    def _match_index(self, operation: Operation) -> tuple[list[str], int] | None:
        if operation.kind != self.operation_kind:
            return None
        try:
            words = shlex.split(operation.value)
        except ValueError:
            return None
        width = len(self.forward_tokens)
        for index in range(len(words) - width + 1):
            if tuple(words[index : index + width]) == self.forward_tokens:
                return words, index
        return None

    def matches(self, operation: Operation) -> bool:
        return self._match_index(operation) is not None

    def inverse_command(self, operation: Operation) -> str | None:
        match = self._match_index(operation)
        if match is None:
            return None
        words, index = match
        words[index : index + len(self.forward_tokens)] = self.inverse_tokens
        return shlex.join(words)


EXPLICIT_INVERSE_RULES = (
    ExplicitInverseRule(
        name="mfcli-kernel",
        operation_kind="mfcli-kernel",
        forward_tokens=("mfcli", "kernel", "install"),
        inverse_tokens=("mfcli", "kernel", "uninstall"),
        forward_action_kind="install-kernel",
        inverse_action_kind="uninstall-kernel",
    ),
)


def explicit_rule_for(operation: Operation) -> ExplicitInverseRule | None:
    return next((rule for rule in EXPLICIT_INVERSE_RULES if rule.matches(operation)), None)


def warning_only_download(artifact: str | None) -> bool:
    """FLA release availability must not stop environment preparation."""
    name = (artifact or "").rsplit("/", 1)[-1].lower().replace("-", "_")
    return name.startswith("flash_linear_attention_npu_") and name.endswith(".whl")


def warning_only_repository(repository: str) -> bool:
    """Keep the existing benchmark available when AISBench cannot be fetched."""
    return repository.rstrip("/").removesuffix(".git").lower() == "https://github.com/aisbench/benchmark"
