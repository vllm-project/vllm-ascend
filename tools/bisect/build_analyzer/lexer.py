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

"""Dockerfile lexical analysis: comments, escape directive, continuations."""

from tools.bisect.build_analyzer.ir import BuildSyntaxError, DockerToken


def lex_dockerfile(source: str, content: str) -> tuple[DockerToken, ...]:
    """Emit logical lines after applying Docker's comment/continuation rules."""
    escape = "\\"
    pending: list[str] = []
    line_numbers: list[int] = []
    start = 0
    result: list[DockerToken] = []
    for number, raw in enumerate(content.splitlines(), 1):
        stripped = raw.lstrip()
        if not pending and stripped.startswith("# escape="):
            candidate = stripped[len("# escape=") :].strip()
            if candidate in ("\\", "`"):
                escape = candidate
        if not stripped or stripped.startswith("#"):
            continue
        if not pending:
            start = number
        continued = raw.rstrip().endswith(escape)
        fragment = raw.rstrip()
        if continued:
            fragment = fragment[:-1]
        pending.append(fragment.strip() + (f" {escape}" if continued else ""))
        line_numbers.append(number)
        if continued:
            continue
        logical = "\n".join(pending)
        result.append(DockerToken(source, start, logical, tuple(line_numbers)))
        pending = []
        line_numbers = []
    if pending:
        raise BuildSyntaxError(f"{source}:{start}: dangling Dockerfile continuation")
    return tuple(result)
