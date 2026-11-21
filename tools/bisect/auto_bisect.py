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
"""Stable command-line bootstrap for the bisect application."""


def main(argv: list[str] | None = None) -> int:
    # Load application dependencies after the CLI bootstrap prepares its environment.
    from tools.bisect.orchestration.cli import main as run

    return run(argv)


if __name__ == "__main__":
    from tools.bisect.execution.tool_env import ensure_tool_environment

    ensure_tool_environment()
    raise SystemExit(main())
