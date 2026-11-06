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

"""Build analyzer front end: lex, parse, analyze, lower, compare."""

from tools.bisect.build_analyzer.diff import compare_builds
from tools.bisect.build_analyzer.ir import BuildAnalysis, BuildSyntaxError, Instruction, Operation
from tools.bisect.build_analyzer.semantics import analyze_build
from tools.bisect.build_analyzer.sources import analyze_commit, analyze_directory
from tools.bisect.build_analyzer.syntax import parse_dockerfile

__all__ = (
    "BuildAnalysis",
    "BuildSyntaxError",
    "Instruction",
    "Operation",
    "parse_dockerfile",
    "analyze_build",
    "analyze_commit",
    "analyze_directory",
    "compare_builds",
)
