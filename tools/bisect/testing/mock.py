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
"""Deterministic test verdicts for exercising real environment preparations."""

from tools.bisect.orchestration.config import Verdict


class MockTestResults:
    def __init__(self, commits: tuple[str, ...]):
        self.bad_index = len(commits) // 2
        self.indices = {commit: index for index, commit in enumerate(commits)}

    def evaluate(self, commit: str) -> tuple[Verdict, str]:
        index = self.indices.get(commit, -1)
        verdict: Verdict = "FAIL" if index >= self.bad_index else "PASS"
        return verdict, f"MOCK {verdict}: environment prepared; test command was not run"
