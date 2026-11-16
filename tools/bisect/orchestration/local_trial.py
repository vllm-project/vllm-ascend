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
"""Compose local preparation and an opaque foreground test entry."""

from pathlib import Path

from tools.bisect.deployment.manager import DEPLOY_ERRORS
from tools.bisect.testing.executor import execute_test
from tools.bisect.testing.outcome import RunOutcome


class LocalCommandTrial:
    def __init__(self, definition, options, deployment):
        self.definition = definition
        self.options = options
        self.deployment = deployment

    def validate(self, candidate, round_idx: int, log_dir: Path) -> RunOutcome:
        log_path = log_dir / f"round{round_idx}_{candidate.short}.log"
        self.deployment.decide(candidate.commit)
        try:
            decision = self.deployment.prepare(candidate.commit, log_path)
        except DEPLOY_ERRORS:
            self.deployment.invalidate_baseline()
            raise
        if self.options.mock_tests:
            with log_path.open("a", encoding="utf-8") as output:
                output.write("[mock-tests] Environment prepared; command entry was not run.\n")
            outcome = RunOutcome(0)
        else:
            outcome = execute_test(self.definition, self.options.repo_dir, self.deployment.test_environment, log_path)
        outcome.rebuilt = decision.rebuild
        return outcome

    def teardown(self) -> None:
        pass

    def finish(self) -> None:
        pass
