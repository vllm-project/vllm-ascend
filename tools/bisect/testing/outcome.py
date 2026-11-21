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
"""Raw signals returned by one test execution."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunOutcome:
    """Raw signals produced by a runner for one trial."""

    exit_code: int
    # Directory holding this trial's benchmark JSON(s). A whole-YAML run writes
    # one file per case, so we scan the directory rather than a single file.
    results_dir: Path | None = None
    infra_error: bool = False  # set when the failure is environmental (-> SKIP)
    skip_reason: str = ""  # human reason when infra_error is set (e.g. vllm skew)
    rebuilt: bool = False
