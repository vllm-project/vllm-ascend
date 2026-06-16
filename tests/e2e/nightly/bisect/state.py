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
"""Persist bisect progress so a preempted/timed-out run can resume."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class BisectState:
    """Serializable search state: window + cached verdicts by commit."""

    lo: int = 0
    hi: int = 0
    round_idx: int = 0
    # commit sha -> "PASS"/"FAIL"/"SKIP" (lets resume skip re-running commits)
    verdicts: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BisectState | None":
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            lo=data.get("lo", 0),
            hi=data.get("hi", 0),
            round_idx=data.get("round_idx", 0),
            verdicts=data.get("verdicts", {}),
        )
