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
"""Pure candidate selection and boundary updates; no deployment or test dependencies."""

from dataclasses import dataclass, field


@dataclass
class BisectSearch:
    lo: int
    hi: int
    skipped: set[int] = field(default_factory=set)

    def next_index(self) -> int | None:
        mid = (self.lo + self.hi) // 2
        for offset in range(self.hi - self.lo):
            for index in (mid + offset, mid - offset):
                if self.lo <= index < self.hi and index not in self.skipped:
                    return index
        return None

    def record(self, index: int, verdict: str) -> None:
        if not self.lo <= index < self.hi or verdict not in {"PASS", "FAIL", "SKIP"}:
            raise ValueError("Invalid search observation")
        if verdict == "FAIL":
            self.hi = index
        elif verdict == "PASS":
            self.lo = index + 1
        else:
            self.skipped.add(index)
