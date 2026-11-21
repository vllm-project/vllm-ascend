#
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
#

"""Triton execution lines are part of the precision coverage map."""

from __future__ import annotations

import json
from pathlib import Path

import test_selector


def test_triton_sidecar_lines_enter_the_coverage_map(tmp_path: Path) -> None:
    case_dir = tmp_path / "tests__ut__ops__test_kernel" / "covdata"
    case_dir.mkdir(parents=True)
    (case_dir / "triton_lines.json").write_text(
        json.dumps({"ops/triton/kernel.py": [24, 26]}),
        encoding="utf-8",
    )

    selector = test_selector.CoverageSelector(str(tmp_path))
    selector.build_test_case_map()

    case = selector.test_case_map["tests/ut/ops/test_kernel.py"]
    assert case["files"] == {"ops/triton/kernel.py": {24, 26}}
