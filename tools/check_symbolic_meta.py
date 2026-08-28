#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

DEFAULT_PATH: Final = Path("csrc/torch_binding_meta.cpp")
EXEMPTION_MARKER: Final = "symbolic-meta-ok:"
HIGH_RISK_PATTERNS: Final = (
    (".size(", "use tensor.sym_size(i) for tensor-derived output dimensions"),
    (".sizes(", "use tensor.sym_sizes() for tensor-derived output shapes"),
    (".numel(", "avoid numel() checks in meta kernels for symbolic tensors"),
    ("at::empty({", "use at::empty_symint(...) with c10::SymDimVector"),
    ("torch::empty({", "use at::empty_symint(...) with c10::SymDimVector"),
    ("at::zeros({", "use at::empty_symint(...) with c10::SymDimVector"),
    ("torch::zeros({", "use at::empty_symint(...) with c10::SymDimVector"),
    ("empty_symint({", "use empty_symint(c10::SymDimVector{...}, ...) to avoid shape type ambiguity"),
    ("std::vector<int64_t>", "use c10::SymDimVector"),
    ("at::SmallVector<int64_t", "use at::SmallVector<c10::SymInt, N> or c10::SymDimVector"),
    ("c10::SmallVector<int64_t", "use at::SmallVector<c10::SymInt, N> or c10::SymDimVector"),
)


def has_exemption(lines: list[str], line_number: int) -> bool:
    line_index = line_number - 1
    candidates = [lines[line_index]]
    if line_index > 0:
        candidates.append(lines[line_index - 1])

    for candidate in candidates:
        marker_index = candidate.find(EXEMPTION_MARKER)
        if marker_index == -1:
            continue
        reason = candidate[marker_index + len(EXEMPTION_MARKER) :].strip()
        if reason:
            return True
    return False


def check_file(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []

    violations: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if has_exemption(lines, line_number):
            continue
        for pattern, suggestion in HIGH_RISK_PATTERNS:
            if pattern in line:
                violations.append(f"{path}:{line_number}: uses `{pattern}`; {suggestion}")
                break
    return violations


def main() -> int:
    paths = tuple(Path(arg) for arg in sys.argv[1:]) or (DEFAULT_PATH,)
    violations: list[str] = []

    for path in paths:
        violations.extend(check_file(path))

    if not violations:
        return 0

    print("Meta kernels should preserve symbolic shapes.")
    print("Use sym_size/sym_sizes and empty_symint(c10::SymDimVector{...}, ...) for tensor-derived output shapes.")
    print("If concrete shape use is required, add `symbolic-meta-ok:` with a reason.\n")
    for violation in violations:
        print(f"  {violation}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
