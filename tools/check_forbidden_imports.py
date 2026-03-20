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
# Adapted from https://github.com/vllm-project/vllm/tree/main/tools
#

import sys
from dataclasses import dataclass, field

import regex as re


@dataclass
class ForbiddenImport:
    pattern: str
    tip: str
    allowed_pattern: re.Pattern = re.compile(r"^$")
    allowed_files: set[str] = field(default_factory=set)


CHECK_IMPORTS = {
    "pickle/cloudpickle": ForbiddenImport(
        pattern=(
            r"^\s*(import\s+(pickle|cloudpickle)(\s|$|\sas)"
            r"|from\s+(pickle|cloudpickle)\s+import\b)"
        ),
        tip=("Avoid using pickle or cloudpickle or add this file to tools/check_forbidden_imports.py."),
        allowed_files={
            "vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/metadata.py",
        },
    ),
    "re": ForbiddenImport(
        pattern=r"^\s*(?:import\s+re(?:$|\s|,)|from\s+re\s+import)",
        tip="Replace 'import re' with 'import regex as re' or 'import regex'.",
        allowed_pattern=re.compile(r"^\s*import\s+regex(\s*|\s+as\s+re\s*)$"),
    ),
    "triton": ForbiddenImport(
        pattern=r"^(from|import)\s+triton(\s|\.|$)",
        tip=(
            "Use 'from vllm.triton_utils import triton'/'tl' or "
            "'from vllm_ascend.ops.triton.triton_utils import ...' instead."
        ),
        allowed_pattern=re.compile(
            r"^\s*import\s+triton\.language\.extra\.cann\.extension\s+as\s+_extension_module(\s+#.*)?$"
        ),
    ),
}


def check_file(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return []

    return_code = 0
    for import_name, forbidden_import in CHECK_IMPORTS.items():
        if path in forbidden_import.allowed_files:
            continue

        for match in re.finditer(forbidden_import.pattern, content, re.MULTILINE):
            if forbidden_import.allowed_pattern.match(match.group()):
                continue

            line_num = content[: match.start() + 1].count("\n") + 1
            print(
                f"{path}:{line_num}: "
                "\033[91merror:\033[0m "
                f"Found forbidden import: {import_name}. {forbidden_import.tip}"
            )
            return_code = 1

    return return_code


def main() -> int:
    return_code = 0
    for path in sys.argv[1:]:
        return_code |= check_file(path)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
