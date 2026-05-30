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
# Check that vllm_ascend modules do not use init_logger(__name__).
#
# vllm's logging config registers a handler only for the "vllm" logger
# namespace.  Any logger created via init_logger(__name__) inside a
# vllm_ascend module ends up in the "vllm_ascend.*" namespace, which has
# no handler, so every log call is silently dropped.
#
# The correct pattern is:
#   from vllm.logger import logger
#

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PATCH_DIR = REPO_ROOT / "vllm_ascend"

PATTERN = re.compile(r"init_logger\s*\(\s*__name__\s*\)")


def main() -> None:
    violations = 0

    for filepath in sorted(PATCH_DIR.rglob("*.py")):
        with open(filepath, encoding="utf-8") as f:
            for linenum, line in enumerate(f, start=1):
                if PATTERN.search(line):
                    if violations == 0:
                        print()
                    print(f"  {filepath}:{linenum}: {line.rstrip()}")
                    violations += 1

    if violations > 0:
        print()
        print(f"Found {violations} violation(s): init_logger(__name__) must not be used in vllm_ascend modules.")
        print()
        print("vllm's logging handler is registered only for the 'vllm' namespace.")
        print("Loggers created with init_logger(__name__) inside vllm_ascend end up")
        print("in the 'vllm_ascend.*' namespace, which has no handler — all log")
        print("messages are silently dropped.")
        print()
        print("Fix: replace")
        print("   from vllm.logger import init_logger")
        print("   logger = init_logger(__name__)")
        print("with")
        print("   from vllm.logger import logger")
        sys.exit(1)


if __name__ == "__main__":
    main()
