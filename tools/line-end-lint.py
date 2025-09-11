#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This script checks that all the lines end with LF in the repository.
#!/usr/bin/env python3
import os
import sys

SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__'}

modified_files = []


def convert_to_lf(filepath: str):
    """replace all the end of lines as LF (\n)."""
    try:
        with open(filepath, "rb") as f:
            content = f.read()
        new_content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

        if new_content != content:
            with open(filepath, "wb") as f:
                f.write(new_content)
            modified_files.append(filepath)
            print(f"[FIXED] {filepath}")
    except Exception as e:
        print(f"[SKIP] {filepath} ({e})")


def check_and_fix_repo(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            convert_to_lf(filepath)


if __name__ == "__main__":
    repo_root = os.getcwd()
    check_and_fix_repo(repo_root)

    if modified_files:
        print("❌ some files were reformatted.")
        sys.exit(1)
    else:
        print("✅ all files already use LF, lint passed.")
        sys.exit(0)
