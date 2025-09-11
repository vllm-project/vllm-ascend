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
import subprocess
import sys


def convert_to_lf(filepath: str):
    """replace all the end of lines as LF (\n)."""
    try:
        with open(filepath, "rb") as f:
            content = f.read()
        new_content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

        if new_content != content:
            with open(filepath, "wb") as f:
                f.write(new_content)
            print(f"[FIXED] {filepath}")
    except Exception as e:
        print(f"[SKIP] {filepath} ({e})")


def find_crlf_files():
    result = subprocess.run(["git", "grep", "-I", "-l", "\r", "."],
                            capture_output=True,
                            text=True,
                            check=False)
    files = result.stdout.strip().splitlines()
    if files:
        print("Files with CRLF line endings:")
        for file in files:
            print(f"- {file}")
            convert_to_lf(file)
        sys.exit(1)
    print("✅ all files have LF line endings.")
    sys.exit(0)


if __name__ == "__main__":
    find_crlf_files()
