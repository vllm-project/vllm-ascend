#!/usr/bin/env python3
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

import os
import subprocess
import sys
from pathlib import Path


def is_git_ignored(file_path):
    """Check if a file is ignored by git."""
    try:
        result = subprocess.run(['git', 'check-ignore', '-q',
                                 str(file_path)],
                                capture_output=True,
                                check=False)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        # If git is not available or command fails, assume not ignored
        return False


def check_excalidraw_metadata(file_path):
    """Check if a PNG file has excalidraw metadata embedded."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            return b'excalidraw+json' in content
    except (IOError, OSError):
        return False


def main():
    """Main function to check all excalidraw PNG files."""
    errors = []

    # Find all .excalidraw.png files
    for root, dirs, files in os.walk('.'):
        # Skip .git directory
        if '.git' in root:
            continue

        for file in files:
            if file.lower().endswith('.excalidraw.png'):
                file_path = Path(root) / file

                # Skip if git-ignored
                if is_git_ignored(file_path):
                    continue

                # Check for excalidraw metadata
                if not check_excalidraw_metadata(file_path):
                    errors.append(str(file_path))
                    print(
                        f"{file_path} was not exported from excalidraw with 'Embed Scene' enabled."
                    )

    if errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
