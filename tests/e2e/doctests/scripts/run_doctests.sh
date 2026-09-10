#!/bin/bash

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

set -Eeuo pipefail

DOCTEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Print the supported doctest commands and arguments.
function usage() {
  echo "Usage:"
  echo "  $0 quickstart {a2|310p}"
  echo "  $0 installation {pip|uv|source}"
}

[[ $# -eq 2 ]] || { usage; exit 1; }

case "$1:$2" in
  quickstart:a2|quickstart:310p)
    worker=001-quickstart-test.sh
    ;;
  installation:pip|installation:uv|installation:source)
    worker=002-installation-test.sh
    ;;
  *)
    usage
    exit 1
    ;;
esac

exec bash "${DOCTEST_DIR}/${worker}" "$2"
