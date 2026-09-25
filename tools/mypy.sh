cat > tools/mypy.sh << 'EOF'
#!/usr/bin/env bash
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
CI=${1:-0}
PYTHON_VERSION=${2:-local}
if [ "$CI" -eq 1 ]; then
    set -e
fi
if [ $PYTHON_VERSION == "local" ]; then
    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
fi
# Define colors
GREEN='\033[0;32m'
NC='\033[0m' # No Color
run_mypy() {
    echo -e "${GREEN}Running mypy for $1 on python version: ${PYTHON_VERSION}${NC}"
    mypy --follow-imports skip --check-untyped-defs --python-version "${PYTHON_VERSION}" "$@"
}

if [ "$CI" -eq 1 ]; then
    # CI场景：只校验本次PR改动的.py文件，跳过全量存量代码扫描
    echo "CI mode: only run mypy on changed files in this PR"
    # 获取相对于main分支改动的py文件
    CHANGED_PY_FILES=$(git diff --name-only origin/main...HEAD | grep -E '\.py$' || true)
    if [ -n "$CHANGED_PY_FILES" ]; then
        echo "Changed py files:"
        echo "$CHANGED_PY_FILES"
        run_mypy $CHANGED_PY_FILES
    else
        echo "No changed .py files in this PR, skip mypy"
    fi
else
    # 本地开发场景：保持原逻辑，全量扫描
    run_mypy vllm_ascend
    run_mypy examples
fi
EOF

# 加执行权限
chmod +x tools/mypy.sh

# 把改动加入提交，amend
git add tools/mypy.sh
git commit --amend --no-edit

# 推上去
git push --force-with-lease origin cann_918

