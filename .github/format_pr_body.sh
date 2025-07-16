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
# Adapted from vllm/.github/scripts/cleanup_pr_body.sh

#!/bin/bash

set -eux

# ensure 2 argument is passed
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <pr_number> <vllm_version> <vllm_commit>"
    exit 1
fi

PR_NUMBER=$1
VLLM_VERSION=$2
VLLM_COMMIT=$3
OLD=/tmp/orig_pr_body.txt
NEW=/tmp/new_pr_body.txt
FINAL=/tmp/final_pr_body.txt

gh pr view --json body --template "{{.body}}" "${PR_NUMBER}" > "${OLD}"
cp "${OLD}" "${NEW}"

# Remove notes in pr description and add vLLM version and commit
sed -i '/<!--/,/-->/d' "${NEW}"
sed -i '/- vLLM .*$/d' "${NEW}"
{
    echo ""
    echo "- vLLM version: $VLLM_VERSION"
    echo "- vLLM main: $VLLM_COMMIT"
} >> "${NEW}"

# Remove redundant empty lines
uniq "${NEW}" > "${FINAL}"

# Run this only if ${NEW} is different than ${OLD}
if ! cmp -s "${OLD}" "${FINAL}"; then
    echo
    echo "Updating PR body:"
    echo
    cat "${NEW}"
    gh pr edit --body-file "${FINAL}" "${PR_NUMBER}"
else
    echo "No changes needed"
fi
