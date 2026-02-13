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

set -euo pipefail

function usage() {
  echo "This script collects the first contributions of new users in a given GitHub repository."
  echo "Please set the environment variable GITHUB_TOKEN with repo read permission."
  echo "Refer to https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api?apiVersion=2022-11-28"
  echo ""
  echo "Usage: $0 owner/repo base_sha head_sha"
  echo "Example: $0 myorg/myrepo abcdef1 ghijk23"
}

if [ "$#" -ne 3 ]; then
  usage
  exit 1
fi

REPO="$1"    # Example: myorg/myrepo
BASE="$2"    # older commit (exclusive)
HEAD="$3"    # newer commit (inclusive)
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

if [ -z "$GITHUB_TOKEN" ]; then
  echo "Please set the environment variable GITHUB_TOKEN with repo read permission."
  echo "Refer to https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api?apiVersion=2022-11-28"
  exit 1
fi

# All commits in the time order, format: sha|email|name|date
ALLCOMMITS=$(mktemp)
git log --pretty=format:'%H|%aE|%aN|%cI' --reverse --all > "$ALLCOMMITS"

# Get the first commit in this repo for each author email
FIRST_BY_EMAIL=$(mktemp)
awk -F'|' '!seen[$2]++ {print $2 "|" $1 "|" $4 "|" $3}' "$ALLCOMMITS" > "$FIRST_BY_EMAIL"
# format: email|first_sha|date|name

# Get the commit sha list in the range (base..head)
RANGE_SHAS=$(mktemp)
git rev-list "${BASE}..${HEAD}" > "$RANGE_SHAS"

echo "New user's first commit in ${REPO} (${BASE}..${HEAD})"
echo

# Get each email's first commit, if the first commit sha is in the range, query GitHub mapping and output
while IFS='|' read -r email sha date name; do
  if grep -Fxq "$sha" "$RANGE_SHAS"; then
    # Using GitHub API to check commit's author.login (if exists)
    api_url="https://api.github.com/repos/${REPO}/commits/${sha}"
    resp=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" -H "Accept: application/vnd.github.v3+json" "$api_url")
    login=$(echo "$resp" | jq -r '.author.login // empty')
    # fallback to name and email if login not found
    if [ -n "$login" ]; then
      user_display="@${login}"
    else
      user_display="$(echo "$name" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//') <${email}>"
    fi
    echo "- ${user_display} — \`${sha}\` — ${date}"
  fi
done < "$FIRST_BY_EMAIL"

# cleanup
rm -f "$ALLCOMMITS" "$FIRST_BY_EMAIL" "$RANGE_SHAS"