#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

# Usage: bash stage_ci_actions.sh [source-root]
# Defaults to GITHUB_WORKSPACE; an explicit source root may be relative to cwd.
set -euo pipefail

source_root="${1:-$GITHUB_WORKSPACE}"
if [[ "$(realpath "$GITHUB_WORKSPACE/../../_temp")" != \
      "$(realpath "$RUNNER_TEMP")" ]]; then
    echo "::error::Runner directory layout changed; local action path needs updating."
    exit 1
fi

mkdir -p "$RUNNER_TEMP/ci-workflow"
cp -a "$source_root/.github" "$RUNNER_TEMP/ci-workflow/"
