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
# Restore public package sources after an internal-mirror build.
#
# Internal CI builds pass cluster-internal mirrors via build-args
# (PIP_INDEX_URL, GIT_PROXY, CRATES_IO_INDEX) to accelerate downloads. Those
# mirrors are unreachable from the published image, so this script resets
# pip/cargo/git back to public defaults before the image is finalized.
#
# If you build with the default build-args (all public / empty), every branch
# below is skipped and the script is a no-op:
#   - pip: reset to PUBLIC_PIP_INDEX_URL only when PIP_INDEX_URL was overridden.
#   - cargo/git: clean up only what internal builds configured.
set -euo pipefail

# pip: reset to the public index when an internal index was used. Users who
# override PIP_INDEX_URL with their own public mirror should also override
# PUBLIC_PIP_INDEX_URL to keep that mirror in the final image.
if [ -n "${PIP_INDEX_URL:-}" ] && [ "${PIP_INDEX_URL}" != "${PUBLIC_PIP_INDEX_URL:-}" ]; then
    pip config set global.index-url "${PUBLIC_PIP_INDEX_URL}"
    pip config unset global.trusted-host 2>/dev/null || true
fi

# cargo/git: remove the cargo config, the build-time registry cache (which may
# embed the internal mirror URL), and the git insteadOf rewrites that internal
# builds configured.
if [ -n "${GIT_PROXY:-}" ] || [ -n "${CRATES_IO_INDEX:-}" ]; then
    rm -f "${HOME}/.cargo/config.toml"
    rm -rf "${HOME}/.cargo/registry" "${HOME}/.cargo/.global-cache"
    for key in $(git config --global --name-only --get-regexp '^url\..*\.insteadof$' 2>/dev/null || true); do
        git config --global --unset-all "$key"
    done
fi
