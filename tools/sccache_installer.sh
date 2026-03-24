#!/usr/bin/env bash

#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

set -e

SCCACHE_VERSION="v0.14.0"

echo "MATRIX_ARCH=${MATRIX_ARCH}"
case "${MATRIX_ARCH}" in
    linux/arm64) SCCACHE_ARCH="aarch64" ;;
    linux/amd64) SCCACHE_ARCH="x86_64" ;;
    *) echo "Unsupported TARGETPLATFORM for sccache: ${MATRIX_ARCH}" >&2; exit 1 ;;
esac

echo "SCCACHE_ARCH=${SCCACHE_ARCH}"
SCCACHE_PACKAGE="sccache-${SCCACHE_VERSION}-${SCCACHE_ARCH}-unknown-linux-musl"
SCCACHE_DOWNLOAD_URL="${SCCACHE_DOWNLOAD_URL:-https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_PACKAGE}.tar.gz}"
echo "SCCACHE_DOWNLOAD_URL=${SCCACHE_DOWNLOAD_URL}"

curl -L -o /tmp/sccache.tar.gz "${SCCACHE_DOWNLOAD_URL}"
tar -xzf /tmp/sccache.tar.gz -C /tmp
mv "/tmp/${SCCACHE_PACKAGE}/sccache" /usr/bin/sccache
chmod +x /usr/bin/sccache
rm -rf /tmp/sccache.tar.gz "/tmp/${SCCACHE_PACKAGE}"

export ACTIONS_RESULTS_URL=$(cat /run/secrets/ACTIONS_RESULTS_URL 2>/dev/null || echo "")
export ACTIONS_RUNTIME_TOKEN=$(cat /run/secrets/ACTIONS_RUNTIME_TOKEN 2>/dev/null || echo "")
export ACTIONS_CACHE_SERVICE_V2=on
export SCCACHE_GHA_ENABLED=${SCCACHE_GHA_ENABLED}

sccache --start-server
sccache --show-stats
