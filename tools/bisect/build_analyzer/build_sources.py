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
"""Select Dockerfile families for the build-chain analyzer."""

ROOT_DOCKERFILES = {
    "a2": "Dockerfile",
    "a3": "Dockerfile.a3",
    "310p": "Dockerfile.310p",
    "a5": "Dockerfile.a5",
}
NIGHTLY_DOCKERFILE_DIR = ".github/workflows/dockerfiles"


def normalize_build_soc(soc: str) -> str:
    """Map device variants to a Dockerfile family without changing case identity."""
    target = soc.lower()
    if target in ("910b", "ascend910b", "ascend910b1") or target.startswith("a2-"):
        target = "a2"
    elif target in ("950", "ascend950") or target.startswith("a5-"):
        target = "a5"
    elif target.startswith("a3-"):
        target = "a3"
    elif target.startswith("310p-"):
        target = "310p"
    if target not in ROOT_DOCKERFILES:
        raise ValueError(f"unsupported bisect SoC {soc!r}")
    return target
