#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from __future__ import annotations

import functools
import sys
from importlib.metadata import version as get_distribution_version

from packaging.version import InvalidVersion, Version

import vllm_ascend.envs as envs_ascend


@functools.cache
def vllm_version_is(target_vllm_version: str) -> bool:
    if envs_ascend.VLLM_VERSION is not None:
        vllm_version = envs_ascend.VLLM_VERSION
    else:
        vllm_module = sys.modules.get("vllm")
        if vllm_module is not None and hasattr(vllm_module, "__version__"):
            vllm_version = vllm_module.__version__
        else:
            vllm_version = get_distribution_version("vllm")
    try:
        return Version(vllm_version).public == Version(target_vllm_version).public
    except InvalidVersion:
        raise ValueError(
            f"Invalid vllm version {vllm_version} found. A dev version of vllm "
            "is installed probably. Set the environment variable VLLM_VERSION "
            "to control it by hand. And please make sure the value follows the "
            "format of x.y.z."
        )
