#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
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
#

import os
from typing import Any, Callable, Dict

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: Dict[str, Callable[[], Any]] = {
    # max compile thread num
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),
    "COMPILE_CUSTOM_KERNELS":
    lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    "SOC_VERSION":
    lambda: os.getenv("SOC_VERSION", "ASCEND910B1"),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),
    "ASCEND_HOME_PATH":
    lambda: os.getenv("ASCEND_HOME_PATH", None),
    "LD_LIBRARY_PATH":
    lambda: os.getenv("LD_LIBRARY_PATH", None),
    # Used for disaggregated prefilling
    "HCCN_PATH":
    lambda: os.getenv("HCCN_PATH", "/usr/local/Ascend/driver/tools/hccn_tool"),
    "HCCL_SO_PATH":
    lambda: os.environ.get("HCCL_SO_PATH", None),
    "PROMPT_DEVICE_ID":
    lambda: os.getenv("PROMPT_DEVICE_ID", None),
    "DECODE_DEVICE_ID":
    lambda: os.getenv("DECODE_DEVICE_ID", None),
    "LLMDATADIST_COMM_PORT":
    lambda: os.getenv("LLMDATADIST_COMM_PORT", "26000"),
    "LLMDATADIST_SYNC_CACHE_WAIT_TIME":
    lambda: os.getenv("LLMDATADIST_SYNC_CACHE_WAIT_TIME", "5000"),
    "CXX_COMPILER":
    lambda: os.getenv("CXX_COMPILER", None),
    "C_COMPILER":
    lambda: os.getenv("C_COMPILER", None),
    "VLLM_VERSION":
    lambda: os.getenv("VLLM_VERSION", None),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
