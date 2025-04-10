#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This fill is mainly Adapted from vllm-project/vllm/vllm/envs.py
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

env_variables: Dict[str, Callable[[], Any]] = {
    # max compile thread num
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    "COMPILE_CUSTOM_KERNELS":
    lambda: os.getenv("COMPILE_CUSTOM_KERNELS", None),

    # If set, vllm-ascend will print verbose logs during compliation
    "VERBOSE": lambda: bool(int(os.getenv('VERBOSE', '0'))),
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    "LD_LIBRARY_PATH": lambda: os.getenv("LD_LIBRARY_PATH", None),

    # Allowing SetDevice at runtime when using ray backend.
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES":
    lambda: os.getenv("RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES", "1"),

    # Fix the bug in torch 2.5.1 that raising segment fault when enable
    # `pin_memory` while creating a tensor using `torch.tensor`.
    "ACL_OP_INIT_MODE":
    lambda: os.getenv("ACL_OP_INIT_MODE", "1"),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
