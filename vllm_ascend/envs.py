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
from typing import TYPE_CHECKING, Any, Callable, Dict

from vllm.envs import VLLM_USE_V1, is_set
from vllm.logger import init_logger
from vllm.utils import update_environment_variables

if TYPE_CHECKING:
    RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES: str = "1"
    ACL_OP_INIT_MODE: str = "1"
    MAX_JOBS = None
    CMAKE_BUILD_TYPE = None
    COMPILE_CUSTOM_KERNELS = None
    VERBOSE = None
    ASCEND_HOME_PATH = None
    LD_LIBRARY_PATH = None

logger = init_logger(__name__)

env_variables: Dict[str, Callable[[], Any]] = {

    # Allowing SetDevice at runtime when using ray backend
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES":
    lambda: os.getenv("RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES", "1"),

    # Fix the bug in torch 2.5.1 that raising segment fault when enable
    # `pin_memory` while creating a tensor using `torch.tensor`
    "ACL_OP_INIT_MODE":
    lambda: os.getenv("ACL_OP_INIT_MODE", "1"),

    # max compile thread num
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),
    "COMPILE_CUSTOM_KERNELS":
    lambda: os.getenv("COMPILE_CUSTOM_KERNELS", None),

    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),
    "ASCEND_HOME_PATH":
    lambda: os.getenv("ASCEND_HOME_PATH", None),
    "LD_LIBRARY_PATH":
    lambda: os.getenv("LD_LIBRARY_PATH", None),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())


def load_env_variables() -> None:
    """
    Set Ascend related environment variables when NPU platform registered.
    NOTE: To make `ACL_OP_INIT_MODE` setting valid, you need to use pta of 0403.
    To install pta of 0403 version, replace the link in `pta_install.sh` with 
    https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.5.1-7.0.0/20250403.3/pytorch_v2.5.1-7.0.0_py310.tar.gz
    """
    env_vars: dict[str, str] = {}

    for name, value in env_variables.items():
        # If this env var is not explicitly set by the user, then set it
        if not is_set(name):
            env_vars[name] = str(value())
            logger.info("Set environment variables: %s = %s", name, value())

    if VLLM_USE_V1:
        env_vars["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        logger.info("Set environment variables: \
                    VLLM_WORKER_MULTIPROC_METHOD = spawn")

    update_environment_variables(env_vars)
