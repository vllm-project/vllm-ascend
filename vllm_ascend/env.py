#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/worker.py
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
from typing import TYPE_CHECKING, Any, Callable

from vllm.envs import VLLM_USE_V1
from vllm.utils import update_environment_variables

if TYPE_CHECKING:
    RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES: str = "1"
    ACL_OP_INIT_MODE: str = "1"

npu_env_vars: dict[str, Callable[[], Any]] = {

    # Allowing set device at runtime when using ray backend
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES":
    lambda: os.getenv("RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES", "1"),

    # Fix the bug in torch 2.5.1 that raising segment fault when enable
    # `pin_memory` while creating a tensor using `torch.tensor`
    "ACL_OP_INIT_MODE":
    lambda: os.getenv("ACL_OP_INIT_MODE", "1"),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in npu_env_vars:
        return npu_env_vars[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(npu_env_vars.keys())


def load_npu_env_vars() -> None:
    env_vars: dict[str, str] = {}

    for k, v in npu_env_vars.items():
        env_vars[k] = str(v())
    if VLLM_USE_V1:
        env_vars["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    update_environment_variables(env_vars)
