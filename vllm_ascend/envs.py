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
from collections.abc import Callable
from typing import Any


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment variable as ``1`` or ``0``.

    Only ``1`` and ``0`` are accepted (surrounding whitespace is ignored).
    Unset or empty values fall back to ``default``, which keeps
    ``export VAR=`` from crashing. Any other value raises ValueError naming the
    variable, instead of failing inside ``int()`` with an opaque message.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip()
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"Invalid value for environment variable {name}: {raw!r}. Expected '0' or '1'.")


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition


env_variables: dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: _bool_env("COMPILE_CUSTOM_KERNELS", True),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: _bool_env("VERBOSE", False),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # Maximum number of compiled ACLNN operator descriptors that the CANN
    # executor keeps cached per worker process. Valid values are positive
    # integers; unset keeps the CANN default (10000).
    # When the cache is full, the executor stops reusing descriptors and
    # recomputes tiling and workspace on the host for every call. In a torch
    # profile that shows up as a large `aclnn*GetWorkspaceSize` / `*_Tiling`
    # share, i.e. as host-side overhead rather than NPU time.
    # Model Runner V2 captures the target and the speculative prefill/decode
    # graphs separately, so it browses a larger operator/shape working set than
    # Model Runner V1. Qwen3.6-35B-A3B (TP=2, MTP=3, FULL_AND_PIECEWISE) was
    # observed holding about 9945 entries after startup, i.e. one warmup away
    # from the default ceiling.
    # The variable is exposed for diagnosis and tuning, not as a known fix: in
    # our own A/B runs on v0.30.0 (Qwen3.6-35B-A3B-w8a8, TP=2, seq=2048 and
    # seq=8192) raising the limit did not change the prefill wave time beyond
    # run-to-run noise, so measure before and after instead of assuming a win.
    # Raising the limit increases host (CPU) memory usage, because more
    # descriptors are retained per worker. CANN reads this variable when the
    # ACLNN executor is created, so it must be present in the worker process
    # environment before startup; it is not sensitive data.
    "ACLNN_CACHE_LIMIT": lambda: os.getenv("ACLNN_CACHE_LIMIT", None),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
