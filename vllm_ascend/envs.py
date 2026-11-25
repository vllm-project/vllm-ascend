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


def _hy4_bool_env(name: str, legacy: str, default: bool) -> bool:
    """Prefer the public control; retain the experiment's 1-only alias semantics."""
    if name in os.environ:
        return _bool_env(name, default)
    return os.getenv(legacy, "1" if default else "0") == "1"


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition


env_variables: dict[str, Callable[[], Any]] = {
    # HY4 target prep extra fusions; default False.
    # Values: 0/1. Not sensitive. Legacy alias: HY4_TARGET_PREP_EXTRA_FUSIONS.
    "VLLM_ASCEND_HY4_TARGET_PREP_EXTRA_FUSIONS": lambda: _hy4_bool_env(
        "VLLM_ASCEND_HY4_TARGET_PREP_EXTRA_FUSIONS", "HY4_TARGET_PREP_EXTRA_FUSIONS", False
    ),
    # HY4 engram triton hash; default True.
    # Values: 0/1. Not sensitive. Legacy alias: HY4_ENGRAM_TRITON_HASH.
    "VLLM_ASCEND_HY4_ENGRAM_TRITON_HASH": lambda: _hy4_bool_env(
        "VLLM_ASCEND_HY4_ENGRAM_TRITON_HASH", "HY4_ENGRAM_TRITON_HASH", True
    ),
    # HY4 engram batch hash; default True.
    # Values: 0/1. Not sensitive. Legacy alias: HY4_ENGRAM_BATCH_HASH.
    "VLLM_ASCEND_HY4_ENGRAM_BATCH_HASH": lambda: _hy4_bool_env(
        "VLLM_ASCEND_HY4_ENGRAM_BATCH_HASH", "HY4_ENGRAM_BATCH_HASH", True
    ),
    # HY4 engram hash graph; default True.
    # Values: 0/1. Not sensitive. Legacy alias: HY4_ENGRAM_HASH_GRAPH.
    "VLLM_ASCEND_HY4_ENGRAM_HASH_GRAPH": lambda: _hy4_bool_env(
        "VLLM_ASCEND_HY4_ENGRAM_HASH_GRAPH", "HY4_ENGRAM_HASH_GRAPH", True
    ),
    # HY4 draft prep graph; default True.
    # Values: 0/1. Not sensitive. Legacy alias: HY4_DRAFT_PREP_GRAPH.
    "VLLM_ASCEND_HY4_DRAFT_PREP_GRAPH": lambda: _hy4_bool_env(
        "VLLM_ASCEND_HY4_DRAFT_PREP_GRAPH", "HY4_DRAFT_PREP_GRAPH", True
    ),
    # HY4 dfly pageable engram; default False.
    # Values: 0/1. Not sensitive. Legacy alias: HY4_DFLY_PAGEABLE_ENGRAM.
    "VLLM_ASCEND_HY4_DFLY_PAGEABLE_ENGRAM": lambda: _hy4_bool_env(
        "VLLM_ASCEND_HY4_DFLY_PAGEABLE_ENGRAM", "HY4_DFLY_PAGEABLE_ENGRAM", False
    ),
    # Optional directory of per-device Mooncake JSON descriptors; default unset.
    # Not sensitive. The original HY4_MOONCAKE_LOCAL_COMM_RES_DIR remains an alias.
    "VLLM_ASCEND_HY4_MOONCAKE_LOCAL_COMM_RES_DIR": lambda: os.getenv(
        "VLLM_ASCEND_HY4_MOONCAKE_LOCAL_COMM_RES_DIR", os.getenv("HY4_MOONCAKE_LOCAL_COMM_RES_DIR")
    ),
    # Independent HY4 graph-external embedding/RoPE-hash preparation controls.
    # Valid values: 0/1; empty is disabled. Unset returns None and retains the
    # legacy HY4_TARGET_PREP_EXTRA_FUSIONS group switch (default disabled).
    # Explicit 0 disables that path even when the legacy group is enabled.
    # Not sensitive: no credentials or keys. D-only pure Decode guards apply.
    "VLLM_ASCEND_HY4_TARGET_EMBED_FUSION": lambda: (
        None
        if os.getenv("VLLM_ASCEND_HY4_TARGET_EMBED_FUSION") is None
        else _bool_env("VLLM_ASCEND_HY4_TARGET_EMBED_FUSION", False)
    ),
    "VLLM_ASCEND_HY4_TARGET_ROPE_HASH_FUSION": lambda: (
        None
        if os.getenv("VLLM_ASCEND_HY4_TARGET_ROPE_HASH_FUSION") is None
        else _bool_env("VLLM_ASCEND_HY4_TARGET_ROPE_HASH_FUSION", False)
    ),
    # Enable HY4 CPU-offloaded BF16 Engram row staging and native batch lookup.
    # Default: False. Valid values: 0/1. Not sensitive; no credentials or keys.
    "VLLM_ASCEND_HY4_ENGRAM_ROW_PREP": lambda: _bool_env("VLLM_ASCEND_HY4_ENGRAM_ROW_PREP", False),
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
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
