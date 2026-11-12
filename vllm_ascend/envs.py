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


def _strict_binary_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be either '0' or '1', got {value!r}")
    return value == "1"


def _positive_int_env(name: str, default: str = "1") -> int:
    value = int(os.getenv(name, default))
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


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
    # Number of reusable 2 MiB-aligned NPU staging chunks used when Mooncake
    # transfers into DSA PD CPU-offloaded KV buffers. Default: 16; valid: >= 1.
    # This setting is not sensitive.
    "VLLM_ASCEND_DSA_PD_MOONCAKE_STAGING_WINDOW": lambda: max(
        1, int(os.getenv("VLLM_ASCEND_DSA_PD_MOONCAKE_STAGING_WINDOW", "16"))
    ),
    # Pipelining of DSA PD Mooncake staged reads. 1 = the staging window
    # (VLLM_ASCEND_DSA_PD_MOONCAKE_STAGING_WINDOW, unchanged total capacity)
    # is split into BANKS banks that network reads round-robin between, and
    # each bank lands on its own dedicated NPU stream so banks' copies can
    # execute in parallel. A bank is only reclaimed, via its completion event,
    # when the pipeline wraps back to it BANKS reads later, plus a final
    # drain before the receive completes. The network batch shrinks to
    # window/BANKS chunks and the staging footprint is identical with the
    # pipeline on or off. BANKS=1 degenerates to per-window landing waits.
    # 0 keeps the legacy synchronous landing path (kill switch). Valid
    # values: 0 or 1. Not sensitive.
    "VLLM_ASCEND_DSA_PD_MOONCAKE_PIPELINE": lambda: _strict_binary_env("VLLM_ASCEND_DSA_PD_MOONCAKE_PIPELINE"),
    # Number of staging banks, landing streams, and Host workers in pipeline
    # mode. Default: 4; valid: integers >= 1 (clamped to the staging window).
    # Each effective bank has one worker and keeps window/BANKS staging slots.
    # A single effective bank uses the caller thread without a worker pool.
    # Pipeline-off, overlapping destinations, and READ_PROBE use serial Host
    # submission. Workers join and landing copies drain before request unlock.
    # Total staging capacity and cache budget stay unchanged. Not sensitive.
    "VLLM_ASCEND_DSA_PD_MOONCAKE_BANKS": lambda: max(1, int(os.getenv("VLLM_ASCEND_DSA_PD_MOONCAKE_BANKS", "4"))),
    # Landing copy streams per staging bank. Default: 1; valid: integers >= 1,
    # capped by slots per bank. Read at receive boundaries; does not add Host
    # workers or change READ batches, staging capacity, or view-cache capacity.
    # Pipeline-off and READ_PROBE use one stream. Overlapping copies within a
    # batch stay on the bank's first stream. Auxiliary completion events join
    # that stream before the reusable bank event protects staging reclamation.
    # This setting is not sensitive.
    "VLLM_ASCEND_DSA_PD_MOONCAKE_COPY_STREAMS_PER_BANK": lambda: _positive_int_env(
        "VLLM_ASCEND_DSA_PD_MOONCAKE_COPY_STREAMS_PER_BANK"
    ),
    # Opt-in probe for the staged-read timing blind spot: after each engine
    # batch read returns, a device-wide torch.npu.synchronize() reveals how
    # much later the network data actually becomes visible on the device
    # (read_settle_s in the receive timing line). 0 = off. Measurement runs
    # only - the device-wide sync defeats the pipelining. Valid values: 0 or
    # 1. Not sensitive.
    "VLLM_ASCEND_DSA_PD_MOONCAKE_READ_PROBE": lambda: _strict_binary_env("VLLM_ASCEND_DSA_PD_MOONCAKE_READ_PROBE"),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
